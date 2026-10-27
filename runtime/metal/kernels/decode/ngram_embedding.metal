#include "metal/abi/KernelABI.h"
#include "metal/abi/NgramEmbedding.h"

#include <metal_stdlib>

using namespace metal;

// The n-gram embedding's verify-step kernels (metal/abi/NgramEmbedding.h),
// Qwen3.8-Flash-Next's PLE: the GPU hash of each lane's verify rows into
// table rows, and the history commit after a lane's retained rows. The gate
// and convolution the same step runs live with the prefill kernels.
// Ported from the vendored runtime/qwen4/metal/kernels/shared/
// ngram_embedding.metal.

constant constexpr uint kNgramRowThreads = RICHENGINE_HYPER_CONNECTION_ROW_THREADS;

// The history after a lane's `rows` retained rows: the history before them,
// then the convolution inputs; threadgroup (j, y) writes row j's channels
// from y x kNgramRowThreads on. Same body as the prefill file's
// ngram_history_rows.
inline void ngram_commit_rows(device const bfloat *input,
                              device const bfloat *history_in,
                              device bfloat *history_out,
                              constant NgramEmbeddingParams &p, uint rows,
                              uint2 group, uint tid) {
  const uint width = p.streams * p.hidden;
  const uint history = (p.taps - 1) * p.dilation;
  const uint c = group.y * kNgramRowThreads + tid, j = group.x;
  if (c >= width || j >= history) return;
  const int source = int(rows) - int(history) + int(j);
  history_out[ulong(j) * width + c] =
      source >= 0 ? input[ulong(source) * width + c]
                  : history_in[ulong(int(history) + source) * width + c];
}

// The history after a lane's retained rows of a verify step, a count the GPU
// wrote.
kernel void ngram_embedding_commit(device const bfloat *input [[buffer(0)]],
                                   device const bfloat *history_in
                                   [[buffer(1)]],
                                   device bfloat *history_out [[buffer(2)]],
                                   device const uint *retained [[buffer(3)]],
                                   constant NgramEmbeddingParams &p
                                   [[buffer(4)]],
                                   uint2 group
                                   [[threadgroup_position_in_grid]],
                                   uint tid [[thread_index_in_threadgroup]]) {
  ngram_commit_rows(input, history_in, history_out, p, retained[0], group,
                    tid);
}

// The table rows of each lane's verify rows (the CPU hash of a prefill
// chunk's rows is model/Qwen4Exp.cpp qwen4ExpPleRows): heads of the 2- to
// ngram-gram hashes of the row's token and those before it, cut at EOS or
// the sequence's start. Lane l's rows are packed at row index
// l * p.row_stride, and `before` holds the ngram - 1 tokens before a lane's
// first row, nearest first.
kernel void ngram_embedding_hash(device const uint *tokens [[buffer(0)]],
                                 device const uint *before [[buffer(1)]],
                                 device uint *indices [[buffer(2)]],
                                 constant NgramHashParams &p [[buffer(3)]],
                                 uint index [[thread_position_in_grid]]) {
  if (index >= p.rows) return;
  const uint lane = index / p.row_stride, r = index % p.row_stride;
  // The row's token, then the ngram - 1 tokens before it, EOS from a cut on.
  ulong context[NGRAM_EMBEDDING_MAXIMUM_NGRAM] = {tokens[index]};
  bool cut = false;
  for (uint back = 1; back < p.ngram; ++back) {
    const uint previous = back <= r
                              ? tokens[index - back]
                              : before[lane * (p.ngram - 1) + (back - r - 1)];
    cut = cut || previous == NGRAM_EMBEDDING_NONE || previous == p.eos;
    context[back] = cut ? p.eos : previous;
  }
  const uint heads = (p.ngram - 1) * p.heads_per_ngram;
  for (uint gram = 2; gram <= p.ngram; ++gram) {
    ulong mixed = context[0] * p.multipliers[0];
    for (uint j = 1; j < gram; ++j) mixed ^= context[j] * p.multipliers[j];
    for (uint g = 0; g < p.heads_per_ngram; ++g) {
      const uint head = (gram - 2) * p.heads_per_ngram + g;
      indices[ulong(index) * heads + head] =
          uint(mixed % p.vocabularies[head] + p.offsets[head]);
    }
  }
}
