#pragma once

#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/activation.h"
#include "metal/kernels/common/rms_inverse.h"

// Four-tap causal convolution of one channel at one token of the command,
// reading the three preceding tokens from the carried state, rounded to bf16
// and gated by SiLU.
inline bfloat gdn_conv_silu(device const bfloat *packed,
                            device const bfloat *conv_state_in,
                            device const bfloat *conv_weights,
                            uint packed_width, uint conv_dim, uint token,
                            uint channel) {
  float value = 0.0f;
  for (uint tap = 0; tap < 4; ++tap) {
    uint position = token + tap;
    bfloat input = position < 3
                       ? conv_state_in[position * conv_dim + channel]
                       : packed[(position - 3) * packed_width + channel];
    value += float(input) * float(conv_weights[channel * 4 + tap]);
  }
  value = float(bfloat(value));
  return bfloat(richengine_silu(value));
}

// Row `row` of the carried state after consumed_tokens: the last three inputs
// seen, still taken from the incoming state when fewer were consumed.
inline bfloat gdn_conv_carry(device const bfloat *packed,
                             device const bfloat *conv_state_in,
                             uint packed_width, uint conv_dim,
                             uint consumed_tokens, uint row, uint channel) {
  uint source = consumed_tokens + row;
  return source < 3 ? conv_state_in[source * conv_dim + channel]
                    : packed[(source - 3) * packed_width + channel];
}

// The row `hops` parents above `row` in one lane's node table (tree verify):
// the convolution and carry of a branch row read their predecessors along
// the node's path, not the adjacent DFS rows.
inline uint gdn_tree_ancestor(device const uint *nodes, uint row, uint hops) {
  for (uint i = 0; i < hops; ++i)
    row = RICHENGINE_TREE_NODE_PARENT(nodes[row]);
  return row;
}

// The tree counterpart of gdn_conv_silu: the node's four taps cover path
// positions depth-3..depth, taken from the carried state below zero and
// from the ancestor row above it.
inline bfloat gdn_conv_silu_tree(device const bfloat *packed,
                                 device const bfloat *conv_state_in,
                                 device const bfloat *conv_weights,
                                 uint packed_width, uint conv_dim, uint row,
                                 uint channel, device const uint *nodes) {
  const uint depth = RICHENGINE_TREE_NODE_DEPTH(nodes[row]);
  float value = 0.0f;
  for (uint tap = 0; tap < 4; ++tap) {
    const int position = int(depth) + int(tap);
    const bfloat input =
        position < 3
            ? conv_state_in[uint(position) * conv_dim + channel]
            : packed[ulong(gdn_tree_ancestor(nodes, row, 3 - tap)) *
                         packed_width +
                     channel];
    value += float(input) * float(conv_weights[channel * 4 + tap]);
  }
  value = float(bfloat(value));
  return bfloat(richengine_silu(value));
}

// The tree counterpart of gdn_conv_carry: the consumed prefix is the
// retained path's DFS rows, so path[p] names the row at path position p.
inline bfloat gdn_conv_carry_tree(device const bfloat *packed,
                                  device const bfloat *conv_state_in,
                                  uint packed_width, uint conv_dim,
                                  device const uint *path, uint consumed,
                                  uint row, uint channel) {
  const uint source = consumed + row;
  return source < 3
             ? conv_state_in[source * conv_dim + channel]
             : packed[ulong(path[source - 3]) * packed_width + channel];
}

// The gates of one (token, value head): beta = sigmoid(b) and
// decay = exp(a_scale * softplus(bf16(a + dt_bias))), the softplus rounded to
// bf16 as the reference does.
struct GdnGates {
  bfloat beta;
  float decay;
};

inline GdnGates gdn_gates(device const bfloat *packed_row,
                          device const bfloat *dt_bias,
                          device const float *a_scale, uint b_offset,
                          uint a_offset, uint head) {
  float b = float(packed_row[b_offset + head]);
  GdnGates gates;
  gates.beta = bfloat(richengine_sigmoid(b));
  bfloat x = bfloat(float(packed_row[a_offset + head]) + float(dt_bias[head]));
  float xf = float(x);
  bfloat softplus =
      bfloat(max(xf, 0.0f) +
             fast::log2(1.0f + fast::exp2(-1.44269504089f * abs(xf))) *
                 0.69314718056f);
  gates.decay = fast::exp(a_scale[head] * float(softplus));
  return gates;
}

// The position of value head `head` among the GDN output's head blocks: the
// head itself, or with `tiled` llama.cpp's GGUF order, which puts value head
// j of every key head next to each other (GDNGatePrefillParams).
template <uint KeyHeads, uint ValueHeads>
inline uint gdn_output_head(uint head, bool tiled) {
  constexpr uint HeadsPerKey = ValueHeads / KeyHeads;
  return tiled ? (head % HeadsPerKey) * KeyHeads + head / HeadsPerKey : head;
}

// Gated RMSNorm of the recurrent row of one task (token, value head), one
// thread per dimension, stored at the head's output position. Prefill
// dispatches one task per threadgroup; decode runs gdn_decode_gate, which
// reproduces these rows bitwise. The norm weights are read in their stored
// type W: bfloat in the packed formats, float for a GGUF's F32 norms.
template <uint KeyHeads, uint ValueHeads, uint HeadDim, uint ConvDim,
          uint PackedWidth, class W>
inline void
gdn_gate_phase(device const bfloat *recurrent, device const bfloat *packed,
               device const W *norm_weight, device bfloat *hidden, uint task,
               bool tiled, threadgroup float *scratch, uint thread_index,
               uint lane, uint simd_group) {
  constexpr uint Simdgroups = 4, ZOffset = ConvDim;
  static_assert(HeadDim == Simdgroups * 32, "one thread per dimension");
  uint token = task / ValueHeads;
  uint head = task % ValueHeads;
  ulong base = ulong(task) * HeadDim;
  ulong hidden_base =
      (ulong(token) * ValueHeads +
       gdn_output_head<KeyHeads, ValueHeads>(head, tiled)) *
      HeadDim;
  float value = float(recurrent[base + thread_index]);
  const float inverse = rms_inverse_of_sums<Simdgroups>(
      value * value, HeadDim, scratch, thread_index, lane, simd_group);
  bfloat normalized =
      bfloat(value * inverse * float(norm_weight[thread_index]));
  float gate = float(packed[token * PackedWidth + ZOffset + head * HeadDim +
                            thread_index]);
  float silu = richengine_silu(gate);
  hidden[hidden_base + thread_index] = bfloat(float(normalized) * silu);
}
