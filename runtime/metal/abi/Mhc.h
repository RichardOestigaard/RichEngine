#pragma once

// Parameters of the manifold-constrained hyper-connection (mHC) kernels
// (kernels/shared/mhc.metal, ops/Mhc.cpp): the maxwell_moe residual, a
// state of `streams` parallel fp32 streams of `hidden` values per row.
// Each decoder layer runs two modules — attention's and the MoE block's.
// A module's mapping gates the streams into the block's input (h_pre),
// injects the block's output back into every stream (h_post) and mixes
// the streams themselves by a doubly-stochastic `streams` x `streams`
// matrix (h_res, Sinkhorn-normalized logits — the "manifold
// constraint"). A head mapping contracts the streams to the model's last
// hidden row.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// Threads of the per-row kernels: one threadgroup per row.
#define RICHENGINE_MHC_ROW_THREADS 256u
// The most streams a module or the head handles — the maxwell_moe
// streams count, which the mix width below covers.
#define RICHENGINE_MHC_MAXIMUM_STREAMS 4u
// h_pre (streams) | h_post (streams) | h_res (streams^2) per row.
#define RICHENGINE_MHC_MAXIMUM_MIX_WIDTH                               \
  (RICHENGINE_MHC_MAXIMUM_STREAMS * RICHENGINE_MHC_MAXIMUM_STREAMS +   \
   2 * RICHENGINE_MHC_MAXIMUM_STREAMS)

// Shared by every mHC kernel. A row's streams are `streams` * `hidden`
// fp32 values; a row's mapping logits and outputs are `mixWidth` fp32
// values ([h_pre | h_post | h_res], h_res row-major); the gates' and
// Sinkhorn's 1e-6 epsilons are compile-time constants of the kernels, as
// the reference module hard-codes them. mhc_mappings alone reads
// `sinkhornIterations`.
struct MhcParams {
  uint32_t rows;
  uint32_t hidden;
  uint32_t streams;
  uint32_t sinkhornIterations;
  uint32_t mixWidth;
};

static_assert(sizeof(MhcParams) == 20,
              "mHC parameters are 20 bytes on both sides");

// mhc_weighted_sum's layout of its weights: row r's weight s sits at
// weights[r * weightsStride + weightsOffset + s], so the one kernel
// serves both the stream aggregate (a h_pre slice of a mappings row:
// stride mixWidth, offset 0) and the head reduce (a standalone
// [rows][streams] weights buffer: stride streams, offset 0).
struct MhcSumParams {
  uint32_t rows;
  uint32_t hidden;
  uint32_t streams;
  uint32_t weightsStride;
  uint32_t weightsOffset;
};

static_assert(sizeof(MhcSumParams) == 20,
              "mHC weighted-sum parameters are 20 bytes on both sides");

// mhc_project's fp32 projection of a row's flattened streams: one
// threadgroup per (column, row), logits[row][column] =
// sum_k streams[row][k] * weights[column][k]. The mHC mapping projection
// (columns == mixWidth) and the HC head's (columns == streams) share it —
// the float projections' fp32 input cannot go through the bf16-input
// GGUF float kernels.
struct MhcProjectParams {
  uint32_t rows;
  uint32_t columns;
  uint32_t width;
  uint32_t pad;
};

static_assert(sizeof(MhcProjectParams) == 16,
              "mHC projection parameters are 16 bytes on both sides");
