#pragma once

// The scalar payloads of the nemotron MTP draft's kernels
// (kernels/decode/nmtp.metal): the nextn block of a Nemotron-3.5-Lightning
// target — one NoPE GQA + ungated relu-squared MoE decoder layer chained
// over the trunk's captured hidden. All are packed u32 records the encode
// fills by value; nothing here exceeds 48 bytes.

#ifdef __METAL__
#include <metal_stdlib>
using namespace metal;
#else
#include <cstdint>
#endif

// nmtp_ehtoken / nmtp_ehtoken_chain: eh[row] = [embedNormed[row] |
// hnorm(source row)] — the fused eh projection's bf16 input. `source`
// picks the hidden row: 0 the bf16 captured rows at sourceBase + row
// (commit and prefill rows alike), 1 the per-lane bf16 anchor row
// (decode pass 0); the chain variant reads the fp32 hidden the previous
// pass left. Nemotron's norms take the target's 1e-5 epsilon.
struct NmtpEhTokenParams {
  uint32_t hidden;
  uint32_t rows;
  uint32_t source;      // 0 captured rows, 1 anchor rows
  uint32_t sourceBase;  // mode 0's first captured row
  float rmsEps;
};

// nmtp_kv_store: raw keys (NoPE, no q/k norms) and values into the lanes'
// rings. Mode 0 is a decode pass (slot = positions[lane * posStride] - 1 +
// pass; the query row needs no store — mtp_attention reads the projection
// output directly); mode 1 commit rows (slot = startPosition[lane] +
// row % stride, live while row % stride < retained[lane]).
struct NmtpKvStoreParams {
  uint32_t rows;
  uint32_t kvHeads;
  uint32_t headDim;
  uint32_t ring;
  uint32_t mode;
  uint32_t stride;    // mode 1: the verify row stride
  uint32_t posStride; // mode 0: the draft positions lane stride
  uint32_t pass;      // mode 0: the decode pass
  uint32_t startPositions[4];
};

// nmtp_kv_store_prefill: one prefill span's keys and values into its
// ring; row r commits slot base + r, dropped when base + r >= end (the
// sequence's last row, whose rolled token is not the prompt's).
struct NmtpPrefillKvParams {
  uint32_t rows;
  uint32_t kvHeads;
  uint32_t headDim;
  uint32_t ring;
  uint32_t base;  // the first row's slot position
  uint32_t end;   // the sequence's position end
};

// nmtp_moe_route: the sigmoid router's per-expert scores plus the
// exp_probs_b selection bias, the top-6 normalized (norm_topk_prob) and
// scaled by expert_weights_scale (2.5) — ids and fp32 weights of the
// row's routes. The route record is 16 u32: ids [0,6), weights [8,14).
struct NmtpMoeRouteParams {
  uint32_t hidden;
  uint32_t experts;
  uint32_t topK;
  float weightScale;
};

// nmtp_moe_up / nmtp_moe_down / nmtp_moe_combine: the ungated experts.
// Slots 0..topK-1 are the routed experts (ids from the route record),
// slot topK the ungated shared expert at unit weight — its intermediate
// width differs. The up pass stores relu(x W)^2 into the intermediate at
// a slot-stride plane; the down pass's fp32 partials land per slot;
// combine folds residual + sum(w_s * partial_s) + partial_shared.
struct NmtpMoeParams {
  uint32_t hidden;
  uint32_t intermediate;       // the routed experts' width
  uint32_t sharedIntermediate; // the shared expert's width
  uint32_t interStride;        // the up intermediate's per-slot stride
  uint32_t experts;
  uint32_t topK;
};
