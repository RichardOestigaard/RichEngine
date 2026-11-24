#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#include "metal/abi/ExecutionGeometry.h"

#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// The Mamba-2 (SSD) mixer parameters, mirroring the GDN contract: the packed
// width, head/group counts and the packed, mixed and gate row strides are
// constants of a kernel's compiled variant (Nemotron 3.5's 64 heads x 64 dims,
// 8 groups x 128 states, 6144 conv channels, 10304 packed); the state cells'
// byte strides come from the host. The packed row carves
// [gate Inner | conv (x | B | C) | dt Heads]; the conv region carves
// [x Inner | B Groups*State | C Groups*State].
//
// Per-lane per-layer state cell layout (same placement contract as the GDN
// cells, so CompositeStateStorage's cell accounting transfers):
//   [0, conv_layer_bytes * layers):        convolution FIFOs — per layer the
//                                        last convTaps-1 raw conv inputs,
//                                        (taps-1) * ConvDim bf16 aligned up.
//   [convolution_state_bytes + layer * recurrent_layer_bytes,
//    + recurrent_layer_bytes):             recurrent matrix S, fp32
//                                        [Heads][HeadDim][StateSize]
//                                        contiguous: head h's matrix at
//                                        h * HeadDim * StateSize, row d at
//                                        d * StateSize, column n.

// The prefill prepare and scan kernels. dt bounds apply after softplus:
// dt' = clamp(softplus(dt + dt_bias), dt_min, dt_max) (Nemotron's
// time_step_limit); the scan recomputes dA = exp(dt' * A[h]) per token.
struct Mamba2PrefillParams {
  uint32_t tokens;
  float dt_min;
  float dt_max;
};

static_assert(sizeof(Mamba2PrefillParams) == 12,
              "Mamba2 prefill parameters are 12 bytes on both sides");

// The grouped RMSNorm's epsilon rides the gate params: Nemotron's norm
// applies silu(z) to the scan rows first and normalizes the gated values per
// 512-column group (NemotronHRMSNormGated), so the epsilon is a runtime
// input rather than an _e5 name tag.
struct Mamba2GateParams {
  uint32_t flags;
  float rms_eps;
};

static_assert(sizeof(Mamba2GateParams) == 8,
              "Mamba2 gate parameters are 8 bytes on both sides");

// MAMBA2_TAPE_FOLD marks a decode whose tape binding is live: only then may
// the kernel read the pending prefix and write this step's rows for the next
// fold. The argument table is shared and never cleared, so an unbound tape
// slot keeps a stale address — the kernel must not dereference it.
#define MAMBA2_TAPE_FOLD 1u

struct Mamba2DecodeBatchParams {
  uint32_t flags;
  uint32_t layer;
  float dt_min;
  float dt_max;
  // The gated group RMSNorm's epsilon (Nemotron: 1e-5).
  float rms_eps;
  uint64_t conv_layer_bytes;
  uint64_t recurrent_layer_bytes;
  uint64_t convolution_state_bytes;
  // Adaptive proposal budgets (RICHENGINE_ADAPTIVE_PROPOSALS): each lane's live
  // verify rows bound the serial scan. Zero or RICHENGINE_TARGET_VERIFY_ROWS
  // scans all sixteen rows; the commit replays only retained rows regardless.
  uint32_t live_rows[RICHENGINE_MAXIMUM_BATCH_WIDTH];
  // The packed lane stride of the verify tensors: RICHENGINE_TARGET_VERIFY_ROWS
  // unless the draft's trained block is shorter (draft.verifyRows()).
  uint32_t verify_stride;
  // The committed-state fold: per lane, the byte offset of the previous
  // step's retained rows in the tape buffer (their raw conv inputs, post-conv
  // x|B|C rows and dt' values, as the decode tensors pack them) and how many
  // the lane kept. The scan advances the state over them without emitting; a
  // lane with zero retains none. MAMBA2_TAPE_FOLD keeps the standalone commit
  // when off. The tape's per-layer region lays out
  //   [Rows][ConvDim] bf16 raw conv inputs,
  //   [Rows][ConvDim] bf16 post-conv x|B|C,
  //   [Rows][Heads]  fp32 dt'.
  uint64_t pending_tape[RICHENGINE_MAXIMUM_BATCH_WIDTH];
  uint32_t pending_rows[RICHENGINE_MAXIMUM_BATCH_WIDTH];
  // The lane's current-step tape region: the fused kernel's prologue leaves
  // the step's rows (raw conv inputs, post-conv rows, dt') there for the next
  // step's pending prefix. Parity alternates with the state slots.
  uint64_t step_tape[RICHENGINE_MAXIMUM_BATCH_WIDTH];
};

static_assert(sizeof(Mamba2DecodeBatchParams) == 152,
              "Mamba2 decode parameters are 152 bytes on both sides");

struct Mamba2BatchCommitParams {
  uint64_t conv_layer_bytes;
  uint64_t recurrent_layer_bytes;
  uint64_t convolution_state_bytes;
  // The packed lane stride of the commit's packed tensors: a chain batch's
  // verify stride.
  uint32_t verify_stride;
};

static_assert(sizeof(Mamba2BatchCommitParams) == 32,
              "Mamba2 commit parameters are 32 bytes on both sides");
