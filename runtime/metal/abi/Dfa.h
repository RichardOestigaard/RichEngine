#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#include "metal/abi/ExecutionGeometry.h"
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// The state a byte-DFA walk reports for a dead row: the table's -1 becomes
// this sentinel in the row-state and persistent-lane-state buffers.
#define RICHENGINE_DFA_DEAD_STATE 0xffffffffu

// A next-state row's slot-0 sentinel marking the whole row disabled: the
// verify mask pass writes it instead of a real row when the verify row the
// mask row serves can never be reached — the lane's accept walk stops at the
// first dead proposal, so deeper rows are never consulted (a dead proposal
// admits nothing and its rejection is unconditional). The selection kernels
// early-out on the sentinel; -1 already means "token not admitted" and real
// states are nonnegative, so -2 is free.
#define RICHENGINE_DFA_ROW_DISABLED (-2)

// The forced-span byte cap: the commit-stage forced kernels follow a lane's
// single-live-byte chain at most this far into the next round.
#define RICHENGINE_DFA_FORCED_BYTES 64u

// The grammar-EOS ids a CONSTRAINT_DFA frame may carry per request, bound
// per lane in the walk parameters.
#define RICHENGINE_DFA_EOS_TOKENS 8u

// Per state lane, the next round's forced-token span the commit-stage DFA
// kernels derive on-device (decode_dfa_forced_bytes/_tokens): bytes holds
// the single-live-byte chain followed from the post-commit state, token_ids
// its greedy longest-prefix tokenization, tokens the emitted proposal count
// and consumed the bytes those tokens cover. Indexed by state lane — a span
// outlives the dispatch lane that produced it.
struct DfaForcedSpan {
  // Proposal tokens emitted for the next round; 0 means no span.
  uint32_t tokens;
  // Forced bytes followed from base_state.
  uint32_t byte_count;
  // Bytes of the span the emitted tokens cover.
  uint32_t consumed;
  // The automaton state the span starts from: the post-commit state walked
  // through the new pending anchor.
  uint32_t base_state;
  uint32_t bytes[RICHENGINE_DFA_FORCED_BYTES / 4];
  uint32_t token_ids[RICHENGINE_DRAFT_PROPOSAL_TOKENS];
};

static_assert(sizeof(DfaForcedSpan) ==
                  16 + RICHENGINE_DFA_FORCED_BYTES +
                      4 * RICHENGINE_DRAFT_PROPOSAL_TOKENS,
              "DFA forced span differs across the ABI");

// Per-dispatch scalars of decode_dfa_mask (kernels/decode/dfa.metal). The
// grid is (tokens, rows, lanes); lane_mask gates the work to the lanes that
// carry a GPU DFA, and live_rows bounds each lane's rows from row 0 — the
// verify dispatch writes its live selected rows plus the anchor's row. A
// row records each token's next state (-1 = not admitted), and admits the
// stop tokens when its start state is accepting.
struct DfaMaskParams {
  uint32_t vocabulary;
  // Rows computed this dispatch: 1 for an initial-anchor mask, the widest
  // lane's live_rows + 1 for a verify round.
  uint32_t rows;
  uint32_t lane_mask;
  // Each lane's grammar-EOS set (CONSTRAINT_DFA's eos ids): a member token
  // admits iff the row's start state is accepting, and the walk ends at the
  // stop state — not the engine's own stop tokens, which only the lane's
  // grammar knows.
  uint32_t eos_count[RICHENGINE_MAXIMUM_BATCH_WIDTH];
  uint32_t eos_ids[RICHENGINE_MAXIMUM_BATCH_WIDTH][RICHENGINE_DFA_EOS_TOKENS];
  // Each lane's DFA state count; bounds the state it reads back.
  uint32_t states[RICHENGINE_MAXIMUM_BATCH_WIDTH];
  // Rows each lane writes, from row 0: the live verify row count + 1 on a
  // verify round, 1 for the initial selection's single row.
  uint32_t live_rows[RICHENGINE_MAXIMUM_BATCH_WIDTH];
  // Indexed mode: each lane writes only the row mask_rows[lane] + 1 — the
  // correction row of the truncation pass — instead of rows 0..live_rows.
  uint32_t indexed;
};

static_assert(sizeof(DfaMaskParams) == 192,
              "DFA mask parameters are 192 bytes on both sides");

// Per-dispatch scalars of decode_dfa_chain, decode_dfa_advance and the
// commit-stage forced-span kernels. The verify input and output tokens pack
// verify_stride rows per lane; the persistent automaton state and the forced
// span index their tables by state lane, not dispatch lane. live_rows bounds
// the chain kernel's walk (0 = all RICHENGINE_TARGET_VERIFY_ROWS); the
// forced-token scan needs vocabulary.
struct DfaWalkParams {
  uint32_t verify_stride;
  uint32_t lane_mask;
  uint32_t states[RICHENGINE_MAXIMUM_BATCH_WIDTH];
  uint32_t state_lanes[RICHENGINE_MAXIMUM_BATCH_WIDTH];
  uint32_t vocabulary;
  uint32_t live_rows[RICHENGINE_MAXIMUM_BATCH_WIDTH];
  // Each lane's grammar-EOS set: a member token is admitted only from an
  // accepting state, by the mask row's own rule (decode_dfa_chain), and
  // ends the walk at the stop state — the lane's DFA state count.
  uint32_t eos_count[RICHENGINE_MAXIMUM_BATCH_WIDTH];
  uint32_t eos_ids[RICHENGINE_MAXIMUM_BATCH_WIDTH][RICHENGINE_DFA_EOS_TOKENS];
};

static_assert(sizeof(DfaWalkParams) == 204,
              "DFA walk parameters are 204 bytes on both sides");

// Per-dispatch scalars of decode_dfa_truncate: one thread per lane scans
// the verify rows and names the truncation row T — the first proposal the
// byte-walk rejects or whose token is not the row's unmasked argmax, capped
// by the acceptance kernel's own proposal limit. proposals bounds each
// lane's compare walk; forced is its GPU forced-span prefix, whose rows
// auto-accept without the argmax check.
struct DfaTruncParams {
  uint32_t verify_stride;
  uint32_t lane_mask;
  uint32_t proposals[RICHENGINE_MAXIMUM_BATCH_WIDTH];
  uint32_t forced[RICHENGINE_MAXIMUM_BATCH_WIDTH];
};

static_assert(sizeof(DfaTruncParams) == 40,
              "DFA truncation parameters are 40 bytes on both sides");

// Per-dispatch scalars of decode_dfa_mask_grid, the RICHENGINE_GPU_DECODE_GRID
// companion of decode_dfa_mask: the single-thread kernel packs the mask
// pass's MTLDispatchThreadgroupsIndirectArguments — {vocabulary_groups,
// rows, lanes} — while consuming the GPU forced spans of the lanes
// gpu_span_lanes names (a span indexes its state lane). live_rows is the
// host's per-lane live verify-row bound, the span-free fallback a lane
// takes when its span was already consumed or cleared; maximum_retained
// caps a span's row count exactly as the host's forced-span derivation
// does.
struct DfaMaskGridParams {
  uint32_t vocabulary_groups;
  uint32_t lanes;
  uint32_t gpu_span_lanes;
  uint32_t state_lanes[RICHENGINE_MAXIMUM_BATCH_WIDTH];
  uint32_t maximum_retained[RICHENGINE_MAXIMUM_BATCH_WIDTH];
  uint32_t live_rows[RICHENGINE_MAXIMUM_BATCH_WIDTH];
};

static_assert(sizeof(DfaMaskGridParams) == 60,
              "DFA mask grid parameters are 60 bytes on both sides");

// Per-dispatch scalars of decode_dfa_span_apply, the
// RICHENGINE_DFA_MERGE bridge between the draft pass and the verify input
// of the same command: each named lane copies its state lane's pending
// DfaForcedSpan proposals over the draft's own — capped exactly as the
// split path's host apply — and clears the span so a later round cannot
// reapply it.
struct DfaSpanApplyParams {
  uint32_t lane_mask;
  uint32_t state_lanes[RICHENGINE_MAXIMUM_BATCH_WIDTH];
  uint32_t maximum_retained[RICHENGINE_MAXIMUM_BATCH_WIDTH];
};

static_assert(sizeof(DfaSpanApplyParams) == 36,
              "DFA span-apply parameters are 36 bytes on both sides");
