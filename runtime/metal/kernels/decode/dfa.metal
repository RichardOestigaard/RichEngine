#include "metal/abi/KernelABI.h"

// GPU-DFA constrained decode: the host ships a compiled byte-DFA
// (CONSTRAINT_DFA) and the packed token vocabulary (VOCABULARY) once per
// request; each verify round then computes the lane's constraint admission
// on the GPU instead of waiting for a host mask response. All kernels share
// one byte-walk over the row-major Q x 256 i16 table (-1 = dead; the host
// ships only live edges, so a non-dead end state always reaches acceptance).
// The walk honors the grammar's marker rule, matching server/dfa.py's
// _walk: a per-state 256-bit mask names the edges only a special token may
// traverse — a regular token dies on a marked edge, and a special token
// dies when it takes none. The vocabulary's special-token bitset rides in
// the VOCABULARY frame.
//
// decode_dfa_mask records the walk's result per (token, row, lane) into the
// lane's DfaNextStates rows — the state the token leaves, or -1 when it is
// not admitted — which the selection kernels read in place of a bitmask
// (TargetRow::admits in decode/sampling.metal). After acceptance,
// decode_dfa_advance advances the persistent state and the forced-span pair
// tokenizes the next round's forced bytes on-device: decode_dfa_forced_bytes
// follows the single-live-byte chain behind the new anchor, and
// decode_dfa_forced_tokens greedily covers it with vocabulary tokens.
//
// A grammar-EOS token (the frame's per-lane eos set — the host contract's
// eos ids, not the engine's own stop tokens) is admitted only from a state
// the shipped accepting bitset marks — the grammar's accepting union
// free-text states — and ends the walk at the stop state, the lane's DFA
// state count, from which nothing walks further.

constant constexpr uint kDfaRows = RICHENGINE_TARGET_VERIFY_ROWS + 1u;

// The four dispatch lanes' transition tables, bound one per buffer index;
// lanes without a DFA rebind lane 0's and are masked out.
inline device const short *dfa_table(
    device const short *table0, device const short *table1,
    device const short *table2, device const short *table3, uint lane) {
  return lane == 0 ? table0 : lane == 1 ? table1 : lane == 2 ? table2 : table3;
}

// The per-lane byte-masks, selected like the tables: the accepting bitset
// and the marker mask (nStates * 32 bytes, bit b of state s marking edge
// (s, b) special-only).
inline device const uchar *dfa_bits(
    device const uchar *bits0, device const uchar *bits1,
    device const uchar *bits2, device const uchar *bits3, uint lane) {
  return lane == 0 ? bits0 : lane == 1 ? bits1 : lane == 2 ? bits2 : bits3;
}

// The bit naming state s an EOS-admitting state.
inline bool dfa_accepting(device const uchar *accepting, uint state) {
  return (accepting[state >> 3] >> (state & 7u)) & 1u;
}

// The bit naming state s's edge on byte b special-only.
inline bool dfa_marked(device const uchar *marker, uint state, uint byte) {
  return (marker[state * 32 + (byte >> 3)] >> (byte & 7u)) & 1u;
}

// The vocabulary-frame bit naming token a special token.
inline bool dfa_special(device const uchar *special, uint token) {
  return (special[token >> 3] >> (token & 7u)) & 1u;
}

// The bit naming token a grammar-EOS id of this lane.
inline bool dfa_eos(constant const uint *eos_ids, uint count, uint token) {
  for (uint i = 0; i < count; ++i)
    if (eos_ids[i] == token)
      return true;
  return false;
}

// The marker-honoring byte walk: the state after token's bytes walked from
// state, or the dead sentinel. A regular (non-special) token dies on a
// marked edge; a special token dies when its walk takes no marked edge.
inline uint dfa_walk(device const short *table,
                     device const uchar *marker,
                     device const uchar *token_bytes,
                     device const int *token_offsets,
                     bool special, uint state, uint token) {
  if (state == RICHENGINE_DFA_DEAD_STATE)
    return state;
  const int end = token_offsets[token + 1];
  bool marked = false;
  for (int byte = token_offsets[token]; byte < end; ++byte) {
    const uint b = token_bytes[byte];
    const int next = table[state * 256 + b];
    if (next < 0)
      return RICHENGINE_DFA_DEAD_STATE;
    if (dfa_marked(marker, state, b)) {
      if (!special)
        return RICHENGINE_DFA_DEAD_STATE;
      marked = true;
    }
    state = uint(next);
  }
  if (special && !marked)
    return RICHENGINE_DFA_DEAD_STATE;
  return state;
}

// The state one committed or simulated token leaves, matching the host's
// advance_token: a grammar-EOS id is admitted only from an EOS-admitting
// state and lands on the stop state (states); anything else byte-walks
// under the marker rule, and a token with empty bytes dies.
inline uint dfa_token_state(
    device const short *table, device const uchar *marker,
    device const uchar *accepting, device const uchar *token_bytes,
    device const int *token_offsets, device const uchar *special,
    constant const uint *eos_ids, uint eos_count, uint states, uint state,
    uint token) {
  const bool live =
      state != RICHENGINE_DFA_DEAD_STATE && state < states;
  if (dfa_eos(eos_ids, eos_count, token))
    return live && dfa_accepting(accepting, state) ? states
                                                  : RICHENGINE_DFA_DEAD_STATE;
  if (!live || token_offsets[token] == token_offsets[token + 1])
    return RICHENGINE_DFA_DEAD_STATE;
  return dfa_walk(table, marker, token_bytes, token_offsets,
                  dfa_special(special, token), state, token);
}

// One thread per (token, row, lane): writes the state the token's byte walk
// leaves from the row's automaton state into the lane's DfaNextStates row —
// -1 when the token is not admitted — a plain store with no clear pass, so a
// row is fully rewritten each round. A row's record admits the tokens legal
// after verify input rows 0..r-1, so verify row r reads row r + 1 as in the
// host path. A grammar-EOS token admits iff the row's start state is
// EOS-admitting; it records the stop state since its emission ends the
// walk. A dead or out-of-range start writes -1 across the row. Only the
// lane's live_rows rows are written: verify selection never reads the rows
// past a lane's live verify rows, and row 0 — the anchor's row — rides
// along for free.
//
// The non-indexed pass bounds each lane's writes at the first dead proposal:
// verify row r reads mask row r + 1, and the accept walk stops at the first
// draft token the chain's byte-walk kills (row_states[p + 2] names the state
// after draft position p — dead means the proposal admits nothing and its
// rejection is unconditional), so the mask rows past that bound are never
// consulted. Each skipped row's token-0 thread writes the disabled sentinel,
// which the selection kernels read as "row unreachable" and return at once
// instead of walking a stale row; its other slots keep stale values nobody
// reads. The rows the bound keeps still write the full walk, so the
// consulted rows are identical to the unbounded pass's.
kernel void decode_dfa_mask(
    device const short *table0 [[buffer(0)]],
    device const short *table1 [[buffer(1)]],
    device const short *table2 [[buffer(2)]],
    device const short *table3 [[buffer(3)]],
    device const uchar *accepting0 [[buffer(4)]],
    device const uchar *accepting1 [[buffer(5)]],
    device const uchar *accepting2 [[buffer(6)]],
    device const uchar *accepting3 [[buffer(7)]],
    device const uchar *token_bytes [[buffer(8)]],
    device const int *token_offsets [[buffer(9)]],
    device const uint *row_states [[buffer(10)]],
    device short *next_states [[buffer(11)]],
    // Indexed mode's per-lane row indices (the truncation pass's rows):
    // each lane writes row mask_rows[lane] + 1 alone.
    device const uint *mask_rows [[buffer(12)]],
    device const uchar *marker0 [[buffer(13)]],
    device const uchar *marker1 [[buffer(14)]],
    device const uchar *marker2 [[buffer(15)]],
    device const uchar *marker3 [[buffer(16)]],
    device const uchar *special [[buffer(17)]],
    constant DfaMaskParams &params [[buffer(18)]],
    uint3 gid [[thread_position_in_grid]]) {
  const uint lane = gid.z;
  const uint token = gid.x;
  if (!(params.lane_mask & (1u << lane)) || gid.y >= params.rows ||
      token >= params.vocabulary)
    return;
  uint row = gid.y;
  if (params.indexed) {
    row = mask_rows[lane] + 1;
    if (row >= kDfaRows)
      return;
  } else if (row >= params.live_rows[lane]) {
    return;
  }
  if (!params.indexed) {
    // The first dead row state bounds the consultable rows: a dead row
    // state at index i means draft position i - 1's token died — the verify
    // row at index i - 2 is the last the accept walk can consult, so mask
    // rows at or past i are never read. (The anchor's own dead row keeps
    // every position dead, bounding the write at row 1.)
    uint bound = params.live_rows[lane];
    for (uint i = 1; i < params.live_rows[lane]; ++i) {
      if (row_states[lane * kDfaRows + i] == RICHENGINE_DFA_DEAD_STATE) {
        bound = i == 1 ? 2 : i;
        break;
      }
    }
    if (row >= bound) {
      if (!token)
        next_states[(ulong(lane) * kDfaRows + row) * params.vocabulary] =
            RICHENGINE_DFA_ROW_DISABLED;
      return;
    }
  }
  const uint start = row_states[lane * kDfaRows + row];
  const uint states = params.states[lane];
  short next = -1;
  if (start != RICHENGINE_DFA_DEAD_STATE && start < states) {
    if (dfa_eos(params.eos_ids[lane], params.eos_count[lane], token)) {
      // A grammar-EOS id is admitted only from an EOS-admitting state.
      if (dfa_accepting(
              dfa_bits(accepting0, accepting1, accepting2, accepting3, lane),
              start))
        next = short(states);
    } else if (token_offsets[token] != token_offsets[token + 1]) {
      // A token whose byte span is empty is dead by definition.
      const uint end =
          dfa_walk(dfa_table(table0, table1, table2, table3, lane),
                   dfa_bits(marker0, marker1, marker2, marker3, lane),
                   token_bytes, token_offsets,
                   dfa_special(special, token), start, token);
      next = end == RICHENGINE_DFA_DEAD_STATE ? -1 : short(end);
    }
  }
  next_states[(ulong(lane) * kDfaRows + row) * params.vocabulary + token] =
      next;
}

// One thread per lane: the automaton state behind each of the lane's
// kDfaRows mask rows — the persistent state before the anchor, then the
// states the verify input rows leave. Input tokens are already clamped to
// the vocabulary by verify_input_tokens. The walk stops at the lane's live
// verify rows: deeper row states feed no mask row this round.
// Each row records the state the token's admission leaves by the host's
// advance_token rule, so a proposal sitting at input row r is admissible
// exactly when rows[r + 1] is not the dead sentinel — decode_dfa_truncate
// reads that as its per-row validity: a grammar-EOS id holds the walk only
// at an EOS-admitting state and lands on the stop state, an empty byte
// span dies, and any other token walks its bytes under the marker rule; a
// state outside the table admits nothing at all.
kernel void decode_dfa_chain(
    device const short *table0 [[buffer(0)]],
    device const short *table1 [[buffer(1)]],
    device const short *table2 [[buffer(2)]],
    device const short *table3 [[buffer(3)]],
    device const uchar *token_bytes [[buffer(4)]],
    device const int *token_offsets [[buffer(5)]],
    device const uint *verify_input [[buffer(6)]],
    device const uint *dfa_states [[buffer(7)]],
    device uint *row_states [[buffer(8)]],
    device const uchar *accepting0 [[buffer(9)]],
    device const uchar *accepting1 [[buffer(10)]],
    device const uchar *accepting2 [[buffer(11)]],
    device const uchar *accepting3 [[buffer(12)]],
    device const uchar *marker0 [[buffer(13)]],
    device const uchar *marker1 [[buffer(14)]],
    device const uchar *marker2 [[buffer(15)]],
    device const uchar *marker3 [[buffer(16)]],
    device const uchar *special [[buffer(17)]],
    constant DfaWalkParams &params [[buffer(18)]],
    uint lane [[thread_position_in_grid]]) {
  if (!(params.lane_mask & (1u << lane)))
    return;
  device const short *table =
      dfa_table(table0, table1, table2, table3, lane);
  device const uchar *accepting =
      dfa_bits(accepting0, accepting1, accepting2, accepting3, lane);
  device const uchar *marker =
      dfa_bits(marker0, marker1, marker2, marker3, lane);
  device uint *rows = row_states + lane * kDfaRows;
  const uint bound = params.live_rows[lane]
      ? min(params.live_rows[lane], uint(RICHENGINE_TARGET_VERIFY_ROWS))
      : uint(RICHENGINE_TARGET_VERIFY_ROWS);
  uint state = dfa_states[params.state_lanes[lane]];
  rows[0] = state;
  for (uint row = 0; row < bound; ++row) {
    const uint token = verify_input[lane * params.verify_stride + row];
    state = dfa_token_state(table, marker, accepting, token_bytes,
                            token_offsets, special, params.eos_ids[lane],
                            params.eos_count[lane], params.states[lane],
                            state, token);
    rows[row + 1] = state;
  }
}

// One thread per lane, after the unmasked argmax pass: the lane's
// truncation row T — the first proposal row whose byte-walk admission
// died (row_states[row + 2] is the dead sentinel) or whose token is not
// the row's unmasked argmax — capped by the acceptance kernel's proposal
// limit; T = limit means every proposal passed. The GPU forced-span
// prefix's rows skip the argmax check: their tokens are the byte-forced
// continuation, so the emitted token is the proposal whatever the argmax
// says, and the kernel writes it over the row's unmasked argmax so the
// acceptance prefix passes there too.
kernel void decode_dfa_truncate(
    device const uint *row_states [[buffer(0)]],
    device const uint *draft_tokens [[buffer(1)]],
    device uint *target_tokens [[buffer(2)]],
    device uint *trunc_rows [[buffer(3)]],
    constant DfaTruncParams &params [[buffer(4)]],
    uint lane [[thread_position_in_grid]]) {
  if (!(params.lane_mask & (1u << lane)))
    return;
  const uint limit =
      min(params.proposals[lane],
          min(uint(RICHENGINE_DRAFT_PROPOSAL_TOKENS),
              params.verify_stride - 1));
  const uint forced = min(params.forced[lane], limit);
  device const uint *rows = row_states + lane * kDfaRows;
  device const uint *draft =
      draft_tokens + lane * RICHENGINE_DRAFT_PROPOSAL_TOKENS;
  device uint *target = target_tokens + lane * params.verify_stride;
  uint t = limit;
  for (uint row = 0; row < limit; ++row) {
    if (rows[row + 2] == RICHENGINE_DFA_DEAD_STATE) {
      t = row;
      break;
    }
    if (row < forced) {
      target[row] = draft[row];
      continue;
    }
    if (target[row] != draft[row]) {
      t = row;
      break;
    }
  }
  trunc_rows[lane] = t;
}

// One thread per lane, after acceptance: the persistent automaton state
// advances by the tokens the step committed — the anchor plus the retained
// outputs except the last, which is the new pending anchor the next
// round's row-0 mask must admit. A committed grammar-EOS id lands on the
// stop state rather than byte-walking its spelling.
kernel void decode_dfa_advance(
    device const short *table0 [[buffer(0)]],
    device const short *table1 [[buffer(1)]],
    device const short *table2 [[buffer(2)]],
    device const short *table3 [[buffer(3)]],
    device const uchar *token_bytes [[buffer(4)]],
    device const int *token_offsets [[buffer(5)]],
    device const uint *verify_input [[buffer(6)]],
    device const uint *output_tokens [[buffer(7)]],
    device const uint *retained [[buffer(8)]],
    device uint *dfa_states [[buffer(9)]],
    device const uchar *accepting0 [[buffer(10)]],
    device const uchar *accepting1 [[buffer(11)]],
    device const uchar *accepting2 [[buffer(12)]],
    device const uchar *accepting3 [[buffer(13)]],
    device const uchar *marker0 [[buffer(14)]],
    device const uchar *marker1 [[buffer(15)]],
    device const uchar *marker2 [[buffer(16)]],
    device const uchar *marker3 [[buffer(17)]],
    device const uchar *special [[buffer(18)]],
    constant DfaWalkParams &params [[buffer(19)]],
    uint lane [[thread_position_in_grid]]) {
  if (!(params.lane_mask & (1u << lane)))
    return;
  device const short *table =
      dfa_table(table0, table1, table2, table3, lane);
  device const uchar *accepting =
      dfa_bits(accepting0, accepting1, accepting2, accepting3, lane);
  device const uchar *marker =
      dfa_bits(marker0, marker1, marker2, marker3, lane);
  const uint state_lane = params.state_lanes[lane];
  uint state = dfa_states[state_lane];
  const uint count = min(retained[lane], params.verify_stride);
  state = dfa_token_state(table, marker, accepting, token_bytes,
                          token_offsets, special, params.eos_ids[lane],
                          params.eos_count[lane], params.states[lane],
                          state, verify_input[lane * params.verify_stride]);
  for (uint row = 0; row + 1 < count; ++row) {
    state = dfa_token_state(
        table, marker, accepting, token_bytes, token_offsets, special,
        params.eos_ids[lane], params.eos_count[lane], params.states[lane],
        state, output_tokens[lane * params.verify_stride + row]);
  }
  dfa_states[state_lane] = state;
}

// One threadgroup per DFA lane, after decode_dfa_advance: the forced byte
// span is the chain of states with exactly one live out-byte, followed from
// the state the new pending anchor leaves — dfa_states holds the pre-anchor
// state, so the kernel first walks the anchor, the last retained output.
// A grammar-EOS anchor lands on the stop state, leaving no span. The bytes
// land in the state lane's DfaForcedSpan for
// decode_dfa_forced_tokens; byte_count of 0 means no span.
kernel void decode_dfa_forced_bytes(
    device const short *table0 [[buffer(0)]],
    device const short *table1 [[buffer(1)]],
    device const short *table2 [[buffer(2)]],
    device const short *table3 [[buffer(3)]],
    device const uchar *token_bytes [[buffer(4)]],
    device const int *token_offsets [[buffer(5)]],
    device const uint *output_tokens [[buffer(6)]],
    device const uint *retained [[buffer(7)]],
    device const uint *dfa_states [[buffer(8)]],
    device DfaForcedSpan *forced [[buffer(9)]],
    device const uchar *accepting0 [[buffer(10)]],
    device const uchar *accepting1 [[buffer(11)]],
    device const uchar *accepting2 [[buffer(12)]],
    device const uchar *accepting3 [[buffer(13)]],
    device const uchar *marker0 [[buffer(14)]],
    device const uchar *marker1 [[buffer(15)]],
    device const uchar *marker2 [[buffer(16)]],
    device const uchar *marker3 [[buffer(17)]],
    device const uchar *special [[buffer(18)]],
    constant DfaWalkParams &params [[buffer(19)]],
    uint lane [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  if (!(params.lane_mask & (1u << lane)))
    return;
  device const short *table =
      dfa_table(table0, table1, table2, table3, lane);
  const uint state_lane = params.state_lanes[lane];
  device DfaForcedSpan &span = forced[state_lane];
  device uchar *bytes = reinterpret_cast<device uchar *>(span.bytes);
  threadgroup uint tg_state = 0;
  threadgroup uint tg_base = 0;
  threadgroup uint tg_byte = 0;
  threadgroup uint tg_count = 0;
  threadgroup uint tg_parts[8];
  if (thread_index == 0) {
    const uint count = max(min(retained[lane], params.verify_stride), 1u);
    const uint anchor =
        output_tokens[lane * params.verify_stride + count - 1];
    tg_state = dfa_token_state(
        table, dfa_bits(marker0, marker1, marker2, marker3, lane),
        dfa_bits(accepting0, accepting1, accepting2, accepting3, lane),
        token_bytes, token_offsets, special, params.eos_ids[lane],
        params.eos_count[lane], params.states[lane],
        dfa_states[state_lane], anchor);
    tg_base = tg_state;
    tg_count = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  while (tg_count < RICHENGINE_DFA_FORCED_BYTES) {
    const uint q = tg_state;
    if (q == RICHENGINE_DFA_DEAD_STATE || q >= params.states[lane])
      break;
    // The first 256 threads test one out-byte each; every thread sums the
    // eight simdgroup shares, so the chain's break is decided uniformly.
    const uint live =
        thread_index < 256 && table[q * 256 + thread_index] >= 0 ? 1u : 0u;
    const uint part = simd_sum(live);
    if (!simd_lane && simd_group < 8)
      tg_parts[simd_group] = part;
    // A racing write only happens when several bytes are live, and the
    // value is read back only when exactly one is.
    if (live)
      tg_byte = thread_index;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint count = 0;
    for (uint g = 0; g < 8; ++g)
      count += tg_parts[g];
    if (count != 1)
      break;
    if (thread_index == 0) {
      bytes[tg_count] = uchar(tg_byte);
      tg_state = uint(table[q * 256 + tg_byte]);
      ++tg_count;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (thread_index == 0) {
    span.byte_count = tg_count;
    span.base_state = tg_base;
  }
}

// One threadgroup per DFA lane: greedily tokenizes the lane's forced byte
// span — at each slot the longest vocabulary token whose bytes continue the
// span, lowest id on ties — checking each emitted token still walks live
// from the evolving state under the marker rule. The tokens go to the
// state lane's DfaForcedSpan, and to the dispatch lane's ProposedTokens for
// any same-round reader; the next constrained round's draft stage adopts
// the span from DfaForcedSpan.
kernel void decode_dfa_forced_tokens(
    device const short *table0 [[buffer(0)]],
    device const short *table1 [[buffer(1)]],
    device const short *table2 [[buffer(2)]],
    device const short *table3 [[buffer(3)]],
    device const uchar *token_bytes [[buffer(4)]],
    device const int *token_offsets [[buffer(5)]],
    device DfaForcedSpan *forced [[buffer(6)]],
    device uint *proposed [[buffer(7)]],
    device const uchar *marker0 [[buffer(8)]],
    device const uchar *marker1 [[buffer(9)]],
    device const uchar *marker2 [[buffer(10)]],
    device const uchar *marker3 [[buffer(11)]],
    device const uchar *special [[buffer(12)]],
    constant DfaWalkParams &params [[buffer(13)]],
    uint lane [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  if (!(params.lane_mask & (1u << lane)))
    return;
  device const short *table =
      dfa_table(table0, table1, table2, table3, lane);
  device const uchar *marker =
      dfa_bits(marker0, marker1, marker2, marker3, lane);
  const uint state_lane = params.state_lanes[lane];
  device DfaForcedSpan &span = forced[state_lane];
  device const uchar *bytes =
      reinterpret_cast<device const uchar *>(span.bytes);
  const uint total = min(span.byte_count, uint(RICHENGINE_DFA_FORCED_BYTES));
  threadgroup uint tg_state = 0;
  threadgroup uint tg_offset = 0;
  threadgroup uint tg_count = 0;
  threadgroup uint tg_len = 0;
  threadgroup uint sg_len[32];
  threadgroup uint sg_id[32];
  if (thread_index == 0) {
    tg_state = span.base_state;
    tg_offset = 0;
    tg_count = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint slot = 0; slot < RICHENGINE_DRAFT_PROPOSAL_TOKENS; ++slot) {
    const uint offset = tg_offset;
    const uint q = tg_state;
    if (offset >= total || q == RICHENGINE_DFA_DEAD_STATE ||
        q >= params.states[lane])
      break;
    // The slot's token: the longest whose bytes prefix-match the remaining
    // span, ties to the lowest id, so every run tokenizes identically.
    const uint remaining = total - offset;
    uint best_len = 0;
    uint best_id = 0xffffffffu;
    for (uint token = thread_index; token < params.vocabulary;
         token += 1024) {
      const int begin = token_offsets[token];
      const uint len = uint(token_offsets[token + 1] - begin);
      if (!len || len > remaining || len < best_len)
        continue;
      bool prefix = true;
      for (uint i = 0; i < len; ++i)
        prefix = prefix && bytes[offset + i] == token_bytes[begin + i];
      if (prefix && (len > best_len || (len == best_len && token < best_id))) {
        best_len = len;
        best_id = token;
      }
    }
    const uint group_len = simd_max(best_len);
    const uint group_id =
        simd_min(best_len == group_len ? best_id : 0xffffffffu);
    if (!simd_lane) {
      sg_len[simd_group] = group_len;
      sg_id[simd_group] = group_id;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index == 0) {
      uint len = 0;
      uint id = 0xffffffffu;
      for (uint g = 0; g < 32; ++g) {
        if (sg_len[g] > len || (sg_len[g] == len && sg_id[g] < id)) {
          len = sg_len[g];
          id = sg_id[g];
        }
      }
      // The span's bytes are the automaton's own single-live chain, so the
      // token's walk can only die on a corrupt table or the marker rule;
      // end the span there.
      if (len) {
        const uint next =
            dfa_walk(table, marker, token_bytes, token_offsets,
                     dfa_special(special, id), q, id);
        if (next == RICHENGINE_DFA_DEAD_STATE) {
          len = 0;
        } else {
          span.token_ids[tg_count] = id;
          proposed[lane * RICHENGINE_DRAFT_PROPOSAL_TOKENS + tg_count] = id;
          ++tg_count;
          tg_offset = offset + len;
          tg_state = next;
        }
      }
      tg_len = len;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!tg_len)
      break;
  }
  if (thread_index == 0) {
    span.tokens = tg_count;
    span.consumed = tg_offset;
  }
}
