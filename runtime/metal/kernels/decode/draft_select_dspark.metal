#include "metal/abi/KernelABI.h"

inline bool top_beats(float value, uint token, float other, uint other_token) {
  return value > other || (value == other && token < other_token);
}

// Register-resident sorted top-16 insert for an entry the caller has already
// checked against the last slot; the unrolled shift keeps every index static.
inline void top16_insert(thread float (&values)[16], thread uint (&ids)[16],
                         float value, uint token) {
#pragma clang loop unroll(full)
  for (uint slot = 15; slot > 0; --slot) {
    bool here = top_beats(value, token, values[slot], ids[slot]);
    bool above = top_beats(value, token, values[slot - 1], ids[slot - 1]);
    values[slot] = here ? (above ? values[slot - 1] : value) : values[slot];
    ids[slot] = here ? (above ? ids[slot - 1] : token) : ids[slot];
  }
  bool top = top_beats(value, token, values[0], ids[0]);
  values[0] = top ? value : values[0];
  ids[0] = top ? token : ids[0];
}

// Pops the head of a register-resident sorted list of Count entries.
template <uint Count>
inline void top_pop(thread float (&values)[Count], thread uint (&ids)[Count]) {
#pragma clang loop unroll(full)
  for (uint slot = 0; slot + 1 < Count; ++slot) {
    values[slot] = values[slot + 1];
    ids[slot] = ids[slot + 1];
  }
  values[Count - 1] = -INFINITY;
  ids[Count - 1] = 0xffffffffu;
}

// The simdgroup's best list head: (value desc, id asc), so ties and the
// empty sentinel (-inf, ~0u) resolve the same way everywhere.
inline void simd_best_head(float value, uint token, thread float &best,
                           thread uint &best_token) {
  best = simd_max(value);
  best_token = simd_min(value == best ? token : 0xffffffffu);
}

// One shard of one proposal row. Threads stream their share of the shard in
// 16-byte vectors, keep the chunk in registers, and first find the 16th
// largest of the per-thread maxima: at least sixteen tokens are that large,
// so nothing below it can be in the row's top-16 and the exact sorted insert
// only runs for the few survivors. The group then pops its best sixteen in
// rank order. The (value desc, id asc) order is total, so the partial is the
// same set in the same order whatever the thread partition.
// The sharded top-16 of one (lane, position)'s logits row. row_offset is the
// logits row of proposal position zero: 1 for DFlash drafts (the anchor row
// reproduces the anchor), 0 for DSpark drafts, whose anchor row already
// predicts the next token.
inline void draft_top16_sharded_phase(
    device const float *logits, device uint *partial_ids,
    device float *partial_values, uint vocabulary, uint row_offset,
    threadgroup float *maxima, threadgroup float *thresholds,
    threadgroup float *round_values, threadgroup uint *round_ids, uint group,
    uint thread_index, uint lane, uint simd_group) {
  constexpr uint Rows = RICHENGINE_DRAFT_QUERY_ROWS;
  constexpr uint Positions = RICHENGINE_DRAFT_PROPOSAL_TOKENS;
  constexpr uint Shards = RICHENGINE_DRAFT_SAMPLING_SHARDS;
  constexpr uint K = RICHENGINE_DRAFT_CANDIDATES;
  constexpr uint VectorTokens = 4, ChunkVectors = 16;
  uint batch = group / (Positions * Shards);
  uint local = group % (Positions * Shards);
  uint position = local / Shards;
  uint shard = local % Shards;
  ulong row_start = (ulong(batch) * Rows + position + row_offset) * vocabulary;
  uint shard_tokens = (vocabulary + Shards - 1) / Shards;
  uint begin = min(shard * shard_tokens, vocabulary);
  uint end = min(begin + shard_tokens, vocabulary);
  // Vector loads require 16-byte alignment. Handle the shard's unaligned head
  // and tail with scalar inserts; the logits binding must also be aligned.
  uint head = uint((VectorTokens - (row_start + begin) % VectorTokens) %
                   VectorTokens);
  head = min(head, end - begin);
  uint vectors = (end - begin - head) / VectorTokens;
  uint vector_begin = begin + head;
  uint tail_begin = vector_begin + vectors * VectorTokens;
  device const float *row = logits + row_start;
  device const float4 *vector_row =
      reinterpret_cast<device const float4 *>(row + vector_begin);

  float values[K];
  uint ids[K];
  for (uint i = 0; i < K; ++i) {
    values[i] = -INFINITY;
    ids[i] = 0xffffffffu;
  }
  if (thread_index < head) {
    uint token = begin + thread_index;
    float value = row[token];
    if (top_beats(value, token, values[K - 1], ids[K - 1]))
      top16_insert(values, ids, value, token);
  }
  if (thread_index < end - tail_begin) {
    uint token = tail_begin + thread_index;
    float value = row[token];
    if (top_beats(value, token, values[K - 1], ids[K - 1]))
      top16_insert(values, ids, value, token);
  }

  for (uint chunk = 0; chunk < vectors; chunk += 256 * ChunkVectors) {
    float4 loaded[ChunkVectors];
    float best = -INFINITY;
    for (uint i = 0; i < ChunkVectors; ++i) {
      uint index = chunk + thread_index + i * 256;
      loaded[i] = index < vectors ? vector_row[index] : float4(0.0f);
      if (index < vectors) {
        for (uint j = 0; j < VectorTokens; ++j)
          if (loaded[i][j] > best)
            best = loaded[i][j];
      }
    }
    // The 16th largest thread maximum: the minimum over the maxima that
    // fewer than sixteen others exceed.
    maxima[thread_index] = best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint above = 0;
    for (uint other = 0; other < 256; ++other)
      above += maxima[other] > best ? 1u : 0u;
    float candidate = simd_min(above < K ? best : INFINITY);
    if (lane == 0)
      thresholds[simd_group] = candidate;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float threshold = thresholds[0];
    for (uint other = 1; other < 8; ++other)
      threshold = min(threshold, thresholds[other]);

    for (uint i = 0; i < ChunkVectors; ++i) {
      uint index = chunk + thread_index + i * 256;
      if (index >= vectors)
        continue;
      uint token = vector_begin + index * VectorTokens;
      for (uint j = 0; j < VectorTokens; ++j) {
        float value = loaded[i][j];
        if (value >= threshold &&
            top_beats(value, token + j, values[K - 1], ids[K - 1]))
          top16_insert(values, ids, value, token + j);
      }
    }
  }

  // Sixteen rounds pop the group-wide best head; the simdgroup bests
  // alternate between two slots so one barrier per round suffices.
  for (uint rank = 0; rank < K; ++rank) {
    float head_value = values[0];
    uint head_id = ids[0];
    float simd_value;
    uint simd_id;
    simd_best_head(head_value, head_id, simd_value, simd_id);
    uint slot = rank & 1;
    if (lane == 0) {
      round_values[slot * 8 + simd_group] = simd_value;
      round_ids[slot * 8 + simd_group] = simd_id;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float best = round_values[slot * 8];
    uint best_id = round_ids[slot * 8];
    for (uint other = 1; other < 8; ++other) {
      float value = round_values[slot * 8 + other];
      uint token = round_ids[slot * 8 + other];
      if (top_beats(value, token, best, best_id)) {
        best = value;
        best_id = token;
      }
    }
    if (thread_index == rank) {
      partial_ids[group * K + rank] = best_id;
      partial_values[group * K + rank] = best;
    }
    if (head_value == best && head_id == best_id)
      top_pop<K>(values, ids);
  }
}

// The DSpark draft's anchor row already predicts the next token: position p
// reads logits row p, not p + 1.
kernel void dspark_select_top16_sharded(
    device const float *logits [[buffer(0)]],
    device uint *partial_ids [[buffer(1)]],
    device float *partial_values [[buffer(2)]],
    constant uint &vocabulary [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup float maxima[256];
  threadgroup float thresholds[8];
  threadgroup float round_values[2][8];
  threadgroup uint round_ids[2][8];
  draft_top16_sharded_phase(logits, partial_ids, partial_values, vocabulary, 0,
                            maxima, thresholds, &round_values[0][0],
                            &round_ids[0][0], group, thread_index, lane,
                            simd_group);
}

// One group per DSpark proposal row scores every predecessor/candidate edge
// of the merged shard pool: all 128 partial slots the top-16 shards emitted,
// not just their best sixteen. The reference applies the Markov bias over
// the whole vocabulary; the pool covers every token any shard ranked, so a
// low-logit token the bias favors can still win the walk. Position zero's
// only predecessor is the anchor, every other row's predecessors are the
// previous row's pool. Each entry accumulates W2[candidate] . W1[predecessor]
// over the rank in order in one thread, so it rounds exactly as the serial
// walk's per-lane dot did. The table follows the shard partials in
// partial-values scratch.
kernel void dspark_select_edges(
    device const uint *partial_ids [[buffer(0)]],
    device float *partial_values [[buffer(1)]],
    device const bfloat *markov_w1 [[buffer(2)]],
    device const bfloat *markov_w2 [[buffer(3)]],
    constant SelectorBatchParams &params [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  constexpr uint Positions = RICHENGINE_DRAFT_PROPOSAL_TOKENS;
  constexpr uint Shards = RICHENGINE_DRAFT_SAMPLING_SHARDS;
  constexpr uint Candidates = RICHENGINE_DRAFT_CANDIDATES;
  constexpr uint Pool = Shards * Candidates;
  constexpr uint Rank = RICHENGINE_DRAFT_SELECTOR_RANK;
  const uint batch = row / Positions;
  const uint position = row % Positions;
  device float *table =
      partial_values + ulong(params.lanes) * Positions * Pool +
      ulong(row) * Pool * Pool;
  const uint edges = position ? Pool * Pool : Pool;
  for (uint e = thread_index; e < edges; e += 256) {
    const uint predecessor = e / Pool;
    const uint candidate = e % Pool;
    // Ids come from the shard top-k and are always in range; the clamp only
    // keeps a corrupted id inside the Markov tables.
    const uint prev_token = position
        ? partial_ids[(row - 1) * Pool + predecessor]
        : params.anchor[batch];
    const uint safe_predecessor = min(prev_token, params.vocabulary - 1u);
    const uint safe_candidate =
        min(partial_ids[row * Pool + candidate], params.vocabulary - 1u);
    device const bfloat *feature_row =
        markov_w1 + ulong(safe_predecessor) * Rank;
    device const bfloat *bias_row = markov_w2 + ulong(safe_candidate) * Rank;
    float bias = 0.0f;
    for (uint dim = 0; dim < Rank; ++dim)
      bias += float(bias_row[dim]) * float(feature_row[dim]);
    table[predecessor * Pool + candidate] = bias;
  }
}

// One simdgroup per lane walks a DSpark draft's proposal positions in order:
// each position's score is its pool slot's shard score plus the edge the
// parallel dspark_select_edges pass scored for the previously chosen slot —
// the anchor's row for position zero — then the greedy or drawn pick feeds
// the next position, exactly as the serial Markov walk did. Each lane owns
// four of the pool's 128 slots.
kernel void draft_select_dspark(
    device const uint *partial_ids [[buffer(0)]],
    device const float *partial_values [[buffer(1)]],
    device const float *uniforms [[buffer(2)]],
    device float *probabilities [[buffer(3)]],
    device uint *tokens [[buffer(4)]],
    constant SelectorBatchParams &params [[buffer(5)]],
    uint batch [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr ulong Positions = RICHENGINE_DRAFT_PROPOSAL_TOKENS;
  constexpr ulong Shards = RICHENGINE_DRAFT_SAMPLING_SHARDS;
  constexpr ulong Candidates = RICHENGINE_DRAFT_CANDIDATES;
  constexpr uint Pool = Shards * Candidates;
  constexpr uint Owned = Pool / 32;
  device const float *tables =
      partial_values + ulong(params.lanes) * Positions * Pool +
      ulong(batch) * Positions * Pool * Pool;
  const bool sampling = (params.sampling_mask & (1u << batch)) != 0;
  const float temperature = sampling ? params.temperature[batch] : 1.0f;
  uint predecessor_index = 0;
  for (uint position = 0; position < Positions; ++position) {
    const uint row = batch * Positions + position;
    device const float *edge =
        tables + (ulong(position) * Pool + predecessor_index) * Pool;
    float biased[Owned];
    float local_best = -INFINITY;
    uint local_slot = 0;
    for (uint k = 0; k < Owned; ++k) {
      const uint slot = lane + 32 * k;
      biased[k] = partial_values[row * Pool + slot] + edge[slot];
      if (biased[k] > local_best) {
        local_best = biased[k];
        local_slot = slot;
      }
    }

    // Sampled softmax over the pool; the CDF order is lane-major, which is a
    // valid draw order for the same distribution.
    float weights[Owned];
    float weight_sum = 0.0f;
    const float maximum = simd_max(local_best / temperature);
    for (uint k = 0; k < Owned; ++k)
      weight_sum += weights[k] =
          fast::exp((biased[k] / temperature) - maximum);
    const float lane_prefix =
        simd_prefix_inclusive_sum(weight_sum) - weight_sum;
    const float total = simd_sum(weight_sum);
    float run = lane_prefix;
    for (uint k = 0; k < Owned; ++k) {
      const uint slot = lane + 32 * k;
      probabilities[row * Pool + slot] = weights[k] / total;
    }

    uint selected;
    if (sampling) {
      const float uniform =
          uniforms[batch * RICHENGINE_SAMPLING_UNIFORMS +
                   RICHENGINE_UNIFORM_PROPOSALS + position] * total;
      // The crossing must be found by draw position (lane * Owned + k), not
      // slot: simd_min over slots would steal the draw for any later lane's
      // first slot, which is always below 32.
      uint pick = 0xffffffffu;
      for (uint k = 0; k < Owned; ++k) {
        run += weights[k];
        if (pick == 0xffffffffu && run > uniform)
          pick = lane * Owned + k;
      }
      selected = simd_min(pick);
      selected = selected == 0xffffffffu
                     ? Pool - 1
                     : selected / Owned + 32 * (selected % Owned);
    } else {
      const float top = simd_max(local_best);
      selected = simd_min(local_best == top ? local_slot : 0xffffffffu);
      if (selected == 0xffffffffu)
        selected = 0;
    }
    if (lane == 0)
      tokens[row] = partial_ids[row * Pool + selected];
    predecessor_index = selected;
  }
}

// draft_select_dspark plus comb-tree emission: the same chain fills
// tree_tokens/nodes rows 1..7, and lanes in tree_mask additionally record
// each position's biased runner-up under the chain's chosen predecessor as
// a sibling leaf at the fixed row 8 + position (parent = the predecessor's
// chain row), matching draft_select_tree's table layout so the shared
// tree-verify path consumes either draft kind. The pool's runner-up needs a
// second simd pass per position; sampled lanes draw the chain identically
// and emit the degenerate linear table.
kernel void draft_select_dspark_tree(
    device const uint *partial_ids [[buffer(0)]],
    device const float *partial_values [[buffer(1)]],
    device const float *uniforms [[buffer(2)]],
    device float *probabilities [[buffer(3)]],
    device uint *tokens [[buffer(4)]], device uint *tree_tokens [[buffer(5)]],
    device uint *tree_nodes [[buffer(6)]],
    device uint *tree_counts [[buffer(7)]],
    constant SelectorBatchParams &params [[buffer(8)]],
    uint batch [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr ulong Positions = RICHENGINE_DRAFT_PROPOSAL_TOKENS;
  constexpr ulong Shards = RICHENGINE_DRAFT_SAMPLING_SHARDS;
  constexpr ulong Candidates = RICHENGINE_DRAFT_CANDIDATES;
  constexpr uint Pool = Shards * Candidates;
  constexpr uint Owned = Pool / 32;
  // The comb splits the node stride in halves: chain nodes lead it, one leaf
  // per leading position follows.
  constexpr uint ChainRows = RICHENGINE_TREE_VERIFY_NODES / 2;
  device const float *tables =
      partial_values + ulong(params.lanes) * Positions * Pool +
      ulong(batch) * Positions * Pool * Pool;
  tree_tokens += batch * RICHENGINE_TREE_VERIFY_NODES;
  tree_nodes += batch * RICHENGINE_TREE_VERIFY_NODES;
  const bool sampling = (params.sampling_mask & (1u << batch)) != 0;
  const bool tree = (params.tree_mask & (1u << batch)) != 0 && !sampling;
  const float temperature = sampling ? params.temperature[batch] : 1.0f;
  if (lane == 0) {
    tree_tokens[0] = params.anchor[batch];
    tree_nodes[0] = RICHENGINE_TREE_NODE_NONE |
                    (RICHENGINE_TREE_NODE_NONE << 16);
  }
  uint predecessor_index = 0;
  for (uint position = 0; position < Positions; ++position) {
    const uint row = batch * Positions + position;
    device const float *edge =
        tables + (ulong(position) * Pool + predecessor_index) * Pool;
    float biased[Owned];
    float local_best = -INFINITY;
    uint local_slot = 0;
    for (uint k = 0; k < Owned; ++k) {
      const uint slot = lane + 32 * k;
      biased[k] = partial_values[row * Pool + slot] + edge[slot];
      if (biased[k] > local_best) {
        local_best = biased[k];
        local_slot = slot;
      }
    }

    float weights[Owned];
    float weight_sum = 0.0f;
    const float maximum = simd_max(local_best / temperature);
    for (uint k = 0; k < Owned; ++k)
      weight_sum += weights[k] =
          fast::exp((biased[k] / temperature) - maximum);
    const float lane_prefix =
        simd_prefix_inclusive_sum(weight_sum) - weight_sum;
    const float total = simd_sum(weight_sum);
    float run = lane_prefix;
    for (uint k = 0; k < Owned; ++k) {
      const uint slot = lane + 32 * k;
      probabilities[row * Pool + slot] = weights[k] / total;
    }

    uint selected;
    uint runner = Pool;
    if (sampling) {
      const float uniform =
          uniforms[batch * RICHENGINE_SAMPLING_UNIFORMS +
                   RICHENGINE_UNIFORM_PROPOSALS + position] * total;
      uint pick = 0xffffffffu;
      for (uint k = 0; k < Owned; ++k) {
        run += weights[k];
        if (pick == 0xffffffffu && run > uniform)
          pick = lane * Owned + k;
      }
      selected = simd_min(pick);
      selected = selected == 0xffffffffu
                     ? Pool - 1
                     : selected / Owned + 32 * (selected % Owned);
    } else {
      const float top = simd_max(local_best);
      selected = simd_min(local_best == top ? local_slot : 0xffffffffu);
      if (selected == 0xffffffffu)
        selected = 0;
      if (tree) {
        // The runner-up under the chain's predecessor becomes the leaf at
        // this position's fixed row; a dead or sentinel pool slot leaves a
        // NONE node the accept walk skips.
        float local_second = -INFINITY;
        uint second_slot = 0;
        for (uint k = 0; k < Owned; ++k) {
          const uint slot = lane + 32 * k;
          if (slot != selected && biased[k] > local_second) {
            local_second = biased[k];
            second_slot = slot;
          }
        }
        const float second = simd_max(local_second);
        runner = simd_min(local_second == second ? second_slot : Pool);
      }
    }
    if (lane == 0) {
      const uint chain_row = position + 1;
      const uint token = partial_ids[row * Pool + selected];
      if (!tree || chain_row < ChainRows) {
        tree_tokens[chain_row] = token;
        tree_nodes[chain_row] =
            (chain_row - 1) | (chain_row << 8) | (position << 16);
      }
      tokens[row] = token;
      const uint leaf_row = ChainRows + position;
      const uint leaf_token =
          runner < Pool ? partial_ids[row * Pool + runner] : 0xffffffffu;
      if (tree && leaf_row < RICHENGINE_TREE_VERIFY_NODES &&
          leaf_token < params.vocabulary) {
        // The leaf's parent is the chain row that was this position's
        // predecessor (row 0 for position 0, row p for position p).
        tree_tokens[leaf_row] = leaf_token;
        tree_nodes[leaf_row] = position | (chain_row << 8) | (position << 16);
      } else if (tree && leaf_row < RICHENGINE_TREE_VERIFY_NODES) {
        tree_tokens[leaf_row] = 0;
        tree_nodes[leaf_row] = RICHENGINE_TREE_NODE_NONE |
                               (RICHENGINE_TREE_NODE_NONE << 8) |
                               (RICHENGINE_TREE_NODE_NONE << 16);
      }
    }
    predecessor_index = selected;
  }
  if (lane == 0) {
    if (1 + Positions < RICHENGINE_TREE_VERIFY_NODES) {
      tree_tokens[RICHENGINE_TREE_VERIFY_NODES - 1] = 0;
      tree_nodes[RICHENGINE_TREE_VERIFY_NODES - 1] =
          RICHENGINE_TREE_NODE_NONE | (RICHENGINE_TREE_NODE_NONE << 8) |
          (RICHENGINE_TREE_NODE_NONE << 16);
    }
    tree_counts[batch] = min(ulong(1) + Positions,
                             ulong(RICHENGINE_TREE_VERIFY_NODES));
  }
}
