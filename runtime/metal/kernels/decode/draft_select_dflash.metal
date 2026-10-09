#include "metal/abi/KernelABI.h"

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

// Merges the eight shard partials of one row into the simdgroup's lanes: each
// lane holds four consecutive entries of one shard's sorted list and sixteen
// rounds pop the simdgroup-wide best into rank order, lane `rank` keeping it.
inline void top16_merge_shards(device const uint *partial_ids,
                               device const float *partial_values,
                               uint row, uint lane, thread float &value,
                               thread uint &token) {
  constexpr uint K = RICHENGINE_DRAFT_CANDIDATES;
  constexpr uint Shards = RICHENGINE_DRAFT_SAMPLING_SHARDS;
  constexpr uint Entries = Shards * K / 32;
  uint origin = row * Shards * K + lane * Entries;
  float values[Entries];
  uint ids[Entries];
  for (uint i = 0; i < Entries; ++i) {
    values[i] = partial_values[origin + i];
    ids[i] = partial_ids[origin + i];
  }
  value = -INFINITY;
  token = 0xffffffffu;
  for (uint rank = 0; rank < K; ++rank) {
    float best;
    uint best_id;
    simd_best_head(values[0], ids[0], best, best_id);
    if (lane == rank) {
      value = best;
      token = best_id;
    }
    if (values[0] == best && ids[0] == best_id)
      top_pop<Entries>(values, ids);
  }
}

// One group per proposal row. Simdgroup 0 merges its candidates and unary
// scores; simdgroup 1 merges the preceding row (the anchor for row 0). The
// remaining simdgroups score the 16 x 16 predecessor/candidate edge table in
// fixed per-lane reduction order. The table follows the shard partials in
// partial-values scratch.
kernel void draft_select_edges(
    device const uint *partial_ids [[buffer(0)]],
    device float *partial_values [[buffer(1)]],
    device uint *candidates [[buffer(2)]],
    device float *unary [[buffer(3)]],
    device const bfloat *hidden [[buffer(4)]],
    device const bfloat *predecessor_codebook [[buffer(5)]],
    device const bfloat *successor_codebook [[buffer(6)]],
    constant SelectorBatchParams &params [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],
    uint threads [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr uint Rows = RICHENGINE_DRAFT_QUERY_ROWS;
  constexpr uint Positions = RICHENGINE_DRAFT_PROPOSAL_TOKENS;
  constexpr uint Shards = RICHENGINE_DRAFT_SAMPLING_SHARDS;
  constexpr uint Candidates = RICHENGINE_DRAFT_CANDIDATES;
  constexpr uint Rank = RICHENGINE_DRAFT_SELECTOR_RANK;
  uint batch = row / Positions;
  uint position = row % Positions;
  threadgroup uint successors[Candidates];
  threadgroup uint predecessors[Candidates];
  if (simd_group == 0) {
    float value;
    uint token;
    top16_merge_shards(partial_ids, partial_values, row, lane, value, token);
    if (lane < Candidates) {
      candidates[row * Candidates + lane] = token;
      unary[row * Candidates + lane] = value;
      successors[lane] = token;
    }
  } else if (simd_group == 1) {
    uint token = params.anchor[batch];
    if (position > 0) {
      float value;
      top16_merge_shards(partial_ids, partial_values, row - 1, lane, value,
                         token);
    }
    if (lane < Candidates)
      predecessors[lane] = token;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // One task is one predecessor and eight of the sixteen candidates, so a
  // simdgroup issues all its codebook loads at once instead of one cold row
  // per edge.
  constexpr uint TaskCandidates = 8, Dims = Rank / 32;
  uint tasks = (position > 0 ? Candidates : 1) * (Candidates / TaskCandidates);
  device float *table = partial_values +
                        ulong(params.lanes) * Positions * Shards * Candidates +
                        ulong(row) * Candidates * Candidates;
  device const bfloat *row_hidden =
      hidden + (ulong(batch) * Rows + position + 1) * Rank;
  for (uint task = simd_group; task < tasks; task += threads / 32) {
    uint predecessor_index = task / (Candidates / TaskCandidates);
    uint first_candidate =
        task % (Candidates / TaskCandidates) * TaskCandidates;
    // Ids are produced by the top-k selection and are always in range; the
    // clamp only keeps a corrupted id inside the codebooks.
    uint safe_predecessor =
        min(predecessors[predecessor_index], params.vocabulary - 1u);
    float context[Dims];
    float successor[TaskCandidates][Dims];
    for (uint i = 0; i < Dims; ++i) {
      uint dim = lane + i * 32;
      context[i] = float(predecessor_codebook[safe_predecessor * Rank + dim]) *
                   float(row_hidden[dim]);
    }
    for (uint j = 0; j < TaskCandidates; ++j) {
      uint safe_candidate =
          min(successors[first_candidate + j], params.vocabulary - 1u);
      for (uint i = 0; i < Dims; ++i)
        successor[j][i] =
            float(successor_codebook[safe_candidate * Rank + lane + i * 32]);
    }
    for (uint j = 0; j < TaskCandidates; ++j) {
      float score = 0.0f;
      for (uint i = 0; i < Dims; ++i)
        score += context[i] * successor[j][i];
      score = simd_sum(score);
      if (lane == 0)
        table[predecessor_index * Candidates + first_candidate + j] = score;
    }
  }
}

// The pool-wide variant of draft_select_edges: the same hidden-conditioned
// codebook edge, but over every shard partial slot (the 128-entry pool the
// merged top-16 truncates away), matching dspark_select_edges' table shape
// so the shared pool walk consumes either draft kind. No merge is needed —
// predecessors and successors are read straight from the shard partials,
// with the anchor as position zero's only predecessor. A simdgroup still
// takes one predecessor's eight-candidate slice per task so the dot rounds
// in the same fixed per-lane order as the merged table's.
kernel void dflash_select_pool_edges(
    device const uint *partial_ids [[buffer(0)]],
    device float *partial_values [[buffer(1)]],
    device const bfloat *hidden [[buffer(2)]],
    device const bfloat *predecessor_codebook [[buffer(3)]],
    device const bfloat *successor_codebook [[buffer(4)]],
    constant SelectorBatchParams &params [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint threads [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr uint Rows = RICHENGINE_DRAFT_QUERY_ROWS;
  constexpr uint Positions = RICHENGINE_DRAFT_PROPOSAL_TOKENS;
  constexpr uint Shards = RICHENGINE_DRAFT_SAMPLING_SHARDS;
  constexpr uint Candidates = RICHENGINE_DRAFT_CANDIDATES;
  constexpr uint Pool = Shards * Candidates;
  constexpr uint Rank = RICHENGINE_DRAFT_SELECTOR_RANK;
  constexpr uint TaskCandidates = 8, Dims = Rank / 32;
  const uint batch = row / Positions;
  const uint position = row % Positions;
  device float *table = partial_values +
                        ulong(params.lanes) * Positions * Pool +
                        ulong(row) * Pool * Pool;
  device const bfloat *row_hidden =
      hidden + (ulong(batch) * Rows + position + 1) * Rank;
  const uint tasks = (position ? Pool : 1u) * (Pool / TaskCandidates);
  for (uint task = simd_group; task < tasks; task += threads / 32) {
    const uint predecessor_index = task / (Pool / TaskCandidates);
    const uint first_candidate =
        task % (Pool / TaskCandidates) * TaskCandidates;
    const uint previous =
        position ? partial_ids[(row - 1) * Pool + predecessor_index]
                 : params.anchor[batch];
    const uint safe_predecessor = min(previous, params.vocabulary - 1u);
    float context[Dims];
    float successor[TaskCandidates][Dims];
    for (uint i = 0; i < Dims; ++i) {
      uint dim = lane + i * 32;
      context[i] = float(predecessor_codebook[safe_predecessor * Rank + dim]) *
                   float(row_hidden[dim]);
    }
    for (uint j = 0; j < TaskCandidates; ++j) {
      const uint safe_candidate =
          min(partial_ids[row * Pool + first_candidate + j],
              params.vocabulary - 1u);
      for (uint i = 0; i < Dims; ++i)
        successor[j][i] =
            float(successor_codebook[safe_candidate * Rank + lane + i * 32]);
    }
    for (uint j = 0; j < TaskCandidates; ++j) {
      float score = 0.0f;
      for (uint i = 0; i < Dims; ++i)
        score += context[i] * successor[j][i];
      score = simd_sum(score);
      if (lane == 0)
        table[predecessor_index * Pool + first_candidate + j] = score;
    }
  }
}

// One thread per lane walks the seven positions: the score of a candidate is
// its unary score plus the edge from the previously chosen candidate, read
// from the table draft_select_edges left in the partial-values scratch.
kernel void draft_select_dflash(
    device const uint *candidates [[buffer(0)]],
    device const float *unary [[buffer(1)]],
    device const float *partial_values [[buffer(2)]],
    device const float *uniforms [[buffer(3)]],
    device uint *tokens [[buffer(4)]], device float *q_probs [[buffer(5)]],
    constant SelectorBatchParams &params [[buffer(6)]],
    uint batch [[thread_position_in_grid]]) {
  constexpr ulong Positions = RICHENGINE_DRAFT_PROPOSAL_TOKENS;
  constexpr ulong Shards = RICHENGINE_DRAFT_SAMPLING_SHARDS;
  constexpr ulong Candidates = RICHENGINE_DRAFT_CANDIDATES;
  candidates += batch * Positions * Candidates;
  unary += batch * Positions * Candidates;
  device const float *tables = partial_values +
                               params.lanes * Positions * Shards * Candidates +
                               batch * Positions * Candidates * Candidates;
  uniforms += batch * RICHENGINE_SAMPLING_UNIFORMS;
  tokens += batch * Positions;
  q_probs += batch * Positions * Candidates;

  const bool sampling = (params.sampling_mask & (1u << batch)) != 0;
  uint predecessor_index = 0;
  for (uint position = 0; position < Positions; ++position) {
    device const float *edges =
        tables + (position * Candidates + predecessor_index) * Candidates;
    float scores[Candidates];
    for (uint i = 0; i < Candidates; ++i)
      scores[i] = unary[position * Candidates + i] + edges[i];
    uint selected = 0;
    if (sampling) {
      float maximum = scores[0];
      for (uint i = 1; i < Candidates; ++i)
        maximum = max(maximum, scores[i]);
      float sum = 0.0f;
      for (uint i = 0; i < Candidates; ++i) {
        float probability =
            exp((scores[i] - maximum) / params.temperature[batch]);
        q_probs[position * Candidates + i] = probability;
        sum += probability;
      }
      float cumulative = 0.0f;
      selected = Candidates - 1;
      for (uint i = 0; i < Candidates; ++i) {
        float probability = q_probs[position * Candidates + i] / sum;
        q_probs[position * Candidates + i] = probability;
        cumulative += probability;
        if (selected == Candidates - 1 &&
            cumulative > uniforms[RICHENGINE_UNIFORM_PROPOSALS + position]) {
          selected = i;
        }
      }
    } else {
      for (uint i = 1; i < Candidates; ++i) {
        if (scores[i] > scores[selected])
          selected = i;
      }
    }
    predecessor_index = selected;
    tokens[position] = candidates[position * Candidates + selected];
  }
}

// draft_select_dflash plus tree emission: the same chain fills
// tree_tokens/nodes rows 1..RICHENGINE_DRAFT_PROPOSAL_TOKENS for a chain
// lane, while a lane in tree_mask fills a comb: the chain holds the front
// half of the node block (rows 0..Nodes/2-1) and each of its positions'
// runner-up candidate under the chain's chosen predecessor becomes a
// sibling leaf in the back half (row Nodes/2 + position, parent = the
// predecessor's chain row). tokens[] still receives the full chain so every
// chain-mode consumer is unchanged; tree_counts is 1 + Positions for a
// chain lane and fills the node stride for a tree lane.
kernel void draft_select_tree(
    device const uint *candidates [[buffer(0)]],
    device const float *unary [[buffer(1)]],
    device const float *partial_values [[buffer(2)]],
    device const float *uniforms [[buffer(3)]],
    device uint *tokens [[buffer(4)]], device float *q_probs [[buffer(5)]],
    device uint *tree_tokens [[buffer(6)]],
    device uint *tree_nodes [[buffer(7)]],
    device uint *tree_counts [[buffer(8)]],
    constant SelectorBatchParams &params [[buffer(9)]],
    uint batch [[thread_position_in_grid]]) {
  constexpr ulong Positions = RICHENGINE_DRAFT_PROPOSAL_TOKENS;
  constexpr ulong Shards = RICHENGINE_DRAFT_SAMPLING_SHARDS;
  constexpr ulong Candidates = RICHENGINE_DRAFT_CANDIDATES;
  // The comb splits the node stride in halves: chain nodes lead it, one leaf
  // per leading position follows.
  constexpr uint ChainRows = RICHENGINE_TREE_VERIFY_NODES / 2;
  candidates += batch * Positions * Candidates;
  unary += batch * Positions * Candidates;
  device const float *tables = partial_values +
                               params.lanes * Positions * Shards * Candidates +
                               batch * Positions * Candidates * Candidates;
  uniforms += batch * RICHENGINE_SAMPLING_UNIFORMS;
  tokens += batch * Positions;
  q_probs += batch * Positions * Candidates;
  tree_tokens += batch * RICHENGINE_TREE_VERIFY_NODES;
  tree_nodes += batch * RICHENGINE_TREE_VERIFY_NODES;

  const bool sampling = (params.sampling_mask & (1u << batch)) != 0;
  const bool tree = (params.tree_mask & (1u << batch)) != 0 && !sampling;
  tree_tokens[0] = params.anchor[batch];
  tree_nodes[0] = RICHENGINE_TREE_NODE_NONE |
                  (RICHENGINE_TREE_NODE_NONE << 16);

  uint predecessor_index = 0;
  for (uint position = 0; position < Positions; ++position) {
    device const float *edges =
        tables + (position * Candidates + predecessor_index) * Candidates;
    float scores[Candidates];
    for (uint i = 0; i < Candidates; ++i)
      scores[i] = unary[position * Candidates + i] + edges[i];
    uint selected = 0;
    uint runner = Candidates;
    if (sampling) {
      float maximum = scores[0];
      for (uint i = 1; i < Candidates; ++i)
        maximum = max(maximum, scores[i]);
      float sum = 0.0f;
      for (uint i = 0; i < Candidates; ++i) {
        float probability =
            exp((scores[i] - maximum) / params.temperature[batch]);
        q_probs[position * Candidates + i] = probability;
        sum += probability;
      }
      float cumulative = 0.0f;
      selected = Candidates - 1;
      for (uint i = 0; i < Candidates; ++i) {
        float probability = q_probs[position * Candidates + i] / sum;
        q_probs[position * Candidates + i] = probability;
        cumulative += probability;
        if (selected == Candidates - 1 &&
            cumulative > uniforms[RICHENGINE_UNIFORM_PROPOSALS + position]) {
          selected = i;
        }
      }
    } else {
      for (uint i = 1; i < Candidates; ++i) {
        if (scores[i] > scores[selected])
          selected = i;
      }
      // The runner-up under the chain's predecessor becomes the leaf at this
      // position's fixed row (8 + position); it can only matter when the
      // chain pick is rejected.
      for (uint i = 0; i < Candidates; ++i) {
        if (i != selected &&
            (runner == Candidates || scores[i] > scores[runner]))
          runner = i;
      }
    }
    const uint chain_row = position + 1;
    if (!tree || chain_row < ChainRows) {
      tree_tokens[chain_row] = candidates[position * Candidates + selected];
      tree_nodes[chain_row] = (chain_row - 1) | (chain_row << 8) |
                              (position << 16);
    }
    predecessor_index = selected;
    tokens[position] = candidates[position * Candidates + selected];
    // The runner-up under the chain's predecessor becomes the leaf in the
    // comb's back half; a chain lane's block holds chain rows instead.
    const uint leaf_row = ChainRows + position;
    if (tree && leaf_row < RICHENGINE_TREE_VERIFY_NODES &&
        runner != Candidates) {
      tree_tokens[leaf_row] = candidates[position * Candidates + runner];
      // The leaf's parent is the chain node that was this position's
      // predecessor (row 0 for position 0, row p for position p).
      tree_nodes[leaf_row] = position | (chain_row << 8) | (position << 16);
    } else if (tree && leaf_row < RICHENGINE_TREE_VERIFY_NODES) {
      tree_tokens[leaf_row] = 0;
      tree_nodes[leaf_row] = RICHENGINE_TREE_NODE_NONE |
                             (RICHENGINE_TREE_NODE_NONE << 8) |
                             (RICHENGINE_TREE_NODE_NONE << 16);
    }
  }
  if (1 + Positions < RICHENGINE_TREE_VERIFY_NODES) {
    tree_tokens[RICHENGINE_TREE_VERIFY_NODES - 1] = 0;
    tree_nodes[RICHENGINE_TREE_VERIFY_NODES - 1] =
        RICHENGINE_TREE_NODE_NONE | (RICHENGINE_TREE_NODE_NONE << 8) |
        (RICHENGINE_TREE_NODE_NONE << 16);
  }
  tree_counts[batch] = min(ulong(1) + Positions,
                           ulong(RICHENGINE_TREE_VERIFY_NODES));
}
