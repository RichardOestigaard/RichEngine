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

// The plain DFlash draft's per-position policy, the DFlash2 walk without the
// codebook edges: one simdgroup per proposal row merges the top-16 partials
// into the candidate list, and its greedy lane takes the best while a
// sampled lane draws over the candidates' softmax weights, which acceptance
// reads as the draft probabilities.
kernel void draft_select_plain(
    device const uint *partial_ids [[buffer(0)]],
    device const float *partial_values [[buffer(1)]],
    device const float *uniforms [[buffer(2)]],
    device uint *candidates [[buffer(3)]],
    device float *probabilities [[buffer(4)]],
    device uint *tokens [[buffer(5)]],
    constant SelectorBatchParams &params [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr ulong Positions = RICHENGINE_DRAFT_PROPOSAL_TOKENS;
  constexpr ulong Candidates = RICHENGINE_DRAFT_CANDIDATES;
  const uint batch = row / Positions;
  const uint position = row % Positions;
  float value;
  uint token;
  top16_merge_shards(partial_ids, partial_values, row, lane, value, token);
  if (lane < Candidates)
    candidates[row * Candidates + lane] = token;

  const bool sampling = (params.sampling_mask & (1u << batch)) != 0;
  const float temperature = sampling ? params.temperature[batch] : 1.0f;
  const float scaled = lane < Candidates ? value / temperature : -INFINITY;
  const float maximum = simd_max(scaled);
  const float weight =
      lane < Candidates ? fast::exp(scaled - maximum) : 0.0f;
  const float probability = weight / simd_sum(weight);
  if (lane < Candidates)
    probabilities[row * Candidates + lane] = probability;

  uint selected = 0;
  if (sampling) {
    const float uniform = uniforms[batch * RICHENGINE_SAMPLING_UNIFORMS +
                                   RICHENGINE_UNIFORM_PROPOSALS + position];
    const float prefix = simd_prefix_inclusive_sum(probability);
    const bool hit =
        lane < Candidates && prefix - probability <= uniform && prefix > uniform;
    selected = simd_min(hit ? lane : 0xffffffffu);
    if (selected == 0xffffffffu)
      selected = Candidates - 1;
  }
  const uint chosen = simd_broadcast(token, selected);
  if (lane == 0)
    tokens[batch * Positions + position] = chosen;
}

// draft_select_plain plus comb-tree emission: the per-row merge, sampling
// draw and tokens[] write are identical, and lanes in tree_mask also fill
// the node's block — the chain in its front half, the position's rank-1
// merged candidate as a sibling leaf in the back half (row Nodes/2 +
// position, parent = the position's predecessor chain row). The plain draft
// has no predecessor edges, so a leaf is just the position's runner-up.
kernel void draft_select_plain_tree(
    device const uint *partial_ids [[buffer(0)]],
    device const float *partial_values [[buffer(1)]],
    device const float *uniforms [[buffer(2)]],
    device uint *candidates [[buffer(3)]],
    device float *probabilities [[buffer(4)]],
    device uint *tokens [[buffer(5)]],
    device uint *tree_tokens [[buffer(6)]],
    device uint *tree_nodes [[buffer(7)]],
    device uint *tree_counts [[buffer(8)]],
    constant SelectorBatchParams &params [[buffer(9)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr ulong Positions = RICHENGINE_DRAFT_PROPOSAL_TOKENS;
  constexpr ulong Candidates = RICHENGINE_DRAFT_CANDIDATES;
  constexpr uint Nodes = RICHENGINE_TREE_VERIFY_NODES;
  constexpr uint ChainRows = Nodes / 2;
  const uint batch = row / Positions;
  const uint position = row % Positions;
  float value;
  uint token;
  top16_merge_shards(partial_ids, partial_values, row, lane, value, token);
  if (lane < Candidates)
    candidates[row * Candidates + lane] = token;

  const bool sampling = (params.sampling_mask & (1u << batch)) != 0;
  const bool tree = (params.tree_mask & (1u << batch)) != 0 && !sampling;
  const float temperature = sampling ? params.temperature[batch] : 1.0f;
  const float scaled = lane < Candidates ? value / temperature : -INFINITY;
  const float maximum = simd_max(scaled);
  const float weight =
      lane < Candidates ? fast::exp(scaled - maximum) : 0.0f;
  const float probability = weight / simd_sum(weight);
  if (lane < Candidates)
    probabilities[row * Candidates + lane] = probability;

  uint selected = 0;
  if (sampling) {
    const float uniform = uniforms[batch * RICHENGINE_SAMPLING_UNIFORMS +
                                   RICHENGINE_UNIFORM_PROPOSALS + position];
    const float prefix = simd_prefix_inclusive_sum(probability);
    const bool hit =
        lane < Candidates && prefix - probability <= uniform && prefix > uniform;
    selected = simd_min(hit ? lane : 0xffffffffu);
    if (selected == 0xffffffffu)
      selected = Candidates - 1;
  }
  const uint chosen = simd_broadcast(token, selected);
  const uint runner_up = simd_broadcast(token, 1);
  if (lane == 0) {
    tokens[batch * Positions + position] = chosen;
    device uint *tt = tree_tokens + batch * Nodes;
    device uint *tn = tree_nodes + batch * Nodes;
    if (position == 0) {
      tt[0] = params.anchor[batch];
      tn[0] = RICHENGINE_TREE_NODE_NONE | (RICHENGINE_TREE_NODE_NONE << 16);
      tree_counts[batch] = min(ulong(1) + Positions, ulong(Nodes));
    }
    const uint chain_row = position + 1;
    if (!tree || chain_row < ChainRows) {
      tt[chain_row] = chosen;
      tn[chain_row] = (chain_row - 1) | (chain_row << 8) | (position << 16);
    }
    const uint leaf_row = ChainRows + position;
    if (tree && leaf_row < Nodes && runner_up != 0xffffffffu &&
        runner_up != chosen) {
      tt[leaf_row] = runner_up;
      tn[leaf_row] = position | (chain_row << 8) | (position << 16);
    } else if (tree && leaf_row < Nodes) {
      tt[leaf_row] = 0;
      tn[leaf_row] = RICHENGINE_TREE_NODE_NONE |
                     (RICHENGINE_TREE_NODE_NONE << 8) |
                     (RICHENGINE_TREE_NODE_NONE << 16);
    }
  }
}
