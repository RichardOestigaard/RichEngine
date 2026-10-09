#include "metal/abi/Gguf.h"
#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/activation.h"
#include "metal/kernels/common/q4_mpp_tiles.h"
#include "metal/kernels/common/split_reduce.h"

// The rows of the target policy. A greedy lane takes each row's argmax. A
// sampled lane draws from each row's min-p/top-k/top-p distribution over the
// whole vocabulary: a sharded scan sums the row's softmax denominator, one
// group per row finds the last token the distribution keeps in the order of
// the logits (logit descending, id ascending) without sorting the
// vocabulary, and the draw takes one pass over the kept tokens in id order,
// shared by the row's kVocabularyGroups groups: each sums ranges of the
// vocabulary, and the one that finishes last (split_arrive_last) draws.
// Every reduction runs in a fixed order, so a row selects the same token on
// every run. The same kernels select the first token after a prompt and the
// verify rows (TargetSamplingParams): each dispatch covers the selected rows
// of every lane, and the groups of the other policy's lanes return at once.

// The row a lane selects from: its logits, the tokens it admits (its
// constraint mask row, less the stop tokens when the lane ignores
// end-of-sequence) and, for a sampled row whose maximum is known, their
// softmax weights.
struct TargetRow {
  device const float *logits;
  device const uint *mask;
  bool constrained;
  bool exclude_stop;
  uint stop_token_0;
  uint stop_token_1;
  uint vocabulary;
  float temperature;
  float maximum;

  bool admits(uint token) const {
    if (constrained && (mask[token / 32] & (1u << (token % 32))) == 0)
      return false;
    return !exclude_stop || (token != stop_token_0 && token != stop_token_1);
  }
  float weight(float value) const {
    return exp((value - maximum) / temperature);
  }
};

// The lane of selected row s, and whether it samples.
inline uint selected_lane(constant TargetSamplingParams &params, uint s) {
  return s / params.rows;
}
inline bool lane_samples(constant TargetSamplingParams &params, uint s) {
  return (params.sampling_mask & (1u << selected_lane(params, s))) != 0;
}
// A selected row past its lane's live verify rows (adaptive proposal
// budgets): the selection kernels skip it and acceptance never reaches it.
inline bool row_dead(constant TargetSamplingParams &params, uint s) {
  const uint live = params.live_rows[selected_lane(params, s)];
  return live && s % params.rows >= live;
}

// Selected row s of a dispatch (TargetSamplingParams). The per-lane stride of
// the logits and mask buffers is the larger of the chain's eight rows and
// the dispatch's rows, which a tree batch widens to
// RICHENGINE_TREE_VERIFY_NODES.
inline TargetRow selected_row(device const float *logits,
                              device const uint *token_mask,
                              constant TargetSamplingParams &params, uint s) {
  const uint lane = selected_lane(params, s);
  const uint index = s % params.rows;
  const uint stride = max(params.rows, RICHENGINE_TARGET_VERIFY_ROWS);
  return {logits + (ulong(lane) * stride + params.logits_row + index) *
                       params.vocabulary,
          token_mask + (ulong(lane) * (stride + 1) + params.mask_row + index) *
                           params.mask_words,
          (params.constrained_mask & (1u << lane)) != 0,
          (params.exclude_stop_mask & (1u << lane)) != 0,
          params.stop_token_0,
          params.stop_token_1,
          params.vocabulary,
          params.temperature[lane],
          0.0f};
}

// Shares of a row's softmax denominator combined in order: their largest
// maximum, their sums rescaled to it and their admitted counts. A group
// combines its simdgroups' shares, and a row's search its shards'.
template <class Shares>
inline TargetShardMass merge_masses(Shares shares, uint count,
                                    float temperature) {
  TargetShardMass total{-FLT_MAX, 0.0f, 0};
  for (uint index = 0; index < count; ++index)
    total.maximum = max(total.maximum, shares[index].maximum);
  for (uint index = 0; index < count; ++index) {
    const TargetShardMass share = shares[index];
    if (share.sum > 0.0f)
      total.sum +=
          share.sum * exp((share.maximum - total.maximum) / temperature);
    total.admitted += share.admitted;
  }
  return total;
}

// One shard's share of a sampled row's softmax denominator: each thread
// keeps a running maximum and sum of its admitted tokens' weights (an online
// softmax), combined over the group in a fixed order. The running maximum
// starts below every finite logit, so a -inf logit weighs nothing.
inline void shard_mass(TargetRow row, uint shard,
                       device TargetShardMass &partial,
                       threadgroup TargetShardMass *group_masses,
                       uint thread_index, uint lane, uint simd_group) {
  constexpr uint Shards = RICHENGINE_TARGET_SAMPLING_SHARDS;
  TargetShardMass mass{-FLT_MAX, 0.0f, 0};
  for (uint token = shard * 256 + thread_index; token < row.vocabulary;
       token += Shards * 256) {
    if (!row.admits(token))
      continue;
    const float value = row.logits[token];
    if (value > mass.maximum) {
      mass.sum =
          mass.sum * exp((mass.maximum - value) / row.temperature) + 1.0f;
      mass.maximum = value;
    } else {
      mass.sum += exp((value - mass.maximum) / row.temperature);
    }
    ++mass.admitted;
  }
  const float simd_maximum = simd_max(mass.maximum);
  const float scaled =
      mass.sum > 0.0f
          ? mass.sum * exp((mass.maximum - simd_maximum) / row.temperature)
          : 0.0f;
  const float simd_total = simd_sum(scaled);
  const uint simd_admitted = simd_sum(mass.admitted);
  if (lane == 0)
    group_masses[simd_group] = {simd_maximum, simd_total, simd_admitted};
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index == 0)
    partial = merge_masses(group_masses, 8, row.temperature);
}

// One shard of each selected row of the sampled lanes.
kernel void decode_sample_mass_sharded(
    device const float *logits [[buffer(0)]],
    device const uint *token_mask [[buffer(1)]],
    device TargetShardMass *partial_masses [[buffer(2)]],
    constant TargetSamplingParams &params [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup TargetShardMass group_masses[8];
  const uint s = group / RICHENGINE_TARGET_SAMPLING_SHARDS;
  if (!lane_samples(params, s) || row_dead(params, s))
    return;
  shard_mass(selected_row(logits, token_mask, params, s),
             group % RICHENGINE_TARGET_SAMPLING_SHARDS, partial_masses[group],
             group_masses, thread_index, lane, simd_group);
}

// The groups that search and draw a sampled row.
constant constexpr uint kVocabularyThreads = RICHENGINE_TARGET_VOCABULARY_THREADS;
constant constexpr uint kVocabularySimdgroups = kVocabularyThreads / 32;
constant constexpr uint kVocabularyGroups = RICHENGINE_TARGET_VOCABULARY_GROUPS;
// The draw's ranges of the vocabulary; the last group loads them one per
// thread.
constant constexpr uint kVocabularyRanges = RICHENGINE_TARGET_VOCABULARY_RANGES;
static_assert(kVocabularyRanges <= kVocabularyThreads,
              "a group holds one draw range per thread");
// Pivots per pass: each pass splits a bracket sixteen ways. In logit order
// kKeyPivots of them split its keys evenly and the others follow the logits'
// scale (place_pivots).
constant constexpr uint kPivots = 15;
constant constexpr uint kKeyPivots = 4;
constant constexpr uint kDraftCandidates = RICHENGINE_DRAFT_CANDIDATES;

// The order of the logits as an unsigned key: a larger logit has a larger
// key, and -0 and +0, which compare equal, share one. Every logit's key is
// above kNoKey.
inline uint logit_key(float value) {
  uint bits = as_type<uint>(value);
  bits = (bits << 1) ? bits : 0u;
  return (bits & 0x80000000u) ? ~bits : bits | 0x80000000u;
}
inline float key_logit(uint key) {
  return as_type<float>((key & 0x80000000u) ? key & 0x7fffffffu : ~key);
}
constant constexpr uint kNoKey = 0u;
// The key of -infinity, at or below that of every admitted logit.
constant constexpr uint kLowestKey = 0x007fffffu;

// A position in the order of the logits: the tokens at or before it have a
// larger key, or the same key and an id at most last. {kNoKey, 0} lies
// after every token.
struct OrderBoundary {
  uint key;
  uint last;
};

inline bool at_or_before(uint key, uint token, OrderBoundary boundary) {
  return key > boundary.key || (key == boundary.key && token <= boundary.last);
}

// The orders a search runs in. The logit order is the order of the logits.
// Among the tokens tied at one logit, the tie order puts a smaller id first;
// every other token has kNoKey.
struct LogitOrder {
  static constexpr constant bool by_logit = true;
  uint key(float value, uint) const { return logit_key(value); }
  float logit(uint key) const { return key_logit(key); }
};
struct TieOrder {
  uint tied_key;
  uint vocabulary;
  uint key(float value, uint token) const {
    return logit_key(value) == tied_key ? vocabulary - token : kNoKey;
  }
  static constexpr constant bool by_logit = false;
  float logit(uint) const { return key_logit(tied_key); }
};

// What a search keeps: the tokens in order until their count, or their mass
// (the sum of their weights), exceeds the target. A count search measures
// counts alone (top-k, and the tie order's first ids), a mass search counts
// and masses (top-p).
struct CountTarget {
  static constexpr constant bool by_mass = false;
  uint count;

  bool exceeded(uint measured_count, float) const {
    return measured_count > count;
  }
};
struct MassTarget {
  static constexpr constant bool by_mass = true;
  float mass;

  bool exceeded(uint, float measured_mass) const {
    return measured_mass > mass;
  }
};

// A key range [lo, hi) and the count and mass of the tokens with at least
// each end's key: lo's exceed the search target, hi's do not. A count
// search narrows the counts alone and leaves the masses as they came in.
struct Bracket {
  uint lo;
  uint hi;
  uint lo_count;
  uint hi_count;
  float lo_mass;
  float hi_mass;

  uint tokens() const { return lo_count - hi_count; }
};

// The last token a search keeps, and the count and mass of the tokens at or
// before it. A ranked selection leaves those tokens in the scratch keys and
// ids, in order (resolve).
struct Selection {
  OrderBoundary last;
  uint count;
  float mass;
  bool ranked;
};

struct VocabularyScratch {
  uint counts[kVocabularySimdgroups][kPivots];
  float masses[kVocabularySimdgroups][kPivots];
  uint total_counts[kPivots];
  float total_masses[kPivots];
  atomic_uint gathered;
  uint keys[kVocabularyThreads];
  uint ids[kVocabularyThreads];
  Selection selection;
};

struct DrawScratch {
  uint arrival;
  // Per range of the draw: the kept weight of its tokens other than the
  // draft's candidates, one past the last of those with weight, and its
  // weight in the draw.
  float range_rest[kVocabularyRanges];
  uint range_last[kVocabularyRanges];
  float range_weights[kVocabularyRanges];
  // The draft's candidates, each listed once: its draft probability, kept
  // weight, weight in the draw and range.
  uint draft_ids[kDraftCandidates];
  float draft_probabilities[kDraftCandidates];
  float draft_weights[kDraftCandidates];
  float draft_draw_weights[kDraftCandidates];
  uint draft_ranges[kDraftCandidates];
  float kept_mass;
  uint drawn_range;
  float drawn_target;
  uint drawn_token;
};

// Pivots inside [lo, hi) that split the bracket. The tie order splits its
// keys evenly. In logit order the last kKeyPivots do, so a pass keeps at
// most a fifth of the bracket's keys, rounded up, and a search ends within
// fourteen passes whatever the logits, temperature or penalties; the others
// follow the logits' scale, which ends most searches within a few: geometric
// distances below the bracket's top, in units of the temperature, while it
// is open at the bottom, and even steps in logit value once it is not.
template <class Order>
inline void place_pivots(Order order, Bracket bracket, float temperature,
                         thread uint (&pivots)[kPivots]) {
  const ulong width = bracket.hi - bracket.lo;
  if (!order.by_logit) {
    for (uint pivot = 0; pivot < kPivots; ++pivot)
      pivots[pivot] = bracket.lo + uint(width * (pivot + 1) / (kPivots + 1));
    return;
  }
  constexpr uint kScaled = kPivots - kKeyPivots;
  const float top = key_logit(bracket.hi - 1);
  const float bottom = key_logit(bracket.lo);
  const bool open = bracket.lo == kLowestKey;
  for (uint pivot = 0; pivot < kScaled; ++pivot) {
    const float value =
        open ? top - temperature * 0.5f * exp2(float(kScaled - 1 - pivot))
             : bottom + (top - bottom) * float(pivot + 1) / float(kScaled + 1);
    pivots[pivot] = clamp(logit_key(value), bracket.lo, bracket.hi - 1);
  }
  for (uint pivot = 0; pivot < kKeyPivots; ++pivot)
    pivots[kScaled + pivot] =
        bracket.lo + uint(width * (pivot + 1) / (kKeyPivots + 1));
}

// The count and, with Masses, the mass of the admitted tokens at or before
// the floor whose keys are at least each pivot's, summed per thread in token
// order and then over the group in a fixed order. Counts alone weigh no
// token.
template <bool Masses, uint Pivots, class Order>
inline void measure_pivots(TargetRow row, Order order, OrderBoundary floor,
                           thread const uint (&pivots)[Pivots],
                           threadgroup VocabularyScratch &scratch,
                           uint thread_index, uint lane, uint simd_group) {
  uint counts[Pivots];
  float masses[Pivots];
  uint lowest = pivots[0];
  for (uint pivot = 0; pivot < Pivots; ++pivot) {
    counts[pivot] = 0;
    if constexpr (Masses)
      masses[pivot] = 0.0f;
    lowest = min(lowest, pivots[pivot]);
  }
  for (uint token = thread_index; token < row.vocabulary;
       token += kVocabularyThreads) {
    const float value = row.logits[token];
    const uint key = order.key(value, token);
    // Tokens below every pivot add nothing.
    if (key < lowest || !row.admits(token) ||
        !at_or_before(logit_key(value), token, floor))
      continue;
    if constexpr (Masses) {
      const float weight = row.weight(value);
      for (uint pivot = 0; pivot < Pivots; ++pivot) {
        const bool above = key >= pivots[pivot];
        counts[pivot] += above ? 1u : 0u;
        masses[pivot] += above ? weight : 0.0f;
      }
    } else {
      for (uint pivot = 0; pivot < Pivots; ++pivot)
        counts[pivot] += key >= pivots[pivot] ? 1u : 0u;
    }
  }
  for (uint pivot = 0; pivot < Pivots; ++pivot) {
    const uint count = simd_sum(counts[pivot]);
    if (lane == 0)
      scratch.counts[simd_group][pivot] = count;
    if constexpr (Masses) {
      const float mass = simd_sum(masses[pivot]);
      if (lane == 0)
        scratch.masses[simd_group][pivot] = mass;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index < Pivots) {
    uint count = 0;
    for (uint simd = 0; simd < kVocabularySimdgroups; ++simd)
      count += scratch.counts[simd][thread_index];
    scratch.total_counts[thread_index] = count;
    if constexpr (Masses) {
      float mass = 0.0f;
      for (uint simd = 0; simd < kVocabularySimdgroups; ++simd)
        mass += scratch.masses[simd][thread_index];
      scratch.total_masses[thread_index] = mass;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// The count and mass of the admitted tokens at or before the floor whose
// logits' keys are at least key, summed as measure_pivots sums a pivot's.
struct Measure {
  uint count;
  float mass;
};
inline Measure measure_at_or_above(TargetRow row, OrderBoundary floor,
                                   uint key,
                                   threadgroup VocabularyScratch &scratch,
                                   uint thread_index, uint lane,
                                   uint simd_group) {
  const uint pivots[1] = {key};
  measure_pivots<true>(row, LogitOrder{}, floor, pivots, scratch,
                       thread_index, lane, simd_group);
  const Measure measure{scratch.total_counts[0], scratch.total_masses[0]};
  // Every thread reads the totals before the next pass rewrites them.
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return measure;
}

// Narrows the bracket until it holds at most one token per thread or a
// single key.
template <class Order, class Target>
inline Bracket narrow(TargetRow row, Order order, OrderBoundary floor,
                      Bracket bracket, Target target,
                      threadgroup VocabularyScratch &scratch,
                      uint thread_index, uint lane, uint simd_group) {
  while (bracket.tokens() > kVocabularyThreads &&
         bracket.hi - bracket.lo > 1) {
    uint pivots[kPivots];
    place_pivots(order, bracket, row.temperature, pivots);
    measure_pivots<Target::by_mass>(row, order, floor, pivots, scratch,
                                    thread_index, lane, simd_group);
    // The highest pivot whose measure exceeds the target and the lowest one
    // whose measure does not bound the new bracket. A pivot at lo never
    // becomes hi: lo's measure may come from another sum (the shards' masses,
    // or the top-k selection's), and rounding must not empty the bracket.
    for (uint pivot = 0; pivot < kPivots; ++pivot) {
      const uint count = scratch.total_counts[pivot];
      const float mass = Target::by_mass ? scratch.total_masses[pivot] : 0.0f;
      if (target.exceeded(count, mass)) {
        if (pivots[pivot] >= bracket.lo) {
          bracket.lo = pivots[pivot];
          bracket.lo_count = count;
          if (Target::by_mass)
            bracket.lo_mass = mass;
        }
      } else if (pivots[pivot] > bracket.lo && pivots[pivot] < bracket.hi) {
        bracket.hi = pivots[pivot];
        bracket.hi_count = count;
        if (Target::by_mass)
          bracket.hi_mass = mass;
      }
    }
    // Every thread reads the totals before the next pass rewrites them.
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  return bracket;
}

// The last token the target keeps in a bracket of at most one token per
// thread: its tokens are gathered, put in order by rank, and added to the
// measure at hi one at a time until it exceeds the target. A count search in
// logit order measures no masses while it narrows: its gather sums the mass
// at hi as measure_pivots would have, or, when every token from lo up fits
// the group, gathers them all and adds them from the first, which leaves the
// tokens it keeps ranked in the scratch (Selection::ranked).
template <class Order, class Target>
inline Selection resolve(TargetRow row, Order order, OrderBoundary floor,
                         Bracket bracket, Target target,
                         threadgroup VocabularyScratch &scratch,
                         uint thread_index, uint lane, uint simd_group) {
  constexpr bool unmeasured = Order::by_logit && !Target::by_mass;
  const bool ranked = unmeasured && bracket.lo_count <= kVocabularyThreads;
  const bool measures = unmeasured && !ranked;
  if (thread_index == 0)
    atomic_store_explicit(&scratch.gathered, 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float above = 0.0f;
  for (uint token = thread_index; token < row.vocabulary;
       token += kVocabularyThreads) {
    const float value = row.logits[token];
    const uint key = order.key(value, token);
    if (key < bracket.lo || !row.admits(token) ||
        !at_or_before(logit_key(value), token, floor))
      continue;
    if (!ranked && key >= bracket.hi) {
      if (measures)
        above += row.weight(value);
      continue;
    }
    const uint slot = atomic_fetch_add_explicit(&scratch.gathered, 1u,
                                                memory_order_relaxed);
    if (slot < kVocabularyThreads) {
      scratch.keys[slot] = key;
      scratch.ids[slot] = token;
    }
  }
  if (measures) {
    above = simd_sum(above);
    if (lane == 0)
      scratch.masses[simd_group][0] = above;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint gathered = min(atomic_load_explicit(&scratch.gathered,
                                                 memory_order_relaxed),
                            kVocabularyThreads);
  uint key = 0;
  uint id = 0;
  uint rank = 0;
  if (thread_index < gathered) {
    key = scratch.keys[thread_index];
    id = scratch.ids[thread_index];
    for (uint other = 0; other < gathered; ++other) {
      const uint other_key = scratch.keys[other];
      rank += other_key > key || (other_key == key && scratch.ids[other] < id)
                  ? 1u
                  : 0u;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index < gathered) {
    scratch.keys[rank] = key;
    scratch.ids[rank] = id;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index == 0) {
    uint count = ranked ? 0 : bracket.hi_count;
    float mass = ranked ? 0.0f : bracket.hi_mass;
    if (measures) {
      mass = 0.0f;
      for (uint simd = 0; simd < kVocabularySimdgroups; ++simd)
        mass += scratch.masses[simd][0];
    }
    // Rounding may leave a mass target unreached: keep the whole bracket.
    Selection selection{{bracket.lo, 0xffffffffu},
                        bracket.lo_count,
                        Target::by_mass ? bracket.lo_mass : mass,
                        ranked};
    for (uint index = 0; index < gathered; ++index) {
      count += 1;
      mass += row.weight(order.logit(scratch.keys[index]));
      selection = {{scratch.keys[index], scratch.ids[index]}, count, mass,
                   ranked};
      if (target.exceeded(count, mass))
        break;
    }
    scratch.selection = selection;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return scratch.selection;
}

// The last token the target keeps among the admitted tokens at or before
// the floor, starting from a bracket in logit order.
template <class Target>
inline Selection select_last(TargetRow row, OrderBoundary floor,
                             Bracket bracket, Target target,
                             threadgroup VocabularyScratch &scratch,
                             uint thread_index, uint lane, uint simd_group) {
  bracket = narrow(row, LogitOrder{}, floor, bracket, target, scratch,
                   thread_index, lane, simd_group);
  if (bracket.tokens() <= kVocabularyThreads)
    return resolve(row, LogitOrder{}, floor, bracket, target, scratch,
                   thread_index, lane, simd_group);
  // More tokens tie at one logit than a group gathers: the target keeps its
  // first ones by id, each of the same weight, after the mass above them,
  // which a count search measures in one more pass.
  const uint ties = bracket.tokens();
  const float weight = row.weight(key_logit(bracket.lo));
  float above;
  uint keep;
  if constexpr (Target::by_mass) {
    above = bracket.hi_mass;
    keep = 1u + uint(clamp((target.mass - above) / weight, 0.0f,
                           float(ties - 1)));
  } else {
    above = measure_at_or_above(row, floor, bracket.hi, scratch, thread_index,
                                lane, simd_group)
                .mass;
    keep = target.count + 1 - bracket.hi_count;
  }
  uint last = bracket.lo == floor.key ? floor.last : 0xffffffffu;
  if (keep < ties) {
    const TieOrder order{bracket.lo, row.vocabulary};
    const CountTarget first{keep - 1};
    Bracket tied{1, row.vocabulary + 1, ties, 0, 0.0f, 0.0f};
    tied = narrow(row, order, floor, tied, first, scratch, thread_index, lane,
                  simd_group);
    last = resolve(row, order, floor, tied, first, scratch, thread_index, lane,
                   simd_group)
               .last.last;
  }
  return {{bracket.lo, last}, bracket.hi_count + keep,
          above + float(keep) * weight, false};
}

// The last token of the row's distribution: the tokens that weigh at least
// min_p of the heaviest, then the top_k of those, then within those the top_p
// nucleus, whose mass is measured against the mass of what the two cuts
// before it keep. A top-k selection the scratch holds ranked is walked for
// the nucleus in that order, with the sums it was selected with; otherwise
// the nucleus is searched for. A row that keeps every admitted token ends at
// {kNoKey, 0}.
inline OrderBoundary distribution_end(TargetRow row, float min_p, uint top_k,
                                      float top_p, float mass, uint admitted,
                                      threadgroup VocabularyScratch &scratch,
                                      uint thread_index, uint lane,
                                      uint simd_group) {
  OrderBoundary end{kNoKey, 0};
  Bracket bracket{kLowestKey, logit_key(row.maximum) + 1, admitted, 0, mass,
                  0.0f};
  if (min_p > 0.0f) {
    // A token weighs min_p of the heaviest, which weighs 1, at the logit
    // -temperature * log(min_p) below the maximum. The tokens at or above
    // that logit stay, the heaviest always.
    const uint lowest = logit_key(
        min(row.maximum + row.temperature * log(min_p), row.maximum));
    const Measure kept = measure_at_or_above(row, end, lowest, scratch,
                                             thread_index, lane, simd_group);
    if (kept.count < admitted) {
      end = {lowest, 0xffffffffu};
      bracket.lo = lowest;
      bracket.lo_count = kept.count;
      bracket.lo_mass = kept.mass;
    }
  }
  if (top_k < bracket.lo_count) {
    // A min_p cut ends on a whole key, which the bracket bounds from below,
    // so this search needs no floor.
    const Selection top = select_last(row, {kNoKey, 0}, bracket,
                                      CountTarget{top_k - 1}, scratch,
                                      thread_index, lane, simd_group);
    end = top.last;
    if (top.ranked && top_p < 1.0f) {
      // Every thread reads the selection before the walk rewrites it.
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (thread_index == 0) {
        // Rounding may leave the nucleus unreached: keep the whole top-k.
        const float nucleus = top_p * top.mass;
        float walked = 0.0f;
        for (uint index = 0; index < top.count; ++index) {
          walked += row.weight(key_logit(scratch.keys[index]));
          if (walked > nucleus) {
            end = {scratch.keys[index], scratch.ids[index]};
            break;
          }
        }
        scratch.selection.last = end;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      return scratch.selection.last;
    }
    bracket.lo = end.key;
    bracket.lo_count = top.count;
    bracket.lo_mass = top.mass;
  }
  if (top_p < 1.0f)
    end = select_last(row, end, bracket, MassTarget{top_p * bracket.lo_mass},
                      scratch, thread_index, lane, simd_group)
              .last;
  return end;
}

// A token's weight in the row's distribution: its softmax weight if the
// row admits it at or before the distribution's end, else 0.
inline float kept_weight(TargetRow row, OrderBoundary end, uint token) {
  if (token >= row.vocabulary || !row.admits(token))
    return 0.0f;
  const float value = row.logits[token];
  return at_or_before(logit_key(value), token, end) ? row.weight(value) : 0.0f;
}

// The draft's candidates in one range of the draw, one bit per staged entry.
inline uint range_candidates(uint range, threadgroup DrawScratch &scratch) {
  uint held = 0;
  for (uint index = 0; index < kDraftCandidates; ++index)
    held |= scratch.draft_ranges[index] == range ? 1u << index : 0u;
  return held;
}

// One range's weight in the draw: the kept weight of its tokens other than
// the draft's candidates, plus its candidates' draw weights.
inline float range_weight(uint range, threadgroup DrawScratch &scratch) {
  float weight = scratch.range_rest[range];
  for (uint held = range_candidates(range, scratch); held; held &= held - 1)
    weight += scratch.draft_draw_weights[ctz(held)];
  return weight;
}

// The last token of a range with weight in the draw, which the draw takes
// when rounding leaves its target unreached.
inline uint range_last(uint range, threadgroup DrawScratch &scratch) {
  uint after = scratch.range_last[range];
  for (uint held = range_candidates(range, scratch); held; held &= held - 1) {
    const uint index = ctz(held);
    if (scratch.draft_draw_weights[index] > 0.0f)
      after = max(after, scratch.draft_ids[index] + 1);
  }
  return after - 1;
}

// Draws the row's token, shared by its groups; returns true in the group that
// finishes last, with the drawn token and the draft token's probability. Each
// kept token weighs its softmax weight; with a draft, each of the draft's
// candidates weighs its weight less its draft probability times the kept
// mass, never below zero, so the draw follows the residual distribution
// acceptance corrects a rejected draft token from (the whole distribution
// when nothing remains). Each simdgroup of the row's groups sums one range of
// the vocabulary without the candidates; the last group computes their
// weights once, adds the ranges in order and walks the one that holds the
// draw with prefix sums. A range's weight and its walk add the same token
// weights, so the range the draw picks always has a token to draw.
inline bool vocabulary_draw(
    TargetRow row, OrderBoundary end, bool drafted, uint draft_token,
    device const uint *draft_ids, device const float *draft_probabilities,
    float uniform, device coherent(device) TargetVocabularyRange *ranges,
    device atomic_uint *arrivals, uint slice, threadgroup DrawScratch &scratch,
    uint thread_index, uint lane, uint simd_group, thread uint &drawn,
    thread float &draft_probability) {
  const uint range_tokens = (row.vocabulary + kVocabularyRanges * 32 - 1) /
                            (kVocabularyRanges * 32) * 32;
  if (thread_index < kDraftCandidates) {
    // A candidate the draft lists twice counts once, as lookups find its
    // first entry.
    uint id = drafted ? draft_ids[thread_index] : 0xffffffffu;
    for (uint earlier = 0; drafted && earlier < thread_index; ++earlier)
      id = draft_ids[earlier] == id ? 0xffffffffu : id;
    scratch.draft_ids[thread_index] = id;
    scratch.draft_ranges[thread_index] =
        id < row.vocabulary ? id / range_tokens : kVocabularyRanges;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint own = slice * kVocabularySimdgroups + simd_group;
  const uint begin = min(own * range_tokens, row.vocabulary);
  const uint finish = min(begin + range_tokens, row.vocabulary);
  const uint held = range_candidates(own, scratch);
  float rest = 0.0f;
  uint after = 0;
  for (uint first = begin; first < finish; first += 32) {
    const uint token = first + lane;
    float weight = token < finish ? kept_weight(row, end, token) : 0.0f;
    for (uint bits = held; bits; bits &= bits - 1)
      weight = scratch.draft_ids[ctz(bits)] == token ? 0.0f : weight;
    rest += weight;
    after = weight > 0.0f ? token + 1 : after;
  }
  rest = simd_sum(rest);
  after = simd_max(after);
  if (lane == 0)
    ranges[own] = {rest, after};
  if (!split_arrive_last(arrivals, kVocabularyGroups, thread_index,
                         &scratch.arrival))
    return false;
  if (thread_index < kVocabularyRanges) {
    const TargetVocabularyRange measured = ranges[thread_index];
    scratch.range_rest[thread_index] = measured.rest;
    scratch.range_last[thread_index] = measured.after;
  }
  if (thread_index < kDraftCandidates) {
    scratch.draft_probabilities[thread_index] =
        drafted ? draft_probabilities[thread_index] : 0.0f;
    scratch.draft_weights[thread_index] =
        kept_weight(row, end, scratch.draft_ids[thread_index]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index == 0) {
    float kept_mass = 0.0f;
    for (uint range = 0; range < kVocabularyRanges; ++range)
      kept_mass += scratch.range_rest[range];
    for (uint index = 0; index < kDraftCandidates; ++index)
      kept_mass += scratch.draft_weights[index];
    bool residual = false;
    for (uint range = 0; drafted && range < kVocabularyRanges; ++range)
      residual = residual || scratch.range_rest[range] > 0.0f;
    for (uint index = 0; drafted && index < kDraftCandidates; ++index) {
      const float left = max(scratch.draft_weights[index] -
                                 scratch.draft_probabilities[index] * kept_mass,
                             0.0f);
      scratch.draft_draw_weights[index] = left;
      residual = residual || left > 0.0f;
    }
    // Without a draft, or with nothing left of its residual, the draw
    // follows the distribution itself.
    for (uint index = 0; !residual && index < kDraftCandidates; ++index)
      scratch.draft_draw_weights[index] = scratch.draft_weights[index];
    scratch.kept_mass = kept_mass;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index < kVocabularyRanges)
    scratch.range_weights[thread_index] = range_weight(thread_index, scratch);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index == 0) {
    float total = 0.0f;
    for (uint range = 0; range < kVocabularyRanges; ++range)
      total += scratch.range_weights[range];
    const float target = uniform * total;
    float cumulative = 0.0f;
    uint range_drawn = 0;
    float range_target = 0.0f;
    for (uint range = 0; range < kVocabularyRanges; ++range) {
      const float weight = scratch.range_weights[range];
      if (!(weight > 0.0f))
        continue;
      // Rounding may leave the target unreached: the last range with weight.
      range_drawn = range;
      range_target = target - cumulative;
      cumulative += weight;
      if (cumulative > target)
        break;
    }
    scratch.drawn_range = range_drawn;
    scratch.drawn_target = range_target;
    scratch.drawn_token = range_last(range_drawn, scratch);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0) {
    const uint walked = scratch.drawn_range;
    const uint walk_begin = min(walked * range_tokens, row.vocabulary);
    const uint walk_finish = min(walk_begin + range_tokens, row.vocabulary);
    const uint walk_held = range_candidates(walked, scratch);
    const float target = scratch.drawn_target;
    float cumulative = 0.0f;
    for (uint first = walk_begin; first < walk_finish; first += 32) {
      const uint token = first + lane;
      float weight = token < walk_finish ? kept_weight(row, end, token) : 0.0f;
      for (uint bits = walk_held; bits; bits &= bits - 1) {
        const uint index = ctz(bits);
        if (scratch.draft_ids[index] == token)
          weight = scratch.draft_draw_weights[index];
      }
      const float prefix = simd_prefix_inclusive_sum(weight);
      const uint hit =
          simd_min(weight > 0.0f && cumulative + prefix > target ? lane : 32u);
      if (hit < 32) {
        if (lane == 0)
          scratch.drawn_token = first + hit;
        break;
      }
      cumulative += simd_broadcast(prefix, 31);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  drawn = scratch.drawn_token;
  draft_probability =
      drafted ? kept_weight(row, end, draft_token) / scratch.kept_mass : 0.0f;
  split_release(arrivals, thread_index);
  return true;
}

// Where a sampled row's distribution ends: its shards' masses, merged in
// shard order into the row's largest admitted logit, softmax denominator and
// admitted count, then the search. The record keeps the maximum and the
// end, which the draw reads.
inline void search_row(TargetRow row, device const TargetShardMass *masses,
                       float min_p, uint top_k, float top_p,
                       device TargetVocabularyRow &record,
                       threadgroup VocabularyScratch &scratch,
                       uint thread_index, uint lane, uint simd_group) {
  const TargetShardMass merged =
      merge_masses(masses, RICHENGINE_TARGET_SAMPLING_SHARDS, row.temperature);
  row.maximum = merged.maximum;
  const OrderBoundary end =
      distribution_end(row, min_p, top_k, top_p, merged.sum, merged.admitted,
                       scratch, thread_index, lane, simd_group);
  if (thread_index == 0)
    record = {merged.maximum, end.key, end.last, 0.0f};
}

// Where the distribution of each selected row of the sampled lanes ends, one
// group per row.
kernel void decode_sample_vocabulary_search(
    device const float *logits [[buffer(0)]],
    device const uint *token_mask [[buffer(1)]],
    device const TargetShardMass *partial_masses [[buffer(2)]],
    device TargetVocabularyRow *vocabulary_rows [[buffer(3)]],
    constant TargetSamplingParams &params [[buffer(4)]],
    uint s [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup VocabularyScratch scratch;
  if (!lane_samples(params, s) || row_dead(params, s))
    return;
  const uint batch = selected_lane(params, s);
  search_row(selected_row(logits, token_mask, params, s),
             partial_masses + ulong(s) * RICHENGINE_TARGET_SAMPLING_SHARDS,
             params.min_p[batch], params.top_k[batch], params.top_p[batch],
             vocabulary_rows[s], scratch, thread_index, lane, simd_group);
}

// The draw of each selected row of the sampled lanes, kVocabularyGroups
// groups per row: the first token after a prompt, a verify row's draft
// token's probability and the correction a rejection takes or, for the last
// verify row, which follows the whole draft, its bonus token.
kernel void decode_sample_vocabulary_draw(
    device const float *logits [[buffer(0)]],
    device const uint *token_mask [[buffer(1)]],
    device TargetVocabularyRow *vocabulary_rows [[buffer(2)]],
    device const uint *input_tokens [[buffer(3)]],
    device const uint *draft_ids [[buffer(4)]],
    device const float *draft_probabilities [[buffer(5)]],
    device const float *uniforms [[buffer(6)]],
    device uint *tokens [[buffer(7)]],
    device coherent(device) TargetVocabularyRange *ranges [[buffer(8)]],
    device atomic_uint *arrivals [[buffer(9)]],
    constant TargetSamplingParams &params [[buffer(10)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup DrawScratch scratch;
  const uint s = group / kVocabularyGroups;
  if (!lane_samples(params, s) || row_dead(params, s))
    return;
  const uint batch = selected_lane(params, s);
  const uint index = s % params.rows;
  device TargetVocabularyRow &record = vocabulary_rows[s];
  TargetRow row = selected_row(logits, token_mask, params, s);
  row.maximum = record.maximum;
  // A drafted row follows its draft token, the next verify input row. The
  // lane's last live row is never drafted: its draw is the bonus a full
  // acceptance takes, sampled from the target, not the residual. A zero
  // live_rows is an unadapted dispatch — every row is live.
  const uint live = params.live_rows[batch];
  const bool drafted =
      index < params.drafted_rows && (!live || index + 1 < live);
  const ulong position =
      ulong(batch) * RICHENGINE_DRAFT_PROPOSAL_TOKENS + (drafted ? index : 0);
  uint token;
  float draft_probability;
  if (vocabulary_draw(
          row, {record.end_key, record.end_last}, drafted,
          drafted ? input_tokens[s + 1] : 0u,
          draft_ids + position * kDraftCandidates,
          draft_probabilities + position * kDraftCandidates,
          uniforms[ulong(batch) * RICHENGINE_SAMPLING_UNIFORMS + params.uniform],
          ranges + ulong(s) * kVocabularyRanges, arrivals + s,
          group % kVocabularyGroups, scratch, thread_index, lane, simd_group,
          token, draft_probability) &&
      thread_index == 0) {
    tokens[s] = token;
    if (drafted)
      record.draft_probability = draft_probability;
  }
}

// The sampling penalties of one logit: repetition divides a positive
// logit and multiplies a negative one when the prompt or the output holds
// the token, and presence and frequency lower it by the output's count of
// the token. The result saturates, so no rewritten logit is infinite.
inline float penalize_logit(float value, uint count, float repetition,
                            float repetition_inverse, float presence,
                            float frequency) {
  value *= value > 0.0f ? repetition_inverse : repetition;
  if (count)
    value -= frequency * float(count) + presence;
  return clamp(value, -FLT_MAX, FLT_MAX);
}

// Rewrites one token's logit in each penalized row of one entry, in place.
// Bit r of drafted is set when the lane's verify input row r holds the
// token: those are the draft tokens that verify row r's context adds to the
// output, so row r counts bits 1..r on top of the table's count.
inline void penalize_token(device float *logits, device const uint *words,
                           uint token, uint entry, uint drafted,
                           constant SamplingPenaltyParams &params) {
  const uint word =
      words[ulong(params.table_row[entry]) * params.vocabulary + token];
  if (!word && !drafted)
    return;
  device float *column =
      logits +
      (ulong(params.logits_lane[entry]) * RICHENGINE_TARGET_VERIFY_ROWS +
       params.row_offset) * params.vocabulary +
      token;
  for (uint row = 0; row < params.rows; ++row) {
    const uint count = (word & RICHENGINE_PENALTY_COUNT_MASK) +
                       popcount(drafted & ((2u << row) - 2u));
    if (!count && !(word & RICHENGINE_PENALTY_PROMPT_BIT))
      continue;
    device float &logit = column[ulong(row) * params.vocabulary];
    logit = penalize_logit(logit, count, params.repetition[entry],
                           params.repetition_inverse[entry],
                           params.presence[entry], params.frequency[entry]);
  }
}

// One thread per vocabulary token and penalized entry, over the rows the
// first token after a prompt is selected from.
kernel void decode_sample_penalize(device float *logits [[buffer(0)]],
                                   device const uint *words [[buffer(1)]],
                                   constant SamplingPenaltyParams &params
                                   [[buffer(2)]],
                                   uint2 position [[thread_position_in_grid]]) {
  if (position.x < params.vocabulary && position.y < params.entries)
    penalize_token(logits, words, position.x, position.y, 0, params);
}

// The verify rows of each penalized lane, whose contexts add the lane's
// draft tokens (verify input rows 1..7) one row at a time.
kernel void decode_sample_penalize_verify(
    device float *logits [[buffer(0)]], device const uint *words [[buffer(1)]],
    device const uint *input_tokens [[buffer(2)]],
    constant SamplingPenaltyParams &params [[buffer(3)]],
    uint2 position [[thread_position_in_grid]]) {
  const uint token = position.x;
  const uint entry = position.y;
  if (token >= params.vocabulary || entry >= params.entries)
    return;
  device const uint *inputs =
      input_tokens + ulong(params.logits_lane[entry]) * RICHENGINE_TARGET_VERIFY_ROWS;
  uint drafted = 0;
  for (uint row = 1; row < RICHENGINE_TARGET_VERIFY_ROWS; ++row)
    drafted |= uint(inputs[row] == token) << row;
  penalize_token(logits, words, token, entry, drafted, params);
}

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

kernel void draft_select_top16_sharded(
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
  draft_top16_sharded_phase(logits, partial_ids, partial_values, vocabulary, 1,
                            maxima, thresholds, &round_values[0][0],
                            &round_ids[0][0], group, thread_index, lane,
                            simd_group);
}

inline float sparse_lookup(device const uint *ids,
                           device const float *probabilities, uint count,
                           uint token) {
  for (uint i = 0; i < count; ++i) {
    if (ids[i] == token)
      return probabilities[i];
  }
  return 0.0f;
}

// One lane's view of AcceptBatchParams, built by decode_accept_dflash.
struct AcceptParams {
  uint remaining;
  uint stop_token_0;
  uint stop_token_1;
  // The lane's proposal budget: the most draft tokens it may accept.
  uint limit;
  // Draft ids/probabilities stride per position (candidate-table width).
  uint candidate_stride;
};

// Keeps at most params.remaining of the accepted tokens plus the correction,
// cut after the first stop token, and records the retained and accepted
// counts.
inline void finish_acceptance(device const uint *tokens, uint accepted,
                              AcceptParams params, device uint &retained,
                              device uint &accepted_count) {
  accepted_count = accepted;
  retained = min(accepted + 1, params.remaining);
  for (uint i = 0; i < retained; ++i) {
    if (tokens[i] == params.stop_token_0 || tokens[i] == params.stop_token_1) {
      retained = i + 1;
      break;
    }
  }
}

// A sampled lane's verify rows carry their draft tokens' target
// probabilities (TargetVocabularyRow), and its output tokens their draws:
// the correction a rejection takes or, for the last row, the bonus token.
// Acceptance overwrites the accepted rows with the draft tokens, so row
// accepted keeps its draw.
inline void accept_sampled_lane(device const uint *draft_tokens,
                                device const uint *draft_ids,
                                device const float *draft_probs,
                                device const TargetVocabularyRow *target_rows,
                                device const float *uniforms,
                                device uint *output_tokens,
                                device uint &retained,
                                device uint &accepted_count,
                                AcceptParams params) {
  uint accepted = 0;
  while (accepted < params.limit) {
    uint token = draft_tokens[accepted];
    float q = sparse_lookup(draft_ids + accepted * params.candidate_stride,
                            draft_probs + accepted * params.candidate_stride,
                            params.candidate_stride, token);
    float p = target_rows[accepted].draft_probability;
    if (!(uniforms[RICHENGINE_UNIFORM_ACCEPTANCE + accepted] * q < p))
      break;
    output_tokens[accepted] = token;
    ++accepted;
  }
  finish_acceptance(output_tokens, accepted, params, retained,
                    accepted_count);
}

// One shard of a greedy row: its admitted token with the largest logit, the
// lowest id among ties, per thread in token order and then over the group.
inline void argmax_shard(TargetRow row, uint shard, device float &partial_value,
                         device uint &partial_index,
                         threadgroup float *group_values,
                         threadgroup uint *group_indices, uint thread_index) {
  constexpr uint Shards = RICHENGINE_TARGET_SAMPLING_SHARDS;
  float best = -INFINITY;
  uint best_index = 0xffffffffu;
  for (uint token = shard * 256 + thread_index; token < row.vocabulary;
       token += Shards * 256) {
    if (!row.admits(token))
      continue;
    float value = row.logits[token];
    if (value > best || (value == best && token < best_index)) {
      best = value;
      best_index = token;
    }
  }
  uint lane = thread_index & 31;
  uint simd_group = thread_index >> 5;
  float simd_best = simd_max(best);
  uint simd_index = simd_min(best == simd_best ? best_index : 0xffffffffu);
  if (lane == 0) {
    group_values[simd_group] = simd_best;
    group_indices[simd_group] = simd_index;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0) {
    float value = lane < 8 ? group_values[lane] : -INFINITY;
    float group_best = simd_max(value);
    uint index =
        lane < 8 && value == group_best ? group_indices[lane] : 0xffffffffu;
    index = simd_min(index);
    if (lane == 0) {
      partial_value = group_best;
      partial_index = index;
    }
  }
}

// One shard of each selected row of the greedy lanes.
kernel void decode_sample_argmax_sharded(
    device const float *logits [[buffer(0)]],
    device const uint *token_mask [[buffer(1)]],
    device float *partial_values [[buffer(2)]],
    device uint *partial_indices [[buffer(3)]],
    constant TargetSamplingParams &params [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  threadgroup float group_values[8];
  threadgroup uint group_indices[8];
  const uint s = group / RICHENGINE_TARGET_SAMPLING_SHARDS;
  if (lane_samples(params, s) || row_dead(params, s))
    return;
  argmax_shard(selected_row(logits, token_mask, params, s),
               group % RICHENGINE_TARGET_SAMPLING_SHARDS, partial_values[group],
               partial_indices[group], group_values, group_indices,
               thread_index);
}

// A greedy row's token from its shards' partials.
inline uint argmax_reduce(device const float *partial_values,
                          device const uint *partial_indices, uint row,
                          uint lane) {
  constexpr uint Shards = RICHENGINE_TARGET_SAMPLING_SHARDS;
  float value = lane < Shards ? partial_values[row * Shards + lane] : -INFINITY;
  float best = simd_max(value);
  uint index = lane < Shards && value == best
                   ? partial_indices[row * Shards + lane]
                   : 0xffffffffu;
  return simd_min(index);
}

// The token of each selected row of the greedy lanes.
kernel void decode_sample_argmax_reduce(
    device const float *partial_values [[buffer(0)]],
    device const uint *partial_indices [[buffer(1)]],
    device uint *tokens [[buffer(2)]],
    constant TargetSamplingParams &params [[buffer(3)]],
    uint s [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  if (lane_samples(params, s) || row_dead(params, s))
    return;
  const uint token = argmax_reduce(partial_values, partial_indices, s, lane);
  if (lane == 0)
    tokens[s] = token;
}

inline void accept_greedy_lane(device const uint *draft_tokens,
                               device uint *target_tokens,
                               device uint &retained,
                               device uint &accepted_count,
                               AcceptParams params) {
  uint accepted = 0;
  while (accepted < params.limit &&
         draft_tokens[accepted] == target_tokens[accepted]) {
    ++accepted;
  }
  finish_acceptance(target_tokens, accepted, params, retained,
                    accepted_count);
}

kernel void decode_accept_dflash(
    device const uint *draft_tokens [[buffer(0)]],
    device const uint *draft_ids [[buffer(1)]],
    device const float *draft_probs [[buffer(2)]],
    device const TargetVocabularyRow *target_rows [[buffer(3)]],
    device const float *uniforms [[buffer(4)]],
    device uint *target_tokens [[buffer(5)]],
    device uint *retained [[buffer(6)]],
    device uint *accepted_count [[buffer(7)]],
    constant AcceptBatchParams &params [[buffer(8)]],
    uint batch [[threadgroup_position_in_grid]]) {
  uint remaining = params.remaining[batch];
  AcceptParams lane_params{remaining, params.stop_token_0,
                           params.stop_token_1,
                           params.proposals[batch]
                               ? min(uint(RICHENGINE_DRAFT_PROPOSAL_TOKENS),
                                     params.proposals[batch])
                               : uint(RICHENGINE_DRAFT_PROPOSAL_TOKENS),
                           params.candidate_stride};
  device const uint *lane_draft =
      draft_tokens + batch * RICHENGINE_DRAFT_PROPOSAL_TOKENS;
  device uint *lane_target =
      target_tokens + batch * RICHENGINE_TARGET_VERIFY_ROWS;
  if (params.sampling_mask & (1u << batch)) {
    accept_sampled_lane(
        lane_draft,
        draft_ids + batch * RICHENGINE_DRAFT_PROPOSAL_TOKENS *
                      params.candidate_stride,
        draft_probs + batch * RICHENGINE_DRAFT_PROPOSAL_TOKENS *
                          params.candidate_stride,
        target_rows + batch * RICHENGINE_TARGET_VERIFY_ROWS,
        uniforms + batch * RICHENGINE_SAMPLING_UNIFORMS, lane_target,
        retained[batch], accepted_count[batch], lane_params);
  } else {
    accept_greedy_lane(lane_draft, lane_target, retained[batch],
                       accepted_count[batch], lane_params);
  }
}

// Greedy acceptance over one lane's verify tree. From the anchor the walk
// descends into the child whose token equals the node's argmax — the chain
// successor or the position's sibling leaf — and emits the walk's tokens in
// path order: the accepted nodes' tokens, then the last node's argmax as the
// correction or bonus, capped at remaining and cut after a stop token.
// retained_path records the committed path's DFS rows for the KV, GDN and
// captured-hidden commits.
kernel void decode_accept_tree(
    device const uint *tree_tokens [[buffer(0)]],
    device const uint *tree_nodes [[buffer(1)]],
    device const uint *tree_counts [[buffer(2)]],
    device const uint *target_tokens [[buffer(3)]],
    device uint *output_tokens [[buffer(4)]],
    device uint *retained [[buffer(5)]],
    device uint *accepted_count [[buffer(6)]],
    device uint *retained_path [[buffer(7)]],
    constant TreeAcceptBatchParams &params [[buffer(8)]],
    uint batch [[threadgroup_position_in_grid]]) {
  constexpr uint Nodes = RICHENGINE_TREE_VERIFY_NODES;
  constexpr uint Emitted = RICHENGINE_TARGET_VERIFY_ROWS;
  tree_tokens += batch * Nodes;
  tree_nodes += batch * Nodes;
  target_tokens += batch * Nodes;
  output_tokens += batch * Emitted;
  retained_path += batch * Emitted;
  const uint node_count = tree_counts[batch];
  const uint remaining = params.remaining[batch];

  uint path[Emitted];
  path[0] = 0;
  uint count = 1;
  uint cursor = 0;
  // The comb's chain nodes lead the node block: a chain cursor's children are
  // the chain successor at cursor + 1 while one exists, and the sibling leaf
  // at row Nodes/2 + cursor. A leaf ends the walk — leaf rows have no
  // children — and the path stops at Emitted rows, the committed block's
  // capacity.
  constexpr uint ChainRows = Nodes / 2;
  while (count < Emitted && count <= remaining && cursor < ChainRows) {
    const uint selected = target_tokens[cursor];
    uint next = Nodes;
    if (cursor + 1 < ChainRows && tree_tokens[cursor + 1] == selected) {
      next = cursor + 1;
    } else {
      const uint leaf = ChainRows + cursor;
      if (leaf < node_count &&
          RICHENGINE_TREE_NODE_PARENT(tree_nodes[leaf]) == cursor &&
          tree_tokens[leaf] == selected) {
        next = leaf;
      }
    }
    if (next == Nodes)
      break;
    path[count] = next;
    output_tokens[count - 1] = tree_tokens[next];
    ++count;
    cursor = next;
  }
  if (count <= remaining)
    output_tokens[count - 1] = target_tokens[cursor];
  AcceptParams lane_params{remaining, params.stop_token_0,
                           params.stop_token_1,
                           RICHENGINE_DRAFT_PROPOSAL_TOKENS,
                           kDraftCandidates};
  finish_acceptance(output_tokens, count - 1, lane_params, retained[batch],
                    accepted_count[batch]);
  for (uint i = 0; i < retained[batch]; ++i)
    retained_path[i] = path[i];
}

// Splices ANE-produced alternates into the comb's sibling-leaf rows (8..14),
// one per proposal slot, when the predictor job finished before this dispatch
// executes. The flag carries the job serial the encode snapshot named
// `expected`; the job writes the token block, then the serial with release
// order, so an acquired match also publishes the tokens. A NONE token keeps
// the draft's own second-best leaf, and rows past the tree's node count are
// not nodes at all.
kernel void tree_leaf_patch(
    device uint *tree_tokens [[buffer(0)]],
    device const uint *medusa_tokens [[buffer(1)]],
    device atomic_uint *medusa_flag [[buffer(2)]],
    device const uint *tree_counts [[buffer(3)]],
    constant TreeLeafPatchParams &params [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
  if (atomic_load_explicit(medusa_flag, memory_order_acquire,
                           mem_flags::mem_device) != params.expected)
    return;
  constexpr uint Leaves = RICHENGINE_DRAFT_PROPOSAL_TOKENS;
  const uint lane = index / Leaves;
  const uint slot = index % Leaves;
  if (lane >= params.lanes)
    return;
  const uint row = RICHENGINE_TREE_VERIFY_NODES / 2 + slot;
  if (row >= tree_counts[lane])
    return;
  const uint token = medusa_tokens[index];
  if (token != 0xffffffffu)
    tree_tokens[lane * RICHENGINE_TREE_VERIFY_NODES + row] = token;
}

// ---------------------------------------------------------------------------
// Fused vocabulary head + argmax (the Ornith-9B head shape): each
// threadgroup computes one 32 x 128 logits tile with the q4_mpp tile and
// holds it in threadgroup memory; threads 0-31 scan one row's 128 logits
// once for the maximum and its index — lowest id on ties, matching
// argmax_shard — writing per-(selected row, column tile) partials that
// decode_head_argmax_reduce_tiles reduces. Runs only on all-greedy
// unconstrained chain-verify steps, so the logits buffer is never written
// or read.
constant constexpr ushort kHeadArgmaxRows = 32;
constant constexpr ushort kHeadArgmaxCols = 128;

kernel void decode_head_argmax_q4(
    device bfloat *input [[buffer(0)]], device uchar *weights [[buffer(1)]],
    device bfloat *scales [[buffer(2)]], device bfloat *biases [[buffer(3)]],
    device float *partial_values [[buffer(4)]],
    device uint *partial_indices [[buffer(5)]],
    constant HeadArgmaxParams &params [[buffer(6)]],
    uint tile [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  constexpr ushort M = kHeadArgmaxRows, TN = kHeadArgmaxCols;
  threadgroup float tg_tile[M * TN];
  threadgroup float input_sums[8 * M];
  const uint col0 = tile * TN;
  q4_mpp_tile_sums<M, TN, false, 8, false>(
      input, weights, scales, biases, weights, scales, biases,
      params.input_size, input_sums, col0, 0, params.input_size / 64,
      simd_lane, simd_group,
      [&](thread auto &sums_0,
          thread auto &sums_1 [[maybe_unused]],
          Q4Traversal traversal) __attribute__((always_inline)) {
        q4_visit(sums_0, traversal, [&](ushort i)
                 __attribute__((always_inline)) {
          auto index = sums_0.get_multidimensional_index(i);
          tg_tile[index[1] * TN + index[0]] = sums_0[i];
        });
      });
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index >= M) return;
  const uint s = thread_index;
  const uint tiles = params.output_size / TN;
  float best = -INFINITY;
  uint best_index = 0xffffffffu;
  const uint live = params.live_rows[s / params.rows];
  if (!live || s % params.rows < live) {
    const uint lane = s / params.rows;
    const bool exclude_stop =
        (params.exclude_stop_mask & (1u << lane)) != 0;
    for (uint col = 0; col < TN; ++col) {
      const uint token = col0 + col;
      if (exclude_stop &&
          (token == params.stop_token_0 || token == params.stop_token_1))
        continue;
      const float value = tg_tile[s * TN + col];
      if (value > best || (value == best && token < best_index)) {
        best = value;
        best_index = token;
      }
    }
  }
  partial_values[s * tiles + tile] = best;
  partial_indices[s * tiles + tile] = best_index;
}

// Reduces the fused head's per-(row, column tile) partials to each selected
// greedy row's token, lowest id on ties.
kernel void decode_head_argmax_reduce_tiles(
    device const float *partial_values [[buffer(0)]],
    device const uint *partial_indices [[buffer(1)]],
    device uint *tokens [[buffer(2)]],
    constant TargetSamplingParams &params [[buffer(3)]],
    uint s [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  if (lane_samples(params, s) || row_dead(params, s))
    return;
  const uint tiles = params.vocabulary / kHeadArgmaxCols;
  float best = -INFINITY;
  uint best_index = 0xffffffffu;
  for (uint t = lane; t < tiles; t += 32) {
    const float value = partial_values[s * tiles + t];
    const uint index = partial_indices[s * tiles + t];
    if (value > best || (value == best && index < best_index)) {
      best = value;
      best_index = index;
    }
  }
  const float group_best = simd_max(best);
  const uint group_index =
      simd_min(best == group_best ? best_index : 0xffffffffu);
  if (lane == 0)
    tokens[s] = group_index;
}

// The GGUF fused head's partial tiles are GGUF_TILE_COLUMNS wide.
kernel void decode_head_argmax_reduce_tiles_gguf(
    device const float *partial_values [[buffer(0)]],
    device const uint *partial_indices [[buffer(1)]],
    device uint *tokens [[buffer(2)]],
    constant TargetSamplingParams &params [[buffer(3)]],
    uint s [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  if (lane_samples(params, s) || row_dead(params, s))
    return;
  const uint tiles = params.vocabulary / GGUF_TILE_COLUMNS;
  float best = -INFINITY;
  uint best_index = 0xffffffffu;
  for (uint t = lane; t < tiles; t += 32) {
    const float value = partial_values[s * tiles + t];
    const uint index = partial_indices[s * tiles + t];
    if (value > best || (value == best && index < best_index)) {
      best = value;
      best_index = index;
    }
  }
  const float group_best = simd_max(best);
  const uint group_index =
      simd_min(best == group_best ? best_index : 0xffffffffu);
  if (lane == 0)
    tokens[s] = group_index;
}

// Gemma 4's final logit softcap: logits <- cap * tanh(logits / cap), in
// place, before argmax or sampling (cap 30). The map is strictly monotonic,
// so the fused decode_head_argmax path needs no variant — its argmax is the
// softcapped logits' argmax; only sampled rows and any path that reads the
// logit values run this.
kernel void decode_logit_softcap(
    device float *logits [[buffer(0)]],
    constant float &cap [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint index [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]]) {
  for (uint element = index; element < count; element += grid_size)
    logits[element] = cap * richengine_tanh(logits[element] / cap);
}
