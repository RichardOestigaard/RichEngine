#include "metal/abi/Embedding.h"
#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/activation.h"
#include "metal/kernels/common/rms_inverse.h"

// DiffusionGemma canvas-mode decode-step kernels. One denoising iteration:
//
//   canvas_logits_scale          (optional) logits *= 1/temperature
//   decode_logit_softcap         (existing) cap * tanh(logits/cap), cap 30
//   canvas_row_stats             per-position softmax stats: multinomial
//                                sample, argmax, entropy
//   canvas_entropy_accept        entropy-sorted accept mask -> new canvas
//   canvas_soft_embed_*          soft embeddings from the step's logits
//   canvas_self_condition        fused residual-add + scaleless RMS norm
//   canvas_uniform_noise         canvas init / renoise token fill
//
// Canvas geometry: the canvas holds at most 256 positions; the live row
// count arrives as a runtime parameter (a shorter canvas skips the dead
// tail of a request's remaining token budget). vocabulary = 262144,
// hidden = 2816 stay compile-time.

constant uint CanvasRows = 256;

// ---------------------------------------------------------------------------
// Counter-based per-position RNG (no state): a splitmix-style 32-bit hash of
// (seed, counter). One draw per (kernel, step, position) keeps every step's
// stream reproducible without a device RNG state buffer.
inline uint richengine_canvas_hash(uint seed, uint counter) {
  uint h = seed + counter * 0x9E3779B9u;
  h ^= h >> 16;
  h *= 0x7FEB352Du;
  h ^= h >> 15;
  h *= 0x846CA68Bu;
  h ^= h >> 16;
  return h;
}

// A uniform in [0, 1) from the hash.
inline float richengine_canvas_uniform(uint seed, uint counter) {
  return float(richengine_canvas_hash(seed, counter)) *
         (1.0f / 4294967296.0f);
}

// Fill `tokens` with Uniform(0, vocabulary) draws over `count` positions,
// one lane per position. Used for the initial canvas and to renoise
// rejected positions; `seed` distinguishes steps.
struct CanvasNoiseParams {
  uint32_t vocabulary;
  uint32_t seed;
  uint32_t count;
};

kernel void canvas_uniform_noise(
    device uint *tokens [[buffer(0)]],
    constant CanvasNoiseParams &params [[buffer(1)]],
    uint index [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]]) {
  for (uint i = index; i < params.count; i += grid_size)
    tokens[i] = richengine_canvas_hash(params.seed, i) % params.vocabulary;
}

// logits[i] *= scale (bind 1/temperature), in place over rows*vocabulary
// elements. Temperature must be applied before canvas_row_stats — its
// entropy, argmax (argmax is scale-invariant) and sample all read the
// scaled logits. decode_logit_softcap may run before or after this; the
// ops commute only approximately, so follow upstream: softcap the raw
// logits first, then scale.
// logits[i] = cap * tanh(logits[i] / cap), in place — the canvas variant
// of decode_logit_softcap for the bf16 canvas logits.
kernel void canvas_logit_softcap(
    device bfloat *logits [[buffer(0)]],
    constant float &cap [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint index [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]]) {
  for (uint element = index; element < count; element += grid_size)
    logits[element] = bfloat(cap * richengine_tanh(float(logits[element]) / cap));
}

kernel void canvas_logits_scale(
    device bfloat *logits [[buffer(0)]],
    constant float &scale [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint index [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]]) {
  for (uint i = index; i < count; i += grid_size)
    logits[i] = bfloat(float(logits[i]) * scale);
}

// Per-canvas-row statistics over a [CanvasRows, vocabulary] fp32 logits row:
// the softmax normalizer, a multinomial sample of softmax(logits), the row's
// argmax and its entropy H = log(Z) + max - sum(exp(l - max) * l) / Z.
//
// One threadgroup of 256 per canvas row (grid = CanvasRows). Each thread
// owns a contiguous slice of vocabulary/256 logits; the draw walks a
// threadgroup-ordered prefix sum of the threads' slice masses, and the
// owning thread rescans its slice — sampling order is deterministic given
// the seed.
//
// Buffers:
//   0 logits   fp32 [rows][vocabulary]
//   1 sampled  uint [rows]  — multinomial draw of softmax(logits)
//   2 argmax   uint [rows]  — row argmax (lowest index on ties)
//   3 entropy  float [rows]
//   4 params   CanvasRowStatsParams {vocabulary, seed}
struct CanvasRowStatsParams {
  uint32_t vocabulary;
  uint32_t seed;
};

kernel void canvas_row_stats(
    device const bfloat *logits [[buffer(0)]],
    device uint *sampled [[buffer(1)]],
    device uint *argmax [[buffer(2)]],
    device float *entropy [[buffer(3)]],
    constant CanvasRowStatsParams &params [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  const uint vocabulary = params.vocabulary;
  device const bfloat *r = logits + ulong(row) * vocabulary;
  const uint slice = (vocabulary + 255) / 256;
  const uint begin = thread_index * slice;
  const uint end = min(begin + slice, vocabulary);

  float local_max = -INFINITY;
  for (uint i = begin; i < end; ++i)
    local_max = max(local_max, r[i]);
  const float row_max_v = simd_max(local_max);
  threadgroup float maxima[8];
  if (thread_index % 32 == 0)
    maxima[thread_index / 32] = row_max_v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float m = simd_max(thread_index < 8 ? maxima[thread_index] : -INFINITY);
  // Broadcast via a second scratch write so every thread agrees.
  threadgroup float row_maximum[1];
  if (thread_index == 0)
    row_maximum[0] = m;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float mx = row_maximum[0];

  float sum = 0.0f, weighted = 0.0f;
  float best = -INFINITY;
  uint best_index = 0;
  for (uint i = begin; i < end; ++i) {
    const float l = float(r[i]);
    const float e = fast::exp(l - mx);
    sum += e;
    weighted += e * l;
    if (l > best) {
      best = l;
      best_index = i;
    }
  }
  // Row sums/argmax across the 256 threads: simd reduce then cross-group.
  threadgroup float sums[8], weights[8];
  threadgroup float best_values[8];
  threadgroup uint best_indices[8];
  const uint lane = thread_index % 32, sg = thread_index / 32;
  const float s_sum = simd_sum(sum);
  const float s_weighted = simd_sum(weighted);
  const float s_best = simd_max(best);
  if (lane == 0) {
    sums[sg] = s_sum;
    weights[sg] = s_weighted;
    best_values[sg] = s_best;
    best_indices[sg] = best_index;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index == 0) {
    float total = 0.0f, wtotal = 0.0f, bv = -INFINITY;
    for (uint i = 0; i < 8; ++i) {
      total += sums[i];
      wtotal += weights[i];
      bv = max(bv, best_values[i]);
    }
    sums[0] = total;
    weights[0] = wtotal;
    best_values[0] = bv;
    entropy[row] = fast::log(total) + mx - wtotal / total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Argmax: the lowest index holding the row max, via an atomic min.
  threadgroup atomic_uint argmax_out[1];
  if (thread_index == 0)
    atomic_store_explicit(&argmax_out[0], vocabulary, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (best == best_values[0])
    atomic_fetch_min_explicit(&argmax_out[0], best_index,
                              memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Multinomial draw: u in [0, Z). Each thread's slice mass; an exclusive
  // scan (thread order) finds the owning thread.
  const float Z = sums[0];
  const float u =
      richengine_canvas_uniform(params.seed, row) * Z;
  threadgroup float mass[256];
  mass[thread_index] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index == 0) {
    float running = 0.0f;
    for (uint i = 0; i < 256; ++i) {
      const float m_i = mass[i];
      mass[i] = running;
      running += m_i;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float prefix = mass[thread_index];
  const bool owner = u >= prefix && u < prefix + sum;
  if (owner) {
    float cumulative = prefix;
    uint drawn = end - 1;
    for (uint i = begin; i < end; ++i) {
      cumulative += fast::exp(r[i] - mx);
      if (u < cumulative) {
        drawn = i;
        break;
      }
    }
    sampled[row] = drawn;
  }
  if (thread_index == 0)
    argmax[row] =
        atomic_load_explicit(&argmax_out[0], memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// Entropy-bound accept. One threadgroup of 256 sorts the canvas rows'
// entropies ascending (single bitonic network over 256 entries), takes the
// exclusive prefix sum of the sorted order, and accepts position i iff
// cumsum_excluded <= entropy_bound — i.e. the cheapest positions first, the
// running mass of cheaper positions never exceeding the bound. Accepted
// positions keep their sampled token; rejected ones are renoised from the
// uniform hash.
//
// Writes the new canvas token buffer, the rows' argmax tokens (for the
// integrator's commit + early-exit bookkeeping), and a stats record:
//   stats[0] = mean entropy over the canvas
//   stats[1] = 1.0 if every argmax equals the previous step's argmax buffer
//              (bind the same buffer twice to force 0/1 as desired; pass the
//              previous canvas argmax buffer at buffer 4)
//
// Buffers:
//   0 entropy    float [256]  (from canvas_row_stats)
//   1 sampled    uint  [256]  (denoised proposals)
//   2 argmax_in  uint  [256]  (previous step's argmax; may alias argmax_out
//                            of the prior step; bind a zero buffer on step 0)
//   3 canvas_out uint  [256]
//   4 argmax_out uint  [256]
//   5 stats      float [2]
//   6 params     CanvasAcceptParams {entropy_bound, vocabulary, seed, rows}
//   7 argmax_cur uint  [rows]  (this step's argmax, from canvas_row_stats)
//
// `rows` can be below CanvasRows: the sort still runs over 256 slots with
// dead rows padded at +inf — they sink to the end of the order, never
// satisfy the bound, and are skipped on write-out.
struct CanvasAcceptParams {
  float entropy_bound;
  uint32_t vocabulary;
  uint32_t seed;
  uint32_t rows;
};

kernel void canvas_entropy_accept(
    device const float *entropy [[buffer(0)]],
    device const uint *sampled [[buffer(1)]],
    device const uint *argmax_in [[buffer(2)]],
    device uint *canvas_out [[buffer(3)]],
    device uint *argmax_out [[buffer(4)]],
    device float *stats [[buffer(5)]],
    constant CanvasAcceptParams &params [[buffer(6)]],
    device const uint *argmax_cur [[buffer(7)]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  threadgroup float keys[CanvasRows];
  threadgroup uint order[CanvasRows];
  const uint i = thread_index;
  keys[i] = i < params.rows ? entropy[i] : INFINITY;
  order[i] = i;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Bitonic sort, ascending by entropy; the original index travels along.
  for (uint k = 2; k <= CanvasRows; k <<= 1) {
    for (uint j = k >> 1; j > 0; j >>= 1) {
      const uint ixj = i ^ j;
      if (ixj > i) {
        const bool descending = ((i & k) != 0);
        const bool swap = descending ? keys[i] < keys[ixj]
                                     : keys[i] > keys[ixj];
        if (swap) {
          const float tk = keys[i];
          const uint to = order[i];
          keys[i] = keys[ixj];
          order[i] = order[ixj];
          keys[ixj] = tk;
          order[ixj] = to;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  // Exclusive prefix sum over the sorted entropies (256 additions, thread 0;
  // the accept test is latency-trivial against the 67M-logit row pass). The
  // mean entropy spans the live rows only — dead rows carry +inf.
  if (i == 0) {
    float running = 0.0f, live_total = 0.0f;
    for (uint k = 0; k < CanvasRows; ++k) {
      const float e = keys[k];
      if (k < params.rows)
        live_total += e;
      keys[k] = running; // keys[] now holds cumsum excluding e_sorted[k]
      running += e;
    }
    stats[0] = live_total / float(params.rows);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Rank i's row: accept iff the cheaper positions' mass fits the bound.
  const uint position = order[i];
  const bool live = position < params.rows;
  const bool accept = keys[i] <= params.entropy_bound;
  if (live) {
    canvas_out[position] =
        accept ? sampled[position]
               : richengine_canvas_hash(params.seed, position) %
                     params.vocabulary;
    // The current argmax is copied through for the next step's comparison
    // and for the integrator's commit path.
    argmax_out[position] = argmax_cur[position];
  }
  const bool mismatch =
      live && argmax_cur[position] != argmax_in[position];
  threadgroup atomic_uint *mismatches =
      (threadgroup atomic_uint *)&keys[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (i == 0)
    atomic_store_explicit(mismatches, 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (mismatch)
    atomic_fetch_add_explicit(mismatches, 1u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (i == 0)
    stats[1] = atomic_load_explicit(mismatches, memory_order_relaxed) == 0
                   ? 1.0f
                   : 0.0f;
}

// ---------------------------------------------------------------------------
// Soft embeddings: the [256, 262144] x [262144, 2816] softmax-weighted
// embedding average. The exact form is upstream's
// softmax(logits) @ embed_tokens * sqrt(2816) — 192 G MACs, kept as the
// correctness reference. The production path is canvas_soft_embed_topk:
// bisect a probability threshold that keeps about top_k vocab rows per
// canvas row (up to 64x less work), renormalize the kept mass, and
// accumulate only those Q4 rows.
//
// Embedding storage is the packed Q4 plane format of embedding_q4_h2816,
// every plane in 256-row tiles of 64-dim group units:
//   weights: uchar [vocab/256][Hidden/64][256][32], dim d's code = nibble
//            d&1 of byte (d%64)/2 in the tile unit of group d/64
//   scales:  bfloat [vocab/256][Hidden/64][256] — same for biases
//   dequant(v, d) = code * scales[tile(v, d/64)] + biases[tile(v, d/64)]
//
// Both variants take logits ALREADY softcapped and temperature-scaled (the
// same buffer canvas_row_stats read). They recompute the row's softmax
// max/Z internally; no normalized-probs buffer is materialized.
//
// Buffers (both):
//   0 logits           fp32 [256][vocabulary]
//   1 weights          uchar [vocabulary][1408]
//   2 scales           bfloat [vocabulary][44]
//   3 biases           bfloat [vocabulary][44]
//   4 output           bfloat [256][2816]
//   5 params           CanvasSoftEmbedParams {vocabulary, top_k (exact: 0)}
//   6 embedding_scale  float (bind sqrt(2816))
struct CanvasSoftEmbedParams {
  uint32_t vocabulary;
  uint32_t top_k;
};

constant uint CanvasHidden = 2816;
constant uint CanvasQuantGroups = CanvasHidden / 64; // 44

inline float richengine_canvas_q4_value(
    device const uchar *weights, device const bfloat *scales,
    device const bfloat *biases, uint vocab_row, uint dim) {
  // Same 256-row tile order as embedding_q4_*: codes
  // [vocab/256][groups][256][32] bytes, scales/biases [vocab/256][groups][256].
  const ulong tile =
      (ulong(vocab_row / 256) * CanvasQuantGroups + dim / 64) * 256 +
      vocab_row % 256;
  const uchar packed = weights[tile * 32 + (dim % 64) / 2];
  const float code = float((packed >> ((dim & 1) * 4)) & 15);
  const ulong parameter = tile;
  return code * float(scales[parameter]) + float(biases[parameter]);
}

// Row max and softmax denominator Z, reduced over the threadgroup; every
// thread returns (mx, Z). Sums keep source order per thread slice.
inline float2 richengine_canvas_softmax_stats(
    device const bfloat *r, uint vocabulary, uint thread_index,
    threadgroup float *reduce8, threadgroup float *broadcast) {
  const uint slice = (vocabulary + 255) / 256;
  const uint begin = thread_index * slice;
  const uint end = min(begin + slice, vocabulary);
  float local_max = -INFINITY;
  for (uint i = begin; i < end; ++i)
    local_max = max(local_max, r[i]);
  float sg_max = simd_max(local_max);
  if (thread_index % 32 == 0)
    reduce8[thread_index / 32] = sg_max;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index == 0)
    broadcast[0] =
        simd_max(thread_index < 8 ? reduce8[thread_index] : -INFINITY);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float mx = broadcast[0];
  float sum = 0.0f;
  for (uint i = begin; i < end; ++i)
    sum += fast::exp(r[i] - mx);
  const float sg_sum = simd_sum(sum);
  if (thread_index % 32 == 0)
    reduce8[thread_index / 32] = sg_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index == 0) {
    float total = 0.0f;
    for (uint i = 0; i < 8; ++i)
      total += reduce8[i];
    broadcast[0] = total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return float2(mx, broadcast[0]);
}

// Accumulate dequant(embed[v]) * p into acc[] across the threadgroup: each
// thread owns dims d = thread_index + 256*j (11 dims for Hidden 2816).
inline void richengine_canvas_embed_add(
    device const uchar *weights, device const bfloat *scales,
    device const bfloat *biases, uint vocab_row, float p,
    threadgroup float *acc, uint thread_index) {
#pragma unroll
  for (uint j = 0; j < (CanvasHidden + 255) / 256; ++j) {
    const uint dim = thread_index + 256 * j;
    if (dim < CanvasHidden)
      acc[dim] += p * richengine_canvas_q4_value(weights, scales, biases,
                                               vocab_row, dim);
  }
}

kernel void canvas_soft_embed_exact(
    device const bfloat *logits [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device const bfloat *scales [[buffer(2)]],
    device const bfloat *biases [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    constant CanvasSoftEmbedParams &params [[buffer(5)]],
    constant float &embedding_scale [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  const uint vocabulary = params.vocabulary;
  device const bfloat *r = logits + ulong(row) * vocabulary;
  threadgroup float reduce8[8];
  threadgroup float broadcast[1];
  const float2 mz = richengine_canvas_softmax_stats(r, vocabulary,
                                                    thread_index, reduce8,
                                                    broadcast);
  const float mx = mz.x, z = mz.y;
  threadgroup float acc[CanvasHidden];
#pragma unroll
  for (uint j = 0; j < (CanvasHidden + 255) / 256; ++j) {
    const uint dim = thread_index + 256 * j;
    if (dim < CanvasHidden)
      acc[dim] = 0.0f;
  }
  threadgroup float prob[256];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint v0 = 0; v0 < vocabulary; v0 += 256) {
    const uint v = v0 + thread_index;
    float p = 0.0f;
    if (v < vocabulary)
      p = fast::exp(r[v] - mx) / z;
    prob[thread_index] = p;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 0; k < 256 && v0 + k < vocabulary; ++k)
      richengine_canvas_embed_add(weights, scales, biases, v0 + k, prob[k],
                                  acc, thread_index);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  device bfloat *o = output + ulong(row) * CanvasHidden;
  for (uint dim = thread_index; dim < CanvasHidden; dim += 256)
    o[dim] = bfloat(acc[dim] * embedding_scale);
}

kernel void canvas_soft_embed_topk(
    device const bfloat *logits [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device const bfloat *scales [[buffer(2)]],
    device const bfloat *biases [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    constant CanvasSoftEmbedParams &params [[buffer(5)]],
    constant float &embedding_scale [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  const uint vocabulary = params.vocabulary;
  const uint top_k = params.top_k == 0 ? 256 : params.top_k;
  device const bfloat *r = logits + ulong(row) * vocabulary;
  threadgroup float reduce8[8];
  threadgroup float broadcast[1];
  const float2 mz = richengine_canvas_softmax_stats(r, vocabulary,
                                                    thread_index, reduce8,
                                                    broadcast);
  const float mx = mz.x, z = mz.y;
  // Bisect a probability threshold tau keeping ~top_k rows: 32 iterations of
  // a strided count. count(p >= tau) is monotone in tau on [0, hi].
  float lo = 0.0f, hi = 1.0f;
  threadgroup uint counts[8];
  for (uint it = 0; it < 32; ++it) {
    const float mid = (lo + hi) * 0.5f;
    uint count = 0;
    for (uint v = thread_index; v < vocabulary; v += 256)
      count += fast::exp(r[v] - mx) / z >= mid ? 1u : 0u;
    const uint sg_count = simd_sum(count);
    if (thread_index % 32 == 0)
      counts[thread_index / 32] = sg_count;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint total = counts[0] + counts[1] + counts[2] + counts[3] + counts[4] +
                 counts[5] + counts[6] + counts[7];
    if (total > top_k)
      lo = mid;
    else
      hi = mid;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float tau = hi;
  // Mass and accumulation over qualifying rows. prob[] stages this block's
  // p so the serial inner loop reads it once per qualifying v.
  threadgroup float acc[CanvasHidden];
#pragma unroll
  for (uint j = 0; j < (CanvasHidden + 255) / 256; ++j) {
    const uint dim = thread_index + 256 * j;
    if (dim < CanvasHidden)
      acc[dim] = 0.0f;
  }
  threadgroup float prob[256];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float kept_mass = 0.0f;
  for (uint v0 = 0; v0 < vocabulary; v0 += 256) {
    const uint v = v0 + thread_index;
    float p = 0.0f;
    if (v < vocabulary)
      p = fast::exp(r[v] - mx) / z;
    p = p >= tau ? p : 0.0f;
    prob[thread_index] = p;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 0; k < 256 && v0 + k < vocabulary; ++k) {
      const float pk = prob[k];
      if (pk > 0.0f) {
        kept_mass += pk;
        richengine_canvas_embed_add(weights, scales, biases, v0 + k, pk, acc,
                                    thread_index);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  // kept_mass is per-thread replicated (every thread ran the same inner
  // adds); all agree, so no reduce.
  const float norm = kept_mass > 0.0f ? embedding_scale / kept_mass : 0.0f;
  device bfloat *o = output + ulong(row) * CanvasHidden;
  for (uint dim = thread_index; dim < CanvasHidden; dim += 256)
    o[dim] = bfloat(acc[dim] * norm);
}

// ---------------------------------------------------------------------------
// Fused-logit variants. Each takes a {cap, scale} pair applied in-register on
// load: cap > 0 applies cap * tanh(l / cap) (decode_logit_softcap's exact
// transform) and scale != 0 multiplies (canvas_logits_scale); both at 0 pass
// the logit through. With the pair bound nonzero the runtime skips the two
// standalone r+w passes over the [rows][vocabulary] buffer entirely —
// RICHENGINE_CANVAS_*_OFF selects the two-pass kernels for A/B.

struct CanvasLogitTransform {
  float cap;
  float scale;
};

inline float canvas_logit_transform(float l, CanvasLogitTransform t) {
  if (t.cap > 0.0f)
    l = t.cap * richengine_tanh(l / t.cap);
  if (t.scale != 0.0f)
    l *= t.scale;
  return l;
}

// Single-pass canvas_row_stats: each thread scans its slice once, keeping an
// online (max, Z, weighted) triple — the accumulators rescale when the
// running max moves — plus the slice argmax. One threadgroup merge resolves
// the row max, then the per-slice partials rescale to it for the draw's
// prefix sum exactly as canvas_row_stats lays them out. Reads the logits
// buffer once (the owner thread's rescan is 1/256 of the row).
struct CanvasRowStatsFusedParams {
  uint32_t vocabulary;
  uint32_t seed;
  CanvasLogitTransform transform;
};

kernel void canvas_row_stats_fused(
    device const bfloat *logits [[buffer(0)]],
    device uint *sampled [[buffer(1)]],
    device uint *argmax [[buffer(2)]],
    device float *entropy [[buffer(3)]],
    constant CanvasRowStatsFusedParams &params [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  const uint vocabulary = params.vocabulary;
  const CanvasLogitTransform xf = params.transform;
  device const bfloat *r = logits + ulong(row) * vocabulary;
  const uint slice = (vocabulary + 255) / 256;
  const uint begin = thread_index * slice;
  const uint end = min(begin + slice, vocabulary);

  float local_max = -INFINITY;
  float sum = 0.0f, weighted = 0.0f;
  float best = -INFINITY;
  uint best_index = 0;
  for (uint i = begin; i < end; ++i) {
    const float l = canvas_logit_transform(r[i], xf);
    if (l > local_max) {
      const float rescale =
          local_max == -INFINITY ? 0.0f : fast::exp(local_max - l);
      sum *= rescale;
      weighted *= rescale;
      local_max = l;
    }
    const float e = fast::exp(l - local_max);
    sum += e;
    weighted += e * l;
    if (l > best) {
      best = l;
      best_index = i;
    }
  }
  threadgroup float maxima[8];
  if (thread_index % 32 == 0)
    maxima[thread_index / 32] = simd_max(local_max);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  threadgroup float shared[2];
  if (thread_index == 0)
    shared[0] =
        simd_max(thread_index < 8 ? maxima[thread_index] : -INFINITY);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float mx = shared[0];
  // Rescale the slice partials onto the row max; the per-slice mass feeds
  // the draw's prefix sum below.
  const float rescale =
      local_max == -INFINITY ? 0.0f : fast::exp(local_max - mx);
  const float partial = sum * rescale;
  const float pweighted = weighted * rescale;
  threadgroup float mass[256];
  mass[thread_index] = partial;
  threadgroup float sums[8], weights[8];
  threadgroup float best_values[8];
  threadgroup uint best_indices[8];
  const uint lane = thread_index % 32, sg = thread_index / 32;
  const float s_sum = simd_sum(partial);
  const float s_weighted = simd_sum(pweighted);
  const float s_best = simd_max(best);
  if (lane == 0) {
    sums[sg] = s_sum;
    weights[sg] = s_weighted;
    best_values[sg] = s_best;
    best_indices[sg] = best_index;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index == 0) {
    float total = 0.0f, wtotal = 0.0f, bv = -INFINITY;
    for (uint i = 0; i < 8; ++i) {
      total += sums[i];
      wtotal += weights[i];
      bv = max(bv, best_values[i]);
    }
    sums[0] = total;
    best_values[0] = bv;
    shared[1] = total;
    entropy[row] = fast::log(total) + mx - wtotal / total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  threadgroup atomic_uint argmax_out[1];
  if (thread_index == 0)
    atomic_store_explicit(&argmax_out[0], vocabulary, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (best == best_values[0])
    atomic_fetch_min_explicit(&argmax_out[0], best_index,
                              memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const float Z = shared[1];
  const float u = richengine_canvas_uniform(params.seed, row) * Z;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index == 0) {
    float running = 0.0f;
    for (uint i = 0; i < 256; ++i) {
      const float m_i = mass[i];
      mass[i] = running;
      running += m_i;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float prefix = mass[thread_index];
  const bool owner = u >= prefix && u < prefix + partial;
  if (owner) {
    float cumulative = prefix;
    uint drawn = end - 1;
    for (uint i = begin; i < end; ++i) {
      cumulative +=
          fast::exp(canvas_logit_transform(r[i], xf) - mx);
      if (u < cumulative) {
        drawn = i;
        break;
      }
    }
    sampled[row] = drawn;
  }
  if (thread_index == 0)
    argmax[row] =
        atomic_load_explicit(&argmax_out[0], memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// Histogram soft-embed: the canvas_soft_embed_topk replacement. Instead of a
// 32-iteration bisection re-streaming the row, it makes a single histogram
// pass over the logits in 256 threadgroup-atomic bins spanning the row's
// [min, max] (p = exp(l - mx)/z is monotone in l, so a logit-domain bin
// threshold IS a probability threshold — the grid resolves whatever spread
// the row has, including the near-uniform case where every p falls in one
// probability bin), prefix-sums the bins top-down to bracket the threshold
// that keeps at most top_k rows, compacts the qualifying vocabulary indices
// into a threadgroup list on the third and final logits pass, then
// accumulates only those Q4 embedding rows (mass renormalized). Three
// logits passes total against the bisection's ~35.
//
// The kept set is count(l >= l_tau) with l_tau on the range/256 grid, at
// most top_k — a subset of the bisection's top_k. top_k > 1024 keeps the
// first 1024 compacted entries; a degenerate row (min == max) keeps none,
// matching the bisection's empty-set outcome.
struct CanvasSoftEmbedHistParams {
  uint32_t vocabulary;
  uint32_t top_k;
  CanvasLogitTransform transform;
};

constant uint CanvasHistListCapacity = 1024;

kernel void canvas_soft_embed_histogram(
    device const bfloat *logits [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device const bfloat *scales [[buffer(2)]],
    device const bfloat *biases [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    constant CanvasSoftEmbedHistParams &params [[buffer(5)]],
    constant float &embedding_scale [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  const uint vocabulary = params.vocabulary;
  const uint top_k = params.top_k == 0 ? 256 : params.top_k;
  const CanvasLogitTransform xf = params.transform;
  device const bfloat *r = logits + ulong(row) * vocabulary;
  const uint slice = (vocabulary + 255) / 256;
  const uint begin = thread_index * slice;
  const uint end = min(begin + slice, vocabulary);

  // Pass 1: online (max, min, Z) in one stream — the same online-softmax
  // merge canvas_row_stats_fused uses, plus the row minimum for the bin
  // range.
  float local_max = -INFINITY, local_min = INFINITY;
  float sum = 0.0f;
  for (uint i = begin; i < end; ++i) {
    const float l = canvas_logit_transform(r[i], xf);
    if (l > local_max) {
      sum *= local_max == -INFINITY ? 0.0f : fast::exp(local_max - l);
      local_max = l;
    }
    local_min = min(local_min, l);
    sum += fast::exp(l - local_max);
  }
  threadgroup float reduce8[8];
  threadgroup float minima[8];
  threadgroup float shared[2];
  const float sg_min = simd_min(local_min);
  if (thread_index % 32 == 0) {
    reduce8[thread_index / 32] = simd_max(local_max);
    minima[thread_index / 32] = sg_min;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index == 0) {
    shared[0] =
        simd_max(thread_index < 8 ? reduce8[thread_index] : -INFINITY);
    shared[1] =
        simd_min(thread_index < 8 ? minima[thread_index] : INFINITY);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float mx = shared[0];
  const float mn = shared[1];
  const float partial =
      local_max == -INFINITY ? 0.0f : sum * fast::exp(local_max - mx);
  const float sg_sum = simd_sum(partial);
  if (thread_index % 32 == 0)
    reduce8[thread_index / 32] = sg_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index == 0) {
    float total = 0.0f;
    for (uint i = 0; i < 8; ++i)
      total += reduce8[i];
    shared[1] = total;
  }

  // Pass 2: 256 bins uniform over [mn, mx]; bin b holds the count of
  // l in [mn + b*w, mn + (b+1)*w) with w = range/256. The descending bin
  // cumsum is count(l >= mn + b*w) — the same threshold grid the bisection
  // refines, in the domain p is monotone in.
  threadgroup atomic_uint bins[256];
  if (thread_index == 0)
    shared[0] = 0.0f; // reused below for tau_l; Z already consumed
  atomic_store_explicit(&bins[thread_index], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float z = shared[1];
  const float range = mx - mn;
  const float inv_width = range > 0.0f ? 256.0f / range : 0.0f;
  for (uint i = begin; i < end; ++i) {
    const float l = canvas_logit_transform(r[i], xf);
    const uint b = min(uint((l - mn) * inv_width), 255u);
    atomic_fetch_add_explicit(&bins[b], 1u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // l_tau = mn + b* * w where b* is the lowest grid point whose tail count
  // is at most top_k; b* == 256 (top bin already overfull, or degenerate)
  // maps to +inf so nothing qualifies, like the bisection's empty set.
  if (thread_index == 0) {
    uint cumulative = 0;
    uint bracket = 256;
    for (uint b = 256; b-- > 0;) {
      const uint next = cumulative + atomic_load_explicit(
          &bins[b], memory_order_relaxed);
      if (next > top_k && bracket == 256)
        bracket = b + 1;
      cumulative = next;
    }
    shared[0] = bracket == 256 ? INFINITY
                             : mn + float(bracket) * (range / 256.0f);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float tau_l = shared[0];

  // Pass 3: compact qualifying indices and their probabilities; the kept
  // count is the bin cumsum, at most top_k by construction of bracket.
  threadgroup atomic_uint kept_count[1];
  threadgroup uint kept_index[CanvasHistListCapacity];
  threadgroup float kept_prob[CanvasHistListCapacity];
  threadgroup float acc[CanvasHidden];
  if (thread_index == 0)
    atomic_store_explicit(&kept_count[0], 0u, memory_order_relaxed);
#pragma unroll
  for (uint j = 0; j < (CanvasHidden + 255) / 256; ++j) {
    const uint dim = thread_index + 256 * j;
    if (dim < CanvasHidden)
      acc[dim] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float kept_mass_local = 0.0f;
  for (uint i = begin; i < end; ++i) {
    const float l = canvas_logit_transform(r[i], xf);
    if (l >= tau_l) {
      const float p = fast::exp(l - mx) / z;
      kept_mass_local += p;
      const uint slot = atomic_fetch_add_explicit(&kept_count[0], 1u,
                                                  memory_order_relaxed);
      if (slot < CanvasHistListCapacity) {
        kept_index[slot] = i;
        kept_prob[slot] = p;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint kept =
      min(atomic_load_explicit(&kept_count[0], memory_order_relaxed),
          CanvasHistListCapacity);
  float kept_mass = simd_sum(kept_mass_local);
  if (thread_index % 32 == 0)
    reduce8[thread_index / 32] = kept_mass;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index == 0) {
    float total = 0.0f;
    for (uint i = 0; i < 8; ++i)
      total += reduce8[i];
    shared[1] = total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  kept_mass = shared[1];
  // One accumulation pass: only the kept rows' Q4 weights are read.
  for (uint k = 0; k < kept; ++k)
    richengine_canvas_embed_add(weights, scales, biases, kept_index[k],
                                kept_prob[k], acc, thread_index);
  const float norm =
      kept_mass > 0.0f ? embedding_scale / kept_mass : 0.0f;
  device bfloat *o = output + ulong(row) * CanvasHidden;
  for (uint dim = thread_index; dim < CanvasHidden; dim += 256)
    o[dim] = bfloat(acc[dim] * norm);
}

// ---------------------------------------------------------------------------
// Self-conditioning tail, fused: out = scaleless_rmsnorm(inputs + sc).
// The rest of the block is existing primitives: pre_norm is norm_rms on the
// soft-embedding signal, gate/up/down are the ordinary projection GEMMs, and
// geglu_multiply (gelu_tanh(gate) * up) builds `sc`'s input. One threadgroup
// of 256 per canvas row.
//
// Buffers:
//   0 inputs_embeds  bfloat [rows][hidden]
//   1 sc             bfloat [rows][hidden]  (down_proj output)
//   2 output         bfloat [rows][hidden]
//   3 width          uint (2816)
kernel void canvas_self_condition(
    device const bfloat *inputs_embeds [[buffer(0)]],
    device const bfloat *sc [[buffer(1)]],
    device bfloat *output [[buffer(2)]],
    constant uint &width [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  device const bfloat *x = inputs_embeds + ulong(row) * width;
  device const bfloat *s = sc + ulong(row) * width;
  device bfloat *o = output + ulong(row) * width;
  // The scaleless norm needs the summed row: stage x + sc, reduce, write.
  threadgroup bfloat summed[2816];
  float sum = 0.0f;
  for (uint c = thread_index; c < width; c += 256) {
    const float v = float(x[c]) + float(s[c]);
    summed[c] = bfloat(v);
    sum += v * v;
  }
  threadgroup float reductions[8];
  const float inverse = rms_inverse_of_sums<8>(sum, width, reductions,
                                             thread_index, lane, simd_group);
  for (uint c = thread_index; c < width; c += 256)
    o[c] = bfloat(float(summed[c]) * inverse);
}
