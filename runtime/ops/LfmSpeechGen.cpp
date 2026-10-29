#include "ops/LfmSpeechGen.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>

// Reference: dev/omni-reference/llamacpp/lfm3a-gen.cpp (GEN_CODE/GEN_WAV
// graphs), mtmd-helper-gen.cpp (sample_frame), mtmd-audio.cpp
// (streaming ISTFT), dev/omni-reference/NOTES.md sections 4-6.
namespace richengine::ops {
namespace {

constexpr float kEps = 1e-5F;

// Optional frame breakdown: LFM_SPEECH_PROFILE=1 logs per-section
// microseconds to stderr once per sampleFrame/decodeFrame call.
struct Prof {
  static bool on() {
    static const bool v = std::getenv("LFM_SPEECH_PROFILE") != nullptr;
    return v;
  }
  using C = std::chrono::steady_clock;
  C::time_point t0 = C::now();
  static double us(C::time_point a, C::time_point b) {
    return std::chrono::duration<double, std::micro>(b - a).count();
  }
  void mark(const char *name) {
    if (!on()) return;
    const auto t = C::now();
    std::fprintf(stderr, "    %-14s %9.1f us\n", name, us(t0, t));
    t0 = t;
  }
};

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define LFM_NEON 1
#endif

// bf16 -> f32 of the packed image rows.
float bf16ToF32(uint16_t v) {
  const uint32_t bits = uint32_t{v} << 16;
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

#ifndef LFM_NEON
float dotScalar(const float *a, const float *b, int n) {
  float acc = 0.0F;
  for (int i = 0; i < n; ++i) acc += a[i] * b[i];
  return acc;
}
#endif

#ifdef LFM_NEON
// Horizontally reduced a·b, 4 accumulator chains for ILP.
float dotNeon(const float *a, const float *b, int n) {
  float32x4_t s0 = vdupq_n_f32(0), s1 = s0, s2 = s0, s3 = s0;
  int i = 0;
  for (; i + 16 <= n; i += 16) {
    s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
    s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    s2 = vfmaq_f32(s2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
    s3 = vfmaq_f32(s3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
  }
  for (; i + 4 <= n; i += 4)
    s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
  float acc = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
  for (; i < n; ++i) acc += a[i] * b[i];
  return acc;
}

// Polynomial expf on 4 lanes (cephes-style): ~1 ulp, clamps to expf's
// useful range. Used where small fp drift is acceptable (silu, softmax).
float32x4_t expf4(float32x4_t x) {
  const float32x4_t hi = vdupq_n_f32(88.3762626647949F);
  const float32x4_t lo = vdupq_n_f32(-88.3762626647949F);
  x = vminq_f32(vmaxq_f32(x, lo), hi);
  const float32x4_t invLn2 = vdupq_n_f32(1.4426950408889634F);
  float32x4_t n = vrndnq_f32(vmulq_f32(x, invLn2));
  float32x4_t r = vfmsq_f32(x, n, vdupq_n_f32(0.693359375F));
  r = vfmsq_f32(r, n, vdupq_n_f32(-2.12194440e-4F));
  float32x4_t y = vdupq_n_f32(1.9875691500e-4F);
  y = vfmaq_f32(vdupq_n_f32(1.3981999507e-3F), y, r);
  y = vfmaq_f32(vdupq_n_f32(8.3334519073e-3F), y, r);
  y = vfmaq_f32(vdupq_n_f32(4.1665795894e-2F), y, r);
  y = vfmaq_f32(vdupq_n_f32(1.6666665459e-1F), y, r);
  y = vfmaq_f32(vdupq_n_f32(5.0000001201e-1F), y, r);
  y = vfmaq_f32(vdupq_n_f32(1.0F), y, r);
  y = vfmaq_f32(vdupq_n_f32(1.0F), y, r);
  const int32x4_t ni = vcvtq_s32_f32(n);
  const int32x4_t pow2 =
      vshlq_n_s32(vaddq_s32(ni, vdupq_n_s32(127)), 23);
  return vmulq_f32(y, vreinterpretq_f32_s32(pow2));
}
#endif

float dot(const float *a, const float *b, int n) {
#ifdef LFM_NEON
  return dotNeon(a, b, n);
#else
  return dotScalar(a, b, n);
#endif
}

#ifdef LFM_NEON
// 4x4 tile: rows r0..r0+3 against positions p0..p0+3, accumulating
// acc[row][pos]. Shares each x vector load across the 4 rows, halving
// load traffic versus the row-at-a-time kernel.
void matvecTile4(const float *w, const float *x, float *out, int rows,
                 int cols, int p0, const float *bias) {
  const int r4 = rows & ~3;
  for (int r0 = 0; r0 < r4; r0 += 4) {
    float32x4_t acc[4][4] = {};
    const float *w0 = w + size_t(r0) * cols;
    const float *w1 = w0 + cols;
    const float *w2 = w1 + cols;
    const float *w3 = w2 + cols;
    int c = 0;
    for (; c + 4 <= cols; c += 4) {
      const float32x4_t wv[4] = {vld1q_f32(w0 + c), vld1q_f32(w1 + c),
                                 vld1q_f32(w2 + c), vld1q_f32(w3 + c)};
      for (int j = 0; j < 4; ++j) {
        const float32x4_t xv =
            vld1q_f32(x + size_t(p0 + j) * cols + c);
        acc[0][j] = vfmaq_f32(acc[0][j], wv[0], xv);
        acc[1][j] = vfmaq_f32(acc[1][j], wv[1], xv);
        acc[2][j] = vfmaq_f32(acc[2][j], wv[2], xv);
        acc[3][j] = vfmaq_f32(acc[3][j], wv[3], xv);
      }
    }
    for (int i = 0; i < 4; ++i) {
      const float *row = w + size_t(r0 + i) * cols;
      for (int j = 0; j < 4; ++j) {
        float a = vaddvq_f32(acc[i][j]);
        for (int t = c; t < cols; ++t)
          a += row[t] * x[size_t(p0 + j) * cols + t];
        const int r = r0 + i;
        out[size_t(p0 + j) * rows + r] = (bias ? bias[r] : 0.0F) + a;
      }
    }
  }
  for (int r = r4; r < rows; ++r) {
    const float *row = w + size_t(r) * cols;
    for (int j = 0; j < 4; ++j)
      out[size_t(p0 + j) * rows + r] =
          (bias ? bias[r] : 0.0F) +
          dotNeon(row, x + size_t(p0 + j) * cols, cols);
  }
}

// 4x6 tile: the codec runs 6 positions through the weights; a 4-row by
// 6-column accumulator block (24 NEON regs) serves them in one scan.
void matvecTile6(const float *w, const float *x, float *out, int rows,
                 int cols, int p0, const float *bias) {
  const int r4 = rows & ~3;
  for (int r0 = 0; r0 < r4; r0 += 4) {
    float32x4_t acc[4][6] = {};
    const float *wr[4] = {w + size_t(r0) * cols, w + size_t(r0 + 1) * cols,
                          w + size_t(r0 + 2) * cols,
                          w + size_t(r0 + 3) * cols};
    int c = 0;
    for (; c + 4 <= cols; c += 4) {
      const float32x4_t wv[4] = {vld1q_f32(wr[0] + c), vld1q_f32(wr[1] + c),
                                 vld1q_f32(wr[2] + c), vld1q_f32(wr[3] + c)};
      for (int j = 0; j < 6; ++j) {
        const float32x4_t xv =
            vld1q_f32(x + size_t(p0 + j) * cols + c);
        for (int i = 0; i < 4; ++i)
          acc[i][j] = vfmaq_f32(acc[i][j], wv[i], xv);
      }
    }
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 6; ++j) {
        float a = vaddvq_f32(acc[i][j]);
        for (int t = c; t < cols; ++t)
          a += wr[i][t] * x[size_t(p0 + j) * cols + t];
        const int r = r0 + i;
        out[size_t(p0 + j) * rows + r] = (bias ? bias[r] : 0.0F) + a;
      }
  }
  for (int r = r4; r < rows; ++r) {
    const float *row = w + size_t(r) * cols;
    for (int j = 0; j < 6; ++j)
      out[size_t(p0 + j) * rows + r] =
          (bias ? bias[r] : 0.0F) +
          dotNeon(row, x + size_t(p0 + j) * cols, cols);
  }
}

// Same tile shape for a leftover group of 1-5 positions.
void matvecTileRem(const float *w, const float *x, float *out, int rows,
                   int cols, int p0, int pb, const float *bias) {
  const int r4 = rows & ~3;
  for (int r0 = 0; r0 < r4; r0 += 4) {
    float32x4_t acc[4][6] = {};
    const float *wr[4] = {w + size_t(r0) * cols, w + size_t(r0 + 1) * cols,
                          w + size_t(r0 + 2) * cols,
                          w + size_t(r0 + 3) * cols};
    int c = 0;
    for (; c + 4 <= cols; c += 4) {
      const float32x4_t wv[4] = {vld1q_f32(wr[0] + c), vld1q_f32(wr[1] + c),
                                 vld1q_f32(wr[2] + c), vld1q_f32(wr[3] + c)};
      for (int j = 0; j < pb; ++j) {
        const float32x4_t xv =
            vld1q_f32(x + size_t(p0 + j) * cols + c);
        for (int i = 0; i < 4; ++i)
          acc[i][j] = vfmaq_f32(acc[i][j], wv[i], xv);
      }
    }
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < pb; ++j) {
        float a = vaddvq_f32(acc[i][j]);
        for (int t = c; t < cols; ++t)
          a += wr[i][t] * x[size_t(p0 + j) * cols + t];
        const int r = r0 + i;
        out[size_t(p0 + j) * rows + r] = (bias ? bias[r] : 0.0F) + a;
      }
  }
  for (int r = r4; r < rows; ++r) {
    const float *row = w + size_t(r) * cols;
    for (int j = 0; j < pb; ++j)
      out[size_t(p0 + j) * rows + r] =
          (bias ? bias[r] : 0.0F) +
          dotNeon(row, x + size_t(p0 + j) * cols, cols);
  }
}
#endif

// out[p*rows + r] = sum_c w[r*cols + c] * x[p*cols + c] (+ bias[r]).
// One pass over w serves every position: the weight stream is the
// expensive part and x stays cache-resident.
void matvecRows(const float *w, const float *x, float *out, int rows,
                int cols, int nPos, const float *bias = nullptr) {
#ifdef LFM_NEON
  int p0 = 0;
  while (nPos - p0 >= 6) {
    matvecTile6(w, x, out, rows, cols, p0, bias);
    p0 += 6;
  }
  if (nPos - p0 >= 4) {
    matvecTile4(w, x, out, rows, cols, p0, bias);
    p0 += 4;
  }
  const int rem = nPos - p0;
  if (rem > 0) matvecTileRem(w, x, out, rows, cols, p0, rem, bias);
#else
  for (int r = 0; r < rows; ++r) {
    const float *row = w + size_t(r) * cols;
    for (int p = 0; p < nPos; ++p)
      out[size_t(p) * rows + r] =
          (bias ? bias[r] : 0.0F) +
          dotScalar(row, x + size_t(p) * cols, cols);
  }
#endif
}

void matvec(const float *w, const float *x, float *out, int rows, int cols,
            const float *bias = nullptr) {
  matvecRows(w, x, out, rows, cols, 1, bias);
}

// y = x * (w / rms(x)), fp32.
void rmsNorm(const float *x, const float *w, float *y, int n) {
  const float scale = float(1.0 / std::sqrt(dot(x, x, n) / n + kEps));
  int i = 0;
#ifdef LFM_NEON
  const float32x4_t sv = vdupq_n_f32(scale);
  for (; i + 4 <= n; i += 4)
    vst1q_f32(y + i,
              vmulq_f32(vmulq_f32(vld1q_f32(x + i), sv), vld1q_f32(w + i)));
#endif
  for (; i < n; ++i) y[i] = x[i] * scale * w[i];
}

// NEOX rope, full head dimension, theta 1e6: pairs (i, i+hd/2).
// The freq table is fixed per headDim; cos/sin per position are computed
// once and shared by every head at that position.
const float *ropeFreqs(int headDim, float theta) {
  static thread_local std::array<float, 64> cache{};
  static thread_local int cachedDim = -1;
  const int half = headDim / 2;
  if (cachedDim != headDim) {
    for (int i = 0; i < half; ++i)
      cache[i] = std::pow(theta, -float(2 * i) / float(headDim));
    cachedDim = headDim;
  }
  return cache.data();
}

void ropeCosSin(float *cv, float *sv, int headDim, int pos, float theta) {
  const float *freq = ropeFreqs(headDim, theta);
  const int half = headDim / 2;
  for (int i = 0; i < half; ++i) {
    const float ang = float(pos) * freq[i];
    cv[i] = std::cos(ang);
    sv[i] = std::sin(ang);
  }
}

void ropeApply(float *vec, int nHeads, int headDim, const float *cv,
               const float *sv) {
  const int half = headDim / 2;
  for (int h = 0; h < nHeads; ++h) {
    float *v = vec + size_t(h) * headDim;
    for (int i = 0; i < half; ++i) {
      const float c = cv[i], s = sv[i];
      const float x0 = v[i], x1 = v[i + half];
      v[i] = x0 * c - x1 * s;
      v[i + half] = x0 * s + x1 * c;
    }
  }
}

// One bidirectional/sliding attention op shared by both towers: cur is
// [hidden x pos]; writes out [hidden x pos]. cache (nullable) holds the
// last prefix k/v positions for streaming (wav only). Window semantics:
// depthformer passes nPos and no cache (bidirectional full); the codec
// passes the causal sliding window over [cache | chunk].
struct AttnSpec {
  int nHeads;
  int nKvHeads;
  int headDim;
  const std::vector<float> *wq;
  const std::vector<float> *wk;
  const std::vector<float> *wv;
  const std::vector<float> *qNorm;
  const std::vector<float> *kNorm;
  const std::vector<float> *gate;
  const std::vector<float> *o;
};

void attention(const AttnSpec &s, const float *cur, float *out, int hidden,
               int nPos, const int32_t *positions, bool causal,
               const std::vector<float> *kCache,
               const std::vector<float> *vCache,
               std::vector<float> *kOut, std::vector<float> *vOut) {
  const int qDim = s.nHeads * s.headDim;
  const int kvDim = s.nKvHeads * s.headDim;
  // Reused across calls: scratch capacity survives between frames.
  static thread_local std::vector<float> q, k, v;
  q.resize(size_t(qDim) * nPos);
  k.resize(size_t(kvDim) * nPos);
  v.resize(size_t(kvDim) * nPos);
  matvecRows(s.wq->data(), cur, q.data(), qDim, hidden, nPos);
  matvecRows(s.wk->data(), cur, k.data(), kvDim, hidden, nPos);
  matvecRows(s.wv->data(), cur, v.data(), kvDim, hidden, nPos);
  // per-head q/k RMSNorm then rope
  std::array<float, 32> cosT{}, sinT{};
  for (int p = 0; p < nPos; ++p) {
    for (int h = 0; h < s.nHeads; ++h)
      rmsNorm(q.data() + size_t(p) * qDim + size_t(h) * s.headDim,
              s.qNorm->data(),
              q.data() + size_t(p) * qDim + size_t(h) * s.headDim, s.headDim);
    for (int h = 0; h < s.nKvHeads; ++h)
      rmsNorm(k.data() + size_t(p) * kvDim + size_t(h) * s.headDim,
              s.kNorm->data(),
              k.data() + size_t(p) * kvDim + size_t(h) * s.headDim, s.headDim);
    ropeCosSin(cosT.data(), sinT.data(), s.headDim, positions[p], 1e6F);
    ropeApply(q.data() + size_t(p) * qDim, s.nHeads, s.headDim, cosT.data(),
              sinT.data());
    ropeApply(k.data() + size_t(p) * kvDim, s.nKvHeads, s.headDim, cosT.data(),
              sinT.data());
  }
  // [cache | chunk] key/value rows
  const int prefix = kCache ? int(kCache->size()) / kvDim : 0;
  const int nKv = prefix + nPos;
  static thread_local std::vector<float> kAll, vAll;
  kAll.resize(size_t(kvDim) * nKv);
  vAll.resize(size_t(kvDim) * nKv);
  if (prefix) {
    std::memcpy(kAll.data(), kCache->data(), kCache->size() * sizeof(float));
    std::memcpy(vAll.data(), vCache->data(), vCache->size() * sizeof(float));
  }
  std::memcpy(kAll.data() + size_t(prefix) * kvDim, k.data(),
              k.size() * sizeof(float));
  std::memcpy(vAll.data() + size_t(prefix) * kvDim, v.data(),
              v.size() * sizeof(float));
  if (kOut) {
    // Keep the last kAttnPrefix positions; while the window is still
    // filling (nKv < prefix) only the real rows are stored — the previous
    // version sliced from before kAll.begin() (UB, garbage k/v).
    const size_t keep =
        size_t(std::min(LfmSpeechGen::kAttnPrefix, nKv)) * kvDim;
    kOut->assign(kAll.end() - keep, kAll.end());
    vOut->assign(vAll.end() - keep, vAll.end());
  }
  const float scale = 1.0F / std::sqrt(float(s.headDim));
  const int group = s.nHeads / s.nKvHeads;
  static thread_local std::vector<float> attn, score;
  attn.assign(size_t(qDim) * nPos, 0.0F);
  score.resize(nKv);
  for (int p = 0; p < nPos; ++p) {
    for (int h = 0; h < s.nHeads; ++h) {
      const int kv = h / group;
      const float *qp = q.data() + size_t(p) * qDim + size_t(h) * s.headDim;
      const int visible = causal ? prefix + p + 1 : nKv;
      float mx = -INFINITY;
      for (int t = 0; t < visible; ++t) {
        const float *kp = kAll.data() + size_t(t) * kvDim +
                          size_t(kv) * s.headDim;
        const float acc = dot(qp, kp, s.headDim) * scale;
        score[t] = acc;
        mx = std::max(mx, acc);
      }
      double sum = 0.0;
      for (int t = 0; t < visible; ++t) {
        score[t] = std::exp(score[t] - mx);
        sum += score[t];
      }
      float *ap = attn.data() + size_t(p) * qDim + size_t(h) * s.headDim;
      for (int t = 0; t < visible; ++t) {
        const float wt = float(score[t] / sum);
        const float *vp =
            vAll.data() + size_t(t) * kvDim + size_t(kv) * s.headDim;
        int d = 0;
#ifdef LFM_NEON
        const float32x4_t wv = vdupq_n_f32(wt);
        for (; d + 4 <= s.headDim; d += 4)
          vst1q_f32(ap + d,
                    vfmaq_f32(vld1q_f32(ap + d), wv, vld1q_f32(vp + d)));
#endif
        for (; d < s.headDim; ++d) ap[d] += wt * vp[d];
      }
    }
  }
  // per-head sigmoid gate on the attention output, before o_proj
  if (s.gate) {
    static thread_local std::vector<float> g;
    g.resize(size_t(s.nHeads) * nPos);
    matvecRows(s.gate->data(), cur, g.data(), s.nHeads, hidden, nPos);
    for (int p = 0; p < nPos; ++p) {
      for (int h = 0; h < s.nHeads; ++h) {
        const float sig = 1.0F / (1.0F + std::exp(-g[size_t(p) * s.nHeads + h]));
        for (int d = 0; d < s.headDim; ++d)
          attn[size_t(p) * qDim + size_t(h) * s.headDim + d] *= sig;
      }
    }
  }
  matvecRows(s.o->data(), attn.data(), out, hidden, qDim, nPos);
}

// y = w2(silu(w1 x) * w3 x) with optional post norms handled by the caller.
void ffn(const std::vector<float> &w1, const std::vector<float> &w3,
         const std::vector<float> &w2, const float *x, float *out, int hidden,
         int inter, int nPos) {
  static thread_local std::vector<float> a, b;
  a.resize(size_t(inter) * nPos);
  b.resize(size_t(inter) * nPos);
  matvecRows(w1.data(), x, a.data(), inter, hidden, nPos);
  matvecRows(w3.data(), x, b.data(), inter, hidden, nPos);
  const size_t total = size_t(inter) * nPos;
  size_t i = 0;
#ifdef LFM_NEON
  const float32x4_t one = vdupq_n_f32(1.0F);
  for (; i + 4 <= total; i += 4) {
    const float32x4_t g = vld1q_f32(a.data() + i);
    const float32x4_t sig = vdivq_f32(one, vaddq_f32(one, expf4(vnegq_f32(g))));
    vst1q_f32(b.data() + i,
              vmulq_f32(vld1q_f32(b.data() + i), vmulq_f32(g, sig)));
  }
#endif
  for (; i < total; ++i) {
    const float g = a[i];
    b[i] *= g / (1.0F + std::exp(-g)); // silu
  }
  matvecRows(w2.data(), b.data(), out, hidden, inter, nPos);
}

// Generic LFM block: h=norm(x,ln1); y=op(h); post1?y=norm(y,post1); x+=y;
// h=norm(x,ln2); f=ffn; post2?f=norm(f,post2); x+=f.
template <typename Layer>
void lfmLayer(const Layer &layer, float *cur, int hidden, int inter, int nPos,
              const std::function<void(const float *, float *)> &op) {
  static thread_local std::vector<float> h, y;
  h.resize(size_t(hidden) * nPos);
  y.resize(size_t(hidden) * nPos);
  for (int p = 0; p < nPos; ++p)
    rmsNorm(cur + size_t(p) * hidden, layer.ln1.data(),
            h.data() + size_t(p) * hidden, hidden);
  op(h.data(), y.data());
  if constexpr (requires { layer.post1; }) {
    if (!layer.post1.empty())
      for (int p = 0; p < nPos; ++p)
        rmsNorm(y.data() + size_t(p) * hidden, layer.post1.data(),
                y.data() + size_t(p) * hidden, hidden);
  }
  for (size_t i = 0; i < y.size(); ++i) cur[i] += y[i];
  for (int p = 0; p < nPos; ++p)
    rmsNorm(cur + size_t(p) * hidden, layer.ln2.data(),
            h.data() + size_t(p) * hidden, hidden);
  ffn(layer.ffnGate, layer.ffnUp, layer.ffnDown, h.data(), y.data(), hidden,
      inter, nPos);
  if constexpr (requires { layer.post2; }) {
    if (!layer.post2.empty())
      for (int p = 0; p < nPos; ++p)
        rmsNorm(y.data() + size_t(p) * hidden, layer.post2.data(),
                y.data() + size_t(p) * hidden, hidden);
  }
  for (size_t i = 0; i < y.size(); ++i) cur[i] += y[i];
}

// The LFM short conv (wav decoder, conv_L_cache 7): bcx=in_proj(x) split
// [B|C|X]; bx = transpose(b*x) [pos x C]; causal depthwise over
// [conv_state | bx]; y = c * conv; out = out_proj(y). taps stays as the
// last L-1 bx rows.
void shortConv(const LfmSpeechWeights::WavLayer &layer,
               std::vector<float> &taps, const float *cur, float *out,
               int nPos) {
  const int C = LfmSpeechGen::kWavHidden, L = LfmSpeechGen::kConvTaps;
  const int prefix = L - 1; // taps holds the last L-1 b*x rows
  if (taps.size() != size_t(prefix) * C) taps.assign(size_t(prefix) * C, 0.0F);
  static thread_local std::vector<float> bx, c, bcx, y;
  bx.resize(size_t(prefix + nPos) * C);
  c.resize(size_t(nPos) * C);
  bcx.resize(size_t(3 * C) * nPos);
  matvecRows(layer.convPw1.data(), cur, bcx.data(), 3 * C, C, nPos);
  for (int p = 0; p < nPos; ++p) {
    const float *row = bcx.data() + size_t(p) * 3 * C;
    for (int i = 0; i < C; ++i) {
      bx[size_t(prefix + p) * C + i] = row[i] * row[2 * C + i]; // b * x
      c[size_t(p) * C + i] = row[C + i];
    }
  }
  std::memcpy(bx.data(), taps.data(), taps.size() * sizeof(float));
  taps.assign(bx.end() - size_t(prefix) * C, bx.end());
  y.assign(size_t(nPos) * C, 0.0F);
  for (int p = 0; p < nPos; ++p) {
    float *yp = y.data() + size_t(p) * C;
    for (int k = 0; k < L; ++k) {
      const float *row = bx.data() + size_t(prefix + p - (L - 1) + k) * C;
      const float *dw = layer.convDw.data() + size_t(k) * C;
      int i = 0;
#ifdef LFM_NEON
      for (; i + 4 <= C; i += 4)
        vst1q_f32(yp + i, vfmaq_f32(vld1q_f32(yp + i), vld1q_f32(dw + i),
                                    vld1q_f32(row + i)));
#endif
      for (; i < C; ++i) yp[i] += dw[i] * row[i];
    }
    const float *cp = c.data() + size_t(p) * C;
    int i = 0;
#ifdef LFM_NEON
    for (; i + 4 <= C; i += 4)
      vst1q_f32(yp + i, vmulq_f32(vld1q_f32(yp + i), vld1q_f32(cp + i)));
#endif
    for (; i < C; ++i) yp[i] *= cp[i];
  }
  matvecRows(layer.convPw2.data(), y.data(), out, C, C, nPos);
}

// Inverse real FFT of a full Hermitian spectrum (n = 1280 = 2^8 * 5 here).
// Recursive decimation-in-time over interleaved complex bins, positive-sign
// twiddles, 1/n scaling at the end — the same iDFT the O(n^2) version
// computed, at O(n log n).
struct Cx {
  float re, im;
};

Cx twRoot(int n, int k) {
  const float a = 2.0F * float(M_PI) * float(k) / float(n);
  return {std::cos(a), std::sin(a)};
}

// Direct small DFT for prime remainders (n=5 here).
void dftSmall(const float *in, int stride, Cx *out, int n, int nFull) {
  const int sub = nFull / n;
  for (int k = 0; k < n; ++k) {
    float re = 0.0F, im = 0.0F;
    for (int j = 0; j < n; ++j) {
      // twiddle e^{+2 pi i j k / n} as a root of the full transform
      const int idx = j * k % n * sub;
      const Cx w = {std::cos(2.0F * float(M_PI) * float(idx) / float(nFull)),
                    std::sin(2.0F * float(M_PI) * float(idx) / float(nFull))};
      const float xr = in[size_t(j) * stride * 2];
      const float xi = in[size_t(j) * stride * 2 + 1];
      re += xr * w.re - xi * w.im;
      im += xr * w.im + xi * w.re;
    }
    out[k] = {re, im};
  }
}

// in: interleaved complex, strided; out: n Cx.
void ifftRec(const float *in, int stride, Cx *out, int n, int nFull) {
  if (n % 2 == 0) {
    const int h = n / 2;
    ifftRec(in, 2 * stride, out, h, nFull);
    ifftRec(in + 2 * stride, 2 * stride, out + h, h, nFull);
    const int sub = nFull / n;
    for (int k = 0; k < h; ++k) {
      const Cx w = twRoot(nFull, k * sub);
      const Cx o = out[k + h];
      const float tr = o.re * w.re - o.im * w.im;
      const float ti = o.re * w.im + o.im * w.re;
      out[k + h] = {out[k].re - tr, out[k].im - ti};
      out[k] = {out[k].re + tr, out[k].im + ti};
    }
    return;
  }
  dftSmall(in, stride, out, n, nFull);
}

void irfft(const float *spec, float *out, int n) {
  static thread_local std::vector<Cx> buf;
  if (int(buf.size()) < n) buf.resize(n);
  ifftRec(spec, 1, buf.data(), n, n);
  const float inv = 1.0F / float(n);
  for (int t = 0; t < n; ++t) out[t] = buf[t].re * inv;
}

} // namespace

std::vector<float> lfmHannWindow(int n) {
  std::vector<float> w(n);
  for (int i = 0; i < n; ++i)
    w[i] = 0.5F - 0.5F * std::cos(2.0F * float(M_PI) * float(i) / float(n));
  return w;
}

// proj: projIn(hState)+bias, [CB][H], computed once per frame by the
// caller (it is identical across the four passes). need/nNeed: the
// codebook slots whose logits this pass must produce (0 -> head skipped,
// the pure-feedback pass needs no logits at all).
void LfmSpeechGen::depthForward(const float *proj, const int32_t *codes,
                                const float *mask, float *logits,
                                const int32_t *need, int nNeed,
                                float *fbOut) const {
  constexpr int H = kDepthHidden, CB = kCodebooks;
  Prof prof;
  // slot_i = proj_i + (mask_i ? embd_i[code_i] : mask_embd)
  static thread_local std::vector<float> cur;
  cur.assign(proj, proj + size_t(H) * CB);
  for (int i = 0; i < CB; ++i) {
    float *slot = cur.data() + size_t(i) * H;
    const float *row = mask[i] > 0.0F
                           ? weights_.embd.data() +
                                 (size_t(i) * kVocab + codes[i]) * H
                           : weights_.maskEmbd.data();
    for (int d = 0; d < H; ++d) slot[d] += row[d];
  }
  static const int32_t positions[CB] = {0, 1, 2, 3, 4, 5,
                                        6, 7, 8, 9, 10, 11};
  for (int il = 0; il < 3; ++il) {
    const auto &layer = weights_.depth[il];
    const AttnSpec s{32, 8, 32,  &layer.attnQ, &layer.attnK, &layer.attnV,
                     &layer.attnQNorm, &layer.attnKNorm, &layer.attnGate,
                     &layer.attnOut};
    lfmLayer(layer, cur.data(), H, 2816, CB,
             [&](const float *in, float *out) {
               attention(s, in, out, H, CB, positions, false, nullptr,
                         nullptr, nullptr, nullptr);
             });
    prof.mark(il == 0 ? "depth-l0" : il == 1 ? "depth-l1" : "depth-l2");
  }
  // trunk norm then the per-codebook head norm, tied embd head — only
  // for the codebook slots this pass actually samples from.
  for (int u = 0; u < nNeed; ++u) {
    const int i = need[u];
    float *slot = cur.data() + size_t(i) * H;
    rmsNorm(slot, weights_.outputNorm.data(), slot, H);
    rmsNorm(slot, weights_.headNorm.data() + size_t(i) * H, slot, H);
    matvec(weights_.embd.data() + size_t(i) * kVocab * H, slot,
           logits + size_t(i) * kVocab, kVocab, H);
  }
  prof.mark("embd-head");
  if (fbOut) {
    std::fill_n(fbOut, kLmHidden, 0.0F);
    for (int i = 0; i < CB; ++i) {
      const float *row =
          weights_.outEmbd.data() +
          (size_t(i) * kVocab + codes[i]) * kLmHidden;
      int d = 0;
#ifdef LFM_NEON
      for (; d + 4 <= kLmHidden; d += 4)
        vst1q_f32(fbOut + d,
                  vaddq_f32(vld1q_f32(fbOut + d), vld1q_f32(row + d)));
#endif
      for (; d < kLmHidden; ++d) fbOut[d] += row[d];
    }
  }
}

void LfmSpeechGen::repPenalty(float *logits, int cb) const {
  const size_t win = std::min(history_.size(), size_t(kRepWindow));
  std::array<bool, kVocab> seen{};
  for (size_t f = history_.size() - win; f < history_.size(); ++f) {
    const int32_t c = history_[f][cb];
    if (c >= 0 && c < kVocab && !seen[c]) {
      seen[c] = true;
      logits[c] = logits[c] > 0.0F ? logits[c] / kRepPenalty
                                   : logits[c] * kRepPenalty;
    }
  }
}

int32_t LfmSpeechGen::sampleCode(const float *logits) {
  const float mx = *std::max_element(logits, logits + kVocab);
  static thread_local std::vector<double> prob;
  prob.resize(kVocab);
  double sum = 0.0;
  for (int v = 0; v < kVocab; ++v) {
    prob[v] = std::exp(double(logits[v] - mx) / kAudioTemp);
    sum += prob[v];
  }
  double r = std::uniform_real_distribution<double>(0.0, 1.0)(rng_) * sum;
  for (int v = 0; v < kVocab; ++v) {
    r -= prob[v];
    if (r <= 0.0) return v;
  }
  return kVocab - 1;
}

bool LfmSpeechGen::sampleFrame(std::span<const float> hState,
                               std::array<int32_t, kCodebooks> &codes,
                               bool &eos, std::vector<float> &fb) {
  std::array<int, kCodebooks> order{};
  for (int i = 0; i < kCodebooks; ++i) order[i] = i;
  std::shuffle(order.begin(), order.end(), rng_);
  codes.fill(-1);
  std::array<float, kCodebooks> mask{};
  static thread_local std::vector<float> logits, proj;
  logits.resize(size_t(kCodebooks) * kVocab);
  // projIn(hState) is identical across the four passes: compute it once.
  proj.resize(size_t(kDepthHidden) * kCodebooks);
  matvec(weights_.projIn.data(), hState.data(), proj.data(),
         kDepthHidden * kCodebooks, kLmHidden, weights_.projInBias.data());
  int k = 0;
  for (const int pick : {4, 4, 4}) {
    depthForward(proj.data(), codes.data(), mask.data(), logits.data(),
                 order.data() + k, pick, nullptr);
    for (int j = 0; j < pick; ++j, ++k) {
      const int i = order[k];
      float *l = logits.data() + size_t(i) * kVocab;
      repPenalty(l, i);
      codes[i] = sampleCode(l);
      mask[i] = 1.0F;
    }
  }
  eos = std::find(codes.begin(), codes.end(), kEosCode) != codes.end();
  if (eos) codes.fill(kEosCode);
  else history_.push_back(codes);
  while (history_.size() > size_t(kRepWindow)) history_.pop_front();
  fb.assign(kLmHidden, 0.0F);
  depthForward(proj.data(), codes.data(), mask.data(), logits.data(),
               nullptr, 0, fb.data());
  return true;
}

std::vector<float>
LfmSpeechGen::decodeFrame(const std::array<int32_t, kCodebooks> &codes) {
  constexpr int C = kWavHidden;
  const int nPos = kFrameStack; // one codec frame per decodeFrame call
  Prof prof;
  // FSQ lookup + out_proj + in_proj
  static thread_local std::vector<float> q, latent, cur, spec, frame, col;
  q.resize(size_t(kFsqDim) * kCodebooks);
  for (int i = 0; i < kCodebooks; ++i)
    std::memcpy(q.data() + size_t(i) * kFsqDim,
                weights_.fsqTable.data() + size_t(codes[i]) * kFsqDim,
                kFsqDim * sizeof(float));
  latent.resize(kQuantDim);
  cur.resize(size_t(C) * nPos);
  matvec(weights_.quantOut.data(), q.data(), latent.data(), kQuantDim,
         kFsqDim * kCodebooks, weights_.quantOutB.data());
  matvec(weights_.tfmIn.data(), latent.data(), cur.data(), C, kQuantDim,
         weights_.tfmInB.data());
  // nearest-exact x6: every frame becomes 6 decoder positions; cur[0..C)
  // holds the single frame's projection, replicated per position.
  for (int p = nPos - 1; p >= 1; --p)
    std::copy_n(cur.data(), C, cur.data() + size_t(p) * C);
  prof.mark("in-proj");
  std::vector<int32_t> pos(nPos);
  for (int p = 0; p < nPos; ++p)
    pos[p] = int32_t(codec_.position + p);
  codec_.position += nPos;
  for (int il = 0; il < 8; ++il) {
    const auto &layer = weights_.wav[il];
    lfmLayer(layer, cur.data(), C, 2304, nPos,
             [&](const float *in, float *out) {
               if (layer.conv) {
                 shortConv(layer, codec_.convTaps[il], in, out, nPos);
                 return;
               }
               const AttnSpec s{8, 2, 64, &layer.attnQ, &layer.attnK,
                                &layer.attnV, &layer.attnQNorm,
                                &layer.attnKNorm, &layer.attnGate,
                                &layer.attnOut};
               std::vector<float> *kWin = &codec_.keyWindow[il];
               std::vector<float> *vWin = &codec_.valueWindow[il];
               attention(s, in, out, C, nPos, pos.data(), true,
                          kWin->empty() ? nullptr : kWin,
                          vWin->empty() ? nullptr : vWin, kWin, vWin);
             });
    prof.mark(layer.conv ? "wav-conv" : "wav-attn");
  }
  for (int p = 0; p < nPos; ++p)
    rmsNorm(cur.data() + size_t(p) * C, weights_.tfmOutNorm.data(),
            cur.data() + size_t(p) * C, C);
  // per column: [log_mag | angle] -> complex bins -> irfft -> Hann -> OLA
  static const std::vector<float> hann = lfmHannWindow(kNfft);
  spec.assign(size_t(kNfft) * 2, 0.0F);
  frame.resize(kNfft);
  std::vector<float> pcm;
  pcm.reserve(size_t(nPos) * kHop);
  for (int p = 0; p < nPos; ++p) {
    col.resize(2 * kStftBins);
    matvec(weights_.tfmOut.data(), cur.data() + size_t(p) * C, col.data(),
           2 * kStftBins, C, weights_.tfmOutB.data());
    for (int j = 0; j < kStftBins; ++j) {
      const float mag = std::exp(col[j]);
      spec[2 * j] = mag * std::cos(col[kStftBins + j]);
      spec[2 * j + 1] = mag * std::sin(col[kStftBins + j]);
    }
    for (int j = 1; j < kStftBins - 1; ++j) {
      spec[2 * (kNfft - j)] = spec[2 * j];
      spec[2 * (kNfft - j) + 1] = -spec[2 * j + 1];
    }
    prof.mark("tfmOut+spec");
    irfft(spec.data(), frame.data(), kNfft);
    prof.mark("irfft");
    for (int i = 0; i < kNfft; ++i) {
      codec_.windowSum[i] += hann[i] * hann[i];
      codec_.overlap[i] += frame[i] * hann[i];
    }
    for (int i = 0; i < kHop; ++i) {
      const float s = codec_.windowSum[i] > 1e-8F
                          ? codec_.overlap[i] / codec_.windowSum[i]
                          : codec_.overlap[i];
      pcm.push_back(s);
    }
    std::copy(codec_.overlap.begin() + kHop, codec_.overlap.end(),
              codec_.overlap.begin());
    std::fill(codec_.overlap.end() - kHop, codec_.overlap.end(), 0.0F);
    std::copy(codec_.windowSum.begin() + kHop, codec_.windowSum.end(),
              codec_.windowSum.begin());
    std::fill(codec_.windowSum.end() - kHop, codec_.windowSum.end(), 0.0F);
    const int drop = std::min(codec_.paddingToRemove, int(pcm.size()));
    codec_.paddingToRemove -= drop;
    pcm.erase(pcm.begin(), pcm.begin() + drop);
  }
  return pcm;
}

std::vector<float> LfmSpeechGen::flush() {
  std::vector<float> pcm;
  int remaining = kNfft - kHop;
  while (remaining > 0) {
    const int chunk = std::min(remaining, kHop);
    for (int i = 0; i < chunk; ++i) {
      const float s = codec_.windowSum[i] > 1e-8F
                          ? codec_.overlap[i] / codec_.windowSum[i]
                          : codec_.overlap[i];
      pcm.push_back(s);
    }
    std::copy(codec_.overlap.begin() + chunk, codec_.overlap.end(),
              codec_.overlap.begin());
    std::fill(codec_.overlap.end() - chunk, codec_.overlap.end(), 0.0F);
    std::copy(codec_.windowSum.begin() + chunk, codec_.windowSum.end(),
              codec_.windowSum.begin());
    std::fill(codec_.windowSum.end() - chunk, codec_.windowSum.end(), 0.0F);
    remaining -= chunk;
  }
  // the tail symmetric to the 480-sample head drop
  if (pcm.size() > 480) pcm.resize(pcm.size() - 480);
  else pcm.clear();
  return pcm;
}

// ---- packed image decode -------------------------------------------------

namespace {
std::vector<float> section(std::span<const uint16_t> image, uint64_t offset,
                           uint64_t count, const char *name) {
  std::vector<float> out(count);
  const uint16_t *src = image.data() + offset;
  for (uint64_t i = 0; i < count; ++i) out[i] = bf16ToF32(src[i]);
  (void)name;
  return out;
}
} // namespace

LfmSpeechWeights decodeLfmSpeechWeights(std::span<const uint16_t> image,
                                        std::span<const uint64_t> offsets) {
  LfmSpeechWeights w;
  uint32_t at = 0;
  const auto take = [&](std::vector<float> &dst, uint64_t count,
                        const char *name) {
    if (at >= offsets.size())
      throw std::invalid_argument("lfm-gen image ran out of sections");
    if (offsets[at] + count > image.size())
      throw std::invalid_argument(std::string("lfm-gen section ") + name +
                                  " exceeds the image");
    dst = section(image, offsets[at++], count, name);
  };
  take(w.projIn, 2048ULL * 12288, "proj_in");
  take(w.projInBias, 12288, "proj_in.bias");
  take(w.embd, 2017ULL * 12 * 1024, "embd");
  take(w.maskEmbd, 1024, "mask_embd");
  take(w.outEmbd, 24204ULL * 2048, "out_embd");
  take(w.outputNorm, 1024, "output_norm");
  take(w.headNorm, 12ULL * 1024, "head_norm");
  for (auto &l : w.depth) {
    take(l.ln1, 1024, "ln1");
    take(l.attnQ, 1024ULL * 1024, "attn_q");
    take(l.attnK, 256ULL * 1024, "attn_k");
    take(l.attnV, 256ULL * 1024, "attn_v");
    take(l.attnQNorm, 32, "attn_q_norm");
    take(l.attnKNorm, 32, "attn_k_norm");
    take(l.attnGate, 32ULL * 1024, "attn_gate");
    take(l.attnOut, 1024ULL * 1024, "attn_out");
    take(l.ln2, 1024, "ln2");
    take(l.ffnGate, 2816ULL * 1024, "ffn_gate");
    take(l.ffnUp, 2816ULL * 1024, "ffn_up");
    take(l.ffnDown, 1024ULL * 2816, "ffn_down");
  }
  take(w.fsqTable, 2016ULL * 4, "quant.codebook");
  take(w.quantOut, 48ULL * 256, "quant.out_proj");
  take(w.quantOutB, 256, "quant.out_proj.bias");
  take(w.tfmIn, 256ULL * 512, "tfm.in_proj");
  take(w.tfmInB, 512, "tfm.in_proj.bias");
  take(w.tfmOut, 512ULL * 1282, "tfm.out_proj");
  take(w.tfmOutB, 1282, "tfm.out_proj.bias");
  take(w.tfmOutNorm, 512, "tfm.output_norm");
  for (int il = 0; il < 8; ++il) {
    auto &l = w.wav[il];
    l.conv = il == 0 || il == 1 || il == 3 || il == 5 || il == 7;
    take(l.ln1, 512, "ln1");
    take(l.post1, 512, "attn_post_norm");
    if (l.conv) {
      take(l.convPw1, 512ULL * 1536, "conv_pw1");
      take(l.convDw, 7ULL * 512, "conv_dw");
      take(l.convPw2, 512ULL * 512, "conv_pw2");
    } else {
      take(l.attnQ, 512ULL * 512, "attn_q");
      take(l.attnK, 128ULL * 512, "attn_k");
      take(l.attnV, 128ULL * 512, "attn_v");
      take(l.attnQNorm, 64, "attn_q_norm");
      take(l.attnKNorm, 64, "attn_k_norm");
      take(l.attnGate, 8ULL * 512, "attn_gate");
      take(l.attnOut, 512ULL * 512, "attn_out");
    }
    take(l.ln2, 512, "ln2");
    take(l.ffnGate, 2304ULL * 512, "ffn_gate");
    take(l.ffnUp, 2304ULL * 512, "ffn_up");
    take(l.ffnDown, 512ULL * 2304, "ffn_down");
    take(l.post2, 512, "ffn_post_norm");
  }
  return w;
}

} // namespace richengine::ops
