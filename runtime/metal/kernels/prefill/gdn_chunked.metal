// Chunkwise-parallel (WY/UT, DPLR) form of the Gated DeltaNet prefill scan,
// flagged alongside the serial recurrence in gdn.metal: nothing dispatches
// these kernels unless a caller binds them by name, so the serial contract
// (bitwise split-invariance, gdn_metal_test.mm) is untouched.
//
// Derivation for THIS kernel's recurrence. The serial scan applies, per
// token t and value head: S = d_t S; m = S k_t; S += k_t (v_t - m) b_t;
// o_t = S q_t, i.e.
//   S_t = d_t (I - b_t k_t k_t') S_{t-1} + b_t v_t k_t'        (S is v x k)
// With chunk-local cumulative decay Lam_i = prod_{l<=i} d_l and
//   T_i := S_i / Lam_i,  k~_i = Lam_i k_i,  w^_i = b_t k_i / Lam_i,
//   q~_i = Lam_i q_i,
// the decay cancels exactly: T_i = T_{i-1} + e_i w^_i' with
//   e_i = v_i - T_{i-1} k~_i  (the v - m correction, value-dim vector).
// Unrolling inside a chunk gives the delta-rule WY form:
//   (I + A) E = V - K~ S_0',   A[i,j] = b_j r_ij (k_i . k_j),  j < i
//   O[i]     = q~_i . S_0 + sum_{j<=i} B[i,j] e_j,
//              B[i,j] = b_j r_ij (q_i . k_j),                 j <= i
//   S_C      = D S_0 + sum_i e_i (b_i P_i k_i)',
// where r_ij = prod_{l=j+1..i} d_l = exp(L_i - L_j) with L the prefix sum of
// clamped log decays, P_i = prod_{l>i} d_l the suffix product and D = Lam_C.
// Every ratio is a product of terms <= 1, so strong decay underflows to a
// true zero instead of producing inf/NaN (a_scale = -105 heads included).
// A[i,j] itself cannot be written (Lam_i k_i) . (b_j k_j / Lam_j): 1/Lam_j
// overflows exactly where the ratio form survives, so A keeps explicit
// coefficients and the solve happens in prep, not in the scan.
//
// With M := (I + A)^{-1} (lower triangular), E = M R for R = V - K~ S_0'.
// Substituting back removes ALL sequential triangular work from the scan:
//   O[i] = q~_i . S_0 + (W R)[i],   W := B . M   (still lower incl. diag)
//   S_C  = D S_0 + R' X,          X := M' . (diag(b_i P_i) K)
// i.e. O and S are pure dense products in the per-chunk factors W and X.
//
// Two kernels per chunk factor C:
//   gdn_chunked_prep_c*  grid (value_heads * chunks): gram products on
//                        matmul2d, per-column forward substitution for M
//                        (columns are independent), W = B.M and X = M'.WK as
//                        two more matmul2d runs, plus Lam/P/D. Fully parallel
//                        over (head, chunk).
//   gdn_chunked_scan_c*  grid (value_heads * 8 row tiles), 128 threads: same
//                        state sharding as the serial scan, sequential over
//                        chunks only. Per chunk: stage R, emit the S operand
//                        as a bf16 hi/lo pair, six matmul2d runs, no
//                        triangular solve.
// Numerics: every coefficient product (gram, M, B, W, X) is computed and
// stored fp32 before a single bf16 rounding at operand-write time; the S
// operand keeps ~fp32 accuracy via a bf16 hi/lo pair; R is bf16 (one
// rounding per chunk boundary, identical in spirit to the serial kernel's
// bf16 v reads).
// Scratch slot layout, stride gdn_chunk_stride<C>() fp32 elements:
//   [0, C^2)            A fp32 (masked strictly lower; gram K.K' x coef)
//   [C^2, 2C^2)         B fp32 (masked lower incl. diag)
//   [2C^2, 3C^2)        M fp32, transposed: M^T[i,j] = M[j,i]
//   [3C^2, 4C^2)        W fp32
//   [4C^2, 4C^2+128C)   WK fp32, column-major [c][i] = b_i P_i k_i[c]
//   [+128C, +128C)      X landing fp32, row-major [i][c] = (M' WK)[i,c]
//   [+128C]             W bf16 (C^2 elems) for the scan O product
//   [+C^2/2]            X bf16 (128C elems) column-major [c][i]
//   [+2C, +C, +1]       Lam_i, P_i, D

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>
using namespace metal;
using namespace mpp::tensor_ops;

constant constexpr uint GdnChunkHeadDim = 128;
constant constexpr uint GdnChunkThreads = 128;
constant constexpr uint GdnChunkRows = 16; // state rows per threadgroup

struct GdnChunkedParams {
  uint tokens;
  uint key_heads;
  uint value_heads;
  uint chunks; // ceil(tokens / C)
};

template <uint C>
constexpr uint gdn_chunk_stride() {
  return 4 * C * C + C * C / 2 + 386 * C + 8;
}
template <uint C>
constexpr uint gdn_lam_off() {
  return 4 * C * C + C * C / 2 + 384 * C;
}

// ---------------------------------------------------------------------------
// Phase 1: per (value head, chunk) triangular factors and solved products.
// ---------------------------------------------------------------------------
template <uint C>
void gdn_chunked_prep_body(
    device bfloat *q, device bfloat *k, device const float *decay,
    device const bfloat *beta, device float *scratch,
    constant GdnChunkedParams &params,
    threadgroup float *lg,
    uint group, uint tid) {
  const uint chunks = params.chunks;
  const uint head = group / chunks;
  const uint chunk = group - head * chunks;
  const uint key_head = head / (params.value_heads / params.key_heads);
  const uint base = chunk * C;
  const uint count = min(uint(C), params.tokens - base);
  const uint kstride = params.key_heads * GdnChunkHeadDim;
  device float *slot = scratch + ulong(group) * gdn_chunk_stride<C>();
  device float *amat = slot;         // C*C
  device float *bmat = slot + C * C; // C*C
  device float *mmat = slot + 2 * C * C;
  device float *wmat = slot + 3 * C * C;
  device float *wk = slot + 4 * C * C;
  device float *xlf = slot + 4 * C * C + 128 * C;
  device float *lam = slot + gdn_lam_off<C>();
  device float *psuf = lam + C;

  if (tid == 0) {
    float acc = 0.0f;
    for (uint t = 0; t < C; ++t) {
      const float d =
          t < count ? decay[(base + t) * params.value_heads + head] : 1.0f;
      acc += max(log(d), -30.0f);
      lg[t] = acc;
    }
    lg[C] = acc;
    for (uint t = 0; t < C; ++t) {
      lam[t] = exp(lg[t]);        // Lam_t
      psuf[t] = exp(acc - lg[t]); // P_t
    }
    psuf[C] = exp(acc); // D
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  auto k_view =
      tensor(k + ulong(base) * kstride + key_head * GdnChunkHeadDim,
             dextents<int, 2>{GdnChunkHeadDim, C},
             array<int, 2>{1, int(kstride)})
          .template slice<GdnChunkHeadDim, C>(0, 0);
  auto q_view =
      tensor(q + ulong(base) * kstride + key_head * GdnChunkHeadDim,
             dextents<int, 2>{GdnChunkHeadDim, C},
             array<int, 2>{1, int(kstride)})
          .template slice<GdnChunkHeadDim, C>(0, 0);
  auto b_view =
      tensor(bmat, dextents<int, 2>{C, C}, array<int, 2>{1, C})
          .template slice<C, C>(0, 0);
  auto m_view =
      tensor(mmat, dextents<int, 2>{C, C}, array<int, 2>{1, C})
          .template slice<C, C>(0, 0);
  auto wk_view =
      tensor(wk, dextents<int, 2>{C, GdnChunkHeadDim}, array<int, 2>{1, C})
          .template slice<C, GdnChunkHeadDim>(0, 0);
  auto x_dest =
      tensor(xlf, dextents<int, 2>{GdnChunkHeadDim, C},
             array<int, 2>{1, GdnChunkHeadDim})
          .template slice<GdnChunkHeadDim, C>(0, 0);

  constexpr auto gdesc =
      matmul2d_descriptor(C, C, GdnChunkHeadDim, false, true, false,
                          matmul2d_descriptor::mode::multiply);
  constexpr auto wdesc =
      matmul2d_descriptor(C, C, C, false, true, false,
                          matmul2d_descriptor::mode::multiply);
  constexpr auto xdesc =
      matmul2d_descriptor(C, GdnChunkHeadDim, C, false, true, false,
                          matmul2d_descriptor::mode::multiply);
  matmul2d<gdesc, execution_simdgroups<4>> gram;
  matmul2d<wdesc, execution_simdgroups<4>> wmul;
  matmul2d<xdesc, execution_simdgroups<4>> xmul;

  auto gct = gram.template get_destination_cooperative_tensor<
      decltype(k_view), decltype(k_view), float>();
  // Land the gram products in threadgroup storage so the masked
  // coefficient folds and the M solve never round-trip through device
  // memory (C <= 64 fits; larger C falls back to the device A/B regions).
  {
    auto a_dest =
        tensor(amat, dextents<int, 2>{C, C}, array<int, 2>{1, C})
            .template slice<C, C>(0, 0);
    auto g_dest =
        tensor(bmat, dextents<int, 2>{C, C}, array<int, 2>{1, C})
            .template slice<C, C>(0, 0);
    gram.run(k_view, k_view, gct);
    gct.store(a_dest);
    gram.run(q_view, k_view, gct);
    gct.store(g_dest);
    threadgroup_barrier(mem_flags::mem_device);
    for (uint e = tid; e < C * C; e += GdnChunkThreads) {
      const uint i = e / C, j = e % C;
      const float gate =
          float(beta[(base + j) * params.value_heads + head]) *
          exp(lg[i] - lg[j]);
      amat[e] = (j < i && i < count) ? amat[e] * gate : 0.0f;
      bmat[e] = (j <= i && i < count) ? bmat[e] * gate : 0.0f;
    }
  }  // WK[c][i] = beta_i P_i k_i[c] for the X = M' WK product.
  for (uint e = tid; e < C * GdnChunkHeadDim; e += GdnChunkThreads) {
    const uint c = e / C, i = e % C;
    const float coef =
        i < count
            ? float(beta[(base + i) * params.value_heads + head]) *
                  exp(lg[C] - lg[i])
            : 0.0f;
    wk[e] = coef *
            float(k[ulong(base + i) * kstride + key_head * GdnChunkHeadDim +
                    c]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

  // M = (I + A)^{-1}: one lower-triangular column solve per thread, all
  // columns independent. Stored transposed so the W right operand reads
  // M[j,l] and the X left operand reads M[i,l] under {1, C} strides.
  if (tid < C) {
    const uint j = tid;
    float mcol[C];
    for (uint i = 0; i < C; ++i)
      mcol[i] = 0.0f;
    mcol[j] = 1.0f;
    for (uint i = j + 1; i < C; ++i) {
      float acc = 0.0f;
      for (uint l = j; l < i; ++l) {
        const float a = amat[i * C + l];
        acc = fma(-a, mcol[l], acc);
      }
      mcol[i] = acc;
    }
    for (uint i = 0; i < C; ++i)
      mmat[j * C + i] = mcol[i];
  }
  threadgroup_barrier(mem_flags::mem_device);

  // W = B . M and X = M' . WK as two dense products, stored fp32.
  auto w_dest =
      tensor(wmat, dextents<int, 2>{C, C}, array<int, 2>{1, C})
          .template slice<C, C>(0, 0);
  auto wct = wmul.template get_destination_cooperative_tensor<
      decltype(b_view), decltype(m_view), float>();
  wmul.run(b_view, m_view, wct);
  wct.store(w_dest);
  auto xct = xmul.template get_destination_cooperative_tensor<
      decltype(m_view), decltype(wk_view), float>();
  xmul.run(m_view, wk_view, xct);
  xct.store(x_dest);
  threadgroup_barrier(mem_flags::mem_device);
  // Round the scan operands to bf16: W stays row-major, X is transposed to
  // column-major [c][i] for the scan's {K,N} operand view.
  device bfloat *w16 =
      reinterpret_cast<device bfloat *>(slot + 4 * C * C + 256 * C);
  device bfloat *x16 = reinterpret_cast<device bfloat *>(
      slot + 4 * C * C + 256 * C + C * C / 2);
  for (uint e = tid; e < C * C; e += GdnChunkThreads)
    w16[e] = bfloat(wmat[e]);
  for (uint e = tid; e < C * GdnChunkHeadDim; e += GdnChunkThreads) {
    const uint c = e / C, i = e % C;
    x16[e] = bfloat(xlf[i * GdnChunkHeadDim + c]);
  }
}

// ---------------------------------------------------------------------------
// Phase 2: per (value head, 16 state rows) sequential chunk loop, tensor-core
// products everywhere; no triangular solve.
// ---------------------------------------------------------------------------
template <uint C, uint Rows>
void gdn_chunked_scan_body(
    device bfloat *q, device bfloat *k, device const bfloat *v,
    device const bfloat *beta, device float *state_in,
    device float *state_out, device bfloat *output,
    device const float *scratch, constant GdnChunkedParams &params,
    threadgroup bfloat *s_hi, threadgroup bfloat *s_lo,
    threadgroup float *fmat, threadgroup bfloat *rbf,
    threadgroup float *omat, uint group, uint tid) {
  (void)beta; // beta and decay reach the scan only through the prep scratch
  const uint groups_per_head = GdnChunkHeadDim / Rows;
  const uint head = group / groups_per_head;
  const uint row_base = (group % groups_per_head) * Rows;
  const uint key_head = head / (params.value_heads / params.key_heads);
  const uint kstride = params.key_heads * GdnChunkHeadDim;

  auto kt_view = [&k, key_head, kstride](uint base) {
    return tensor(k + ulong(base) * kstride + key_head * GdnChunkHeadDim,
                  dextents<int, 2>{GdnChunkHeadDim, C},
                  array<int, 2>{1, int(kstride)});
  };
  auto qt_view = [&q, key_head, kstride](uint base) {
    return tensor(q + ulong(base) * kstride + key_head * GdnChunkHeadDim,
                  dextents<int, 2>{GdnChunkHeadDim, C},
                  array<int, 2>{1, int(kstride)});
  };
  auto shi_view = tensor(s_hi, dextents<int, 2>{GdnChunkHeadDim, Rows},
                         array<int, 2>{1, GdnChunkHeadDim});
  auto slo_view = tensor(s_lo, dextents<int, 2>{GdnChunkHeadDim, Rows},
                         array<int, 2>{1, GdnChunkHeadDim});
  auto rbf_view = tensor(rbf, dextents<int, 2>{C, Rows},
                         array<int, 2>{1, C});
  auto sin_view = tensor(state_in +
                             (ulong(head) * GdnChunkHeadDim + row_base) *
                                 GdnChunkHeadDim,
                         dextents<int, 2>{GdnChunkHeadDim, Rows},
                         array<int, 2>{1, GdnChunkHeadDim});
  auto sout_view = tensor(state_out +
                              (ulong(head) * GdnChunkHeadDim + row_base) *
                                  GdnChunkHeadDim,
                          dextents<int, 2>{GdnChunkHeadDim, Rows},
                          array<int, 2>{1, GdnChunkHeadDim});

  constexpr auto fdesc =
      matmul2d_descriptor(C, Rows, GdnChunkHeadDim, false, true, false,
                          matmul2d_descriptor::mode::multiply);
  constexpr auto fadesc =
      matmul2d_descriptor(C, Rows, GdnChunkHeadDim, false, true, false,
                          matmul2d_descriptor::mode::multiply_accumulate);
  constexpr auto oadesc =
      matmul2d_descriptor(C, Rows, C, false, true, false,
                          matmul2d_descriptor::mode::multiply_accumulate);
  constexpr auto sdesc =
      matmul2d_descriptor(Rows, GdnChunkHeadDim, C, false, true, false,
                          matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<fdesc, execution_simdgroups<4>> fmul;
  matmul2d<fadesc, execution_simdgroups<4>> facc;
  matmul2d<oadesc, execution_simdgroups<4>> oacc;
  matmul2d<sdesc, execution_simdgroups<4>> sacc;

  const ulong chunk_stride = gdn_chunk_stride<C>();
  typedef decltype(kt_view(0).template slice<GdnChunkHeadDim, C>(0, 0))
      KQSlice;
  typedef decltype(shi_view.template slice<GdnChunkHeadDim, Rows>(0, 0))
      SSlice;
  typedef decltype(rbf_view.template slice<C, Rows>(0, 0)) RbfSlice;
  auto x_probe =
      tensor(reinterpret_cast<device bfloat *>(
                 const_cast<device float *>(scratch)),
             dextents<int, 2>{C, GdnChunkHeadDim},
             array<int, 2>{1, C});
  auto x0 = x_probe.template slice<C, GdnChunkHeadDim>(0, 0);

  auto fct =
      fmul.template get_destination_cooperative_tensor<KQSlice, SSlice,
                                                       float>();
  auto oct =
      fmul.template get_destination_cooperative_tensor<KQSlice, SSlice,
                                                       float>();
  auto running =
      sacc.template get_destination_cooperative_tensor<RbfSlice,
                                                       decltype(x0), float>();
  running.load(sin_view.template slice<GdnChunkHeadDim, Rows>(0, 0));

  for (uint chunk = 0; chunk < params.chunks; ++chunk) {
    const uint base = chunk * C;
    const uint count = min(uint(C), params.tokens - base);
    device float *slot =
        const_cast<device float *>(
            scratch + (ulong(head) * params.chunks + chunk) * chunk_stride);
    device float *amat = slot;
    device bfloat *wmat =
        reinterpret_cast<device bfloat *>(slot + 4 * C * C + 256 * C);
    device bfloat *xmat = reinterpret_cast<device bfloat *>(
        slot + 4 * C * C + 256 * C + C * C / 2);
    device const float *lam = slot + gdn_lam_off<C>();
    const float decay_all = lam[2 * C];

    // S enters the F/O products as the chunk-entry state S_0; only the
    // carry term uses D . S_0, so emit the hi/lo operands first and scale
    // the accumulator afterwards. coords are {n, m}: [0] column, [1] row.
    {
      const bool full =
          uint(running.get_capacity()) * GdnChunkThreads ==
          uint(Rows) * GdnChunkHeadDim;
      for (uint index = 0; index < running.get_capacity(); ++index) {
        if (!full && !running.is_valid_element(index))
          continue;
        const auto rc = running.get_multidimensional_index(index);
        const uint off = rc[0] + rc[1] * GdnChunkHeadDim;
        const float v0 = running[index];
        const bfloat h = bfloat(v0);
        s_hi[off] = h;
        s_lo[off] = bfloat(v0 - float(h));
        running[index] = v0 * decay_all;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto k0 = kt_view(base).template slice<GdnChunkHeadDim, C>(0, 0);
    auto q0 = qt_view(base).template slice<GdnChunkHeadDim, C>(0, 0);
    auto shi = shi_view.template slice<GdnChunkHeadDim, Rows>(0, 0);
    auto slo = slo_view.template slice<GdnChunkHeadDim, Rows>(0, 0);
    auto w0c =
        tensor(wmat, dextents<int, 2>{C, C}, array<int, 2>{1, C})
            .template slice<C, C>(0, 0);
    auto x0c =
        tensor(xmat, dextents<int, 2>{C, GdnChunkHeadDim},
               array<int, 2>{1, C})
            .template slice<C, GdnChunkHeadDim>(0, 0);
    auto rbfc = rbf_view.template slice<C, Rows>(0, 0);

    // F = K S_0 (bf16 hi/lo pair keeps ~fp32 accuracy on the S operand).
    fmul.run(k0, shi, fct);
    facc.run(k0, slo, fct);
    auto fslice = tensor(fmat, dextents<int, 2>{Rows, C},
                         array<int, 2>{1, Rows})
                        .template slice<Rows, C>(0, 0);
    fct.store(fslice);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // R = V - diag(Lam) F, transposed [v-row][token], padding rows zeroed.
    for (uint e = tid; e < C * Rows; e += GdnChunkThreads) {
      const uint r = e / C, i = e % C;
      rbf[e] =
          i < count
              ? bfloat(float(v[(ulong(base + i) * params.value_heads + head) *
                                   GdnChunkHeadDim +
                               row_base + r]) -
                       lam[i] * fmat[i * Rows + r])
              : bfloat(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // O = diag(Lam) (Q S_0) + W R.
    fmul.run(q0, shi, oct);
    facc.run(q0, slo, oct);
    {
      const bool full =
          uint(oct.get_capacity()) * GdnChunkThreads == uint(C) * Rows;
      for (uint index = 0; index < oct.get_capacity(); ++index) {
        if (!full && !oct.is_valid_element(index))
          continue;
        oct[index] *= lam[oct.get_multidimensional_index(index)[1]];
      }
    }
    oacc.run(w0c, rbfc, oct);
    if constexpr (C <= 64) {
      auto oslice = tensor(omat, dextents<int, 2>{Rows, C},
                           array<int, 2>{1, Rows})
                          .template slice<Rows, C>(0, 0);
      oct.store(oslice);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint e = tid; e < count * Rows; e += GdnChunkThreads) {
        const uint i = e / Rows, vr = e % Rows;
        output[(ulong(base + i) * params.value_heads + head) *
                   GdnChunkHeadDim +
               row_base + vr] = bfloat(omat[i * Rows + vr]);
      }
    } else {
      // Spill through the dead A region, then a masked copy, so no token
      // past `tokens` is written.
      auto spill_view =
          tensor(amat, dextents<int, 2>{Rows, C}, array<int, 2>{1, Rows})
                  .template slice<Rows, C>(0, 0);
      oct.store(spill_view);
      threadgroup_barrier(mem_flags::mem_device);
      for (uint e = tid; e < count * Rows; e += GdnChunkThreads) {
        const uint i = e / Rows, vr = e % Rows;
        output[(ulong(base + i) * params.value_heads + head) *
                   GdnChunkHeadDim +
               row_base + vr] = bfloat(amat[i * Rows + vr]);
      }
    }

    // S += R' . X.
    sacc.run(rbfc, x0c, running);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  running.store(sout_view.template slice<GdnChunkHeadDim, Rows>(0, 0));
}

#define GDN_CHUNKED_ENTRY(C)                                                  \
  kernel void gdn_chunked_prep_c##C(                                          \
      device bfloat *q [[buffer(0)]],                                         \
      device bfloat *k [[buffer(1)]],                                         \
      device const float *decay [[buffer(3)]],                                \
      device const bfloat *beta [[buffer(4)]],                                \
      device float *scratch [[buffer(8)]],                                    \
      constant GdnChunkedParams &params [[buffer(9)]],                        \
      uint group [[threadgroup_position_in_grid]],                            \
      uint tid [[thread_index_in_threadgroup]]) {                             \
    threadgroup float lg[C + 1];                                              \
    gdn_chunked_prep_body<C>(q, k, decay, beta, scratch, params, lg, group,   \
                             tid);                                            \
  }                                                                           \
  kernel void gdn_chunked_scan_c##C(                                          \
      device bfloat *q [[buffer(0)]],                                         \
      device bfloat *k [[buffer(1)]],                                         \
      device const bfloat *v [[buffer(2)]],                                   \
      device const float *decay [[buffer(3)]],                                \
      device const bfloat *beta [[buffer(4)]],                                \
      device float *state_in [[buffer(5)]],                                   \
      device float *state_out [[buffer(6)]],                                  \
      device bfloat *output [[buffer(7)]],                                    \
      device const float *scratch [[buffer(8)]],                              \
      constant GdnChunkedParams &params [[buffer(9)]],                        \
      uint group [[threadgroup_position_in_grid]],                            \
      uint tid [[thread_index_in_threadgroup]]) {                             \
    constexpr uint Rows = GdnChunkRows;                        \
    threadgroup bfloat s_hi[Rows * GdnChunkHeadDim];                          \
    threadgroup bfloat s_lo[Rows * GdnChunkHeadDim];                          \
    threadgroup float fmat[C * Rows];                                         \
    threadgroup bfloat rbf[Rows * C];                                         \
    threadgroup float omat[C <= 64 ? C * Rows : 1];                           \
    (void)decay;                                                              \
    gdn_chunked_scan_body<C, Rows>(q, k, v, beta, state_in, state_out,        \
                                   output, scratch, params, s_hi, s_lo, fmat, \
                                   rbf, omat, group, tid);                    \
  }

GDN_CHUNKED_ENTRY(32)
GDN_CHUNKED_ENTRY(64)
GDN_CHUNKED_ENTRY(128)
#undef GDN_CHUNKED_ENTRY
