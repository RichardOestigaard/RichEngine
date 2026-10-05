#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>

using namespace metal;
using namespace mpp::tensor_ops;

// Probe: which B-stride pattern does a device int4b_format operand honor?
// C[m][n] = sum_k A[k][m] * B'[...] with bf16 A and packed INT4 B.
// Shapes match the paged-attention tiles: qk is (M, N, K) = (48, 32, 256),
// pv is (48, 256, 32). For NT (transpose_right) B extents are {K, N} and
// element (k, n) sits at nibble k*bs0 + n*bs1. For NN B extents are {N, K}
// and element (n, k) sits at nibble n*bs0 + k*bs1.

template <int M, int N, int K, bool TR>
inline void probe_int4_matmul(device bfloat *a, device uchar *b,
                              device float *c, int bs0, int bs1) {
  typedef tensor<device int4b_format, dextents<int, 2>, tensor_inline> BT;
  auto at = tensor(a, dextents<int, 2>{K, M}, array<int, 2>{1, K});
  BT bt = TR ? BT(b, dextents<int, 2>{K, N}, array<int, 2>{bs0, bs1})
             : BT(b, dextents<int, 2>{N, K}, array<int, 2>{bs0, bs1});
  constexpr auto descriptor =
      matmul2d_descriptor(M, N, K, false, TR, false,
                          matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor, execution_simdgroups<8>> op;
  auto a0 = at.template slice<K, M>(0, 0);
  auto b0 = bt.template slice<N, K>(0, 0);
  auto dest = op.template get_destination_cooperative_tensor<decltype(a0),
                                                             decltype(b0),
                                                             float>();
  op.run(a0, b0, dest);
  auto ct = tensor(c, dextents<int, 2>{N, M}, array<int, 2>{1, N});
  dest.store(ct.template slice<N, M>(0, 0));
}

kernel void probe_int4_qk(device bfloat *a [[buffer(0)]],
                          device uchar *b [[buffer(1)]],
                          device float *c [[buffer(2)]],
                          constant int &bs0 [[buffer(3)]],
                          constant int &bs1 [[buffer(4)]],
                          uint thread_index [[thread_index_in_threadgroup]]) {
  (void)thread_index;
  probe_int4_matmul<48, 32, 256, true>(a, b, c, bs0, bs1);
}

// NN: B extents {N, K} = {256, 32}; element (d, t) at nibble d*bs0 + t*bs1.
kernel void probe_int4_pv_nn(device bfloat *a [[buffer(0)]],
                             device uchar *b [[buffer(1)]],
                             device float *c [[buffer(2)]],
                             constant int &bs0 [[buffer(3)]],
                             constant int &bs1 [[buffer(4)]],
                             uint thread_index [[thread_index_in_threadgroup]]) {
  (void)thread_index;
  probe_int4_matmul<48, 256, 32, false>(a, b, c, bs0, bs1);
}
