// NariQ trellis-coded projections and embedding rows (nariq/FORMAT.md). The
// implementation covers the published Qwen3.8 subset: identity or normalized
// block Sylvester Hadamard side transforms, TRL2-TRL6.
#pragma clang fp reassociate(off)
#include "metal/abi/Trellis.h"

#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

constant constexpr uint kTrellisMaxBlock = 4096;
constant constexpr uint kTrellisMaxEmbedding = 5120;

inline float trellis_value(uint state) {
  const uint h = state * 0x83DCD12Du;
  return float(int((((h >> 2) & 0x3f3f3f3fu) * 0x01010101u) >> 24) - 126);
}

inline uint trellis_position(uint row, uint column) {
  return 8 * (4 * (row % 8) + (column % 8) / 2) + 4 * (row / 8) + 2 * (column / 8) + column % 2;
}

template <int bits>
inline float trellis_coded_value(device const uint *tile, uint row, uint column) {
  constexpr uint nw = 8 * bits, total = 256 * bits;
  const uint p = trellis_position(row, column);
  const uint start = ((p + 1) * bits + total - 16) % total;
  const uint word = start / 32, offset = start % 32;
  const uint hi = tile[word], lo = tile[(word + 1) % nw];
  const uint state = (offset <= 16 ? hi >> (16 - offset) : (hi << (offset - 16)) | (lo >> (48 - offset))) & 65535;
  return trellis_value(state);
}

kernel void trellis_input(device const bfloat *input [[buffer(0)]],
                          device const float *pre [[buffer(1)]],
                          device const int *pin [[buffer(2)]],
                          device float *output [[buffer(3)]],
                          constant TrellisParams &p [[buffer(4)]],
                          uint2 index [[thread_position_in_grid]]) {
  if (index.x >= p.input_size || index.y >= p.rows) return;
  const uint column = (p.flags & 1) ? uint(pin[index.x]) : index.x;
  output[ulong(index.y) * p.input_size + index.x] = float(input[ulong(index.y) * p.input_size + column]) * pre[index.x];
}

kernel void trellis_fwht(device float *values [[buffer(0)]],
                         constant TrellisTransformParams &p [[buffer(1)]],
                         uint2 group [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint2 threads [[threads_per_threadgroup]]) {
  threadgroup float blockValues[kTrellisMaxBlock];
  if (p.block > kTrellisMaxBlock) return;
  device float *row = values + ulong(group.y) * p.length + ulong(group.x) * p.block;
  for (uint i = tid; i < p.block; i += threads.x) blockValues[i] = row[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint h = 1; h < p.block; h *= 2) {
    for (uint j = tid; j < p.block / 2; j += threads.x) {
      const uint i = (j / h) * (2 * h) + j % h;
      const float x = blockValues[i], y = blockValues[i + h];
      blockValues[i] = x + y;
      blockValues[i + h] = x - y;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint i = tid; i < p.block; i += threads.x) row[i] = blockValues[i] * p.scale;
}


// Fused input transform: gather/scale and block Hadamard in one pass, so the
// activation plane is read and written once per tensor instead of twice.
kernel void trellis_input_fwht(device const bfloat *input [[buffer(0)]],
                               device const float *pre [[buffer(1)]],
                               device const int *pin [[buffer(2)]],
                               device float *output [[buffer(3)]],
                               constant TrellisParams &p [[buffer(4)]],
                               constant TrellisTransformParams &tp [[buffer(5)]],
                               uint2 group [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float blockValues[kTrellisMaxBlock];
  if (tp.block > kTrellisMaxBlock || group.y >= p.rows) return;
  const uint blockStart = group.x * tp.block;
  for (uint i = tid; i < tp.block; i += 256) {
    const uint k = blockStart + i;
    const uint column = (p.flags & 1) ? uint(pin[k]) : k;
    blockValues[i] = float(input[ulong(group.y) * p.input_size + column]) * pre[k];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint h = 1; h < tp.block; h *= 2) {
    for (uint j = tid; j < tp.block / 2; j += 256) {
      const uint i = (j / h) * (2 * h) + j % h;
      const float x = blockValues[i], y = blockValues[i + h];
      blockValues[i] = x + y;
      blockValues[i + h] = x - y;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint i = tid; i < tp.block; i += 256) output[ulong(group.y) * p.input_size + blockStart + i] = blockValues[i] * tp.scale;
}

// Fused inverse transform and output epilogue, valid where no output
// permutation applies: the inverse Hadamard runs in threadgroup memory and
// each position scales, scatters and converts without a y round-trip.
template <typename T>
inline void trellis_fwht_output_impl(device const float *y, device const float *post, device const int *scatter,
                                     device T *output, device const bfloat *epilogue, constant TrellisParams &p,
                                     constant TrellisTransformParams &tp, uint2 group, uint tid,
                                     threadgroup float *blockValues) {
  if (tp.block > kTrellisMaxBlock || group.y >= p.rows) return;
  const uint blockStart = group.x * tp.block;
  device const float *row = y + ulong(group.y) * p.output_size;
  for (uint i = tid; i < tp.block; i += 256) blockValues[i] = row[blockStart + i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint h = 1; h < tp.block; h *= 2) {
    for (uint j = tid; j < tp.block / 2; j += 256) {
      const uint i = (j / h) * (2 * h) + j % h;
      const float x = blockValues[i], yv = blockValues[i + h];
      blockValues[i] = x + yv;
      blockValues[i + h] = x - yv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint i = tid; i < tp.block; i += 256) {
    const uint index = blockStart + i;
    if (index >= p.output_size) break;
    const uint destination = (p.flags & 4) ? uint(scatter[p.out_offset + index]) : p.out_offset + index;
    float value = blockValues[i] * tp.scale * post[index];
    if (p.flags & 8) value += float(epilogue[ulong(group.y) * p.out_stride + destination]);
    if (p.flags & 16) {
      const float gate = float(epilogue[ulong(group.y) * p.out_stride + destination]);
      value *= gate / (1.0f + exp(-gate));
    }
    output[ulong(group.y) * p.out_stride + destination] = T(value);
  }
}

kernel void trellis_fwht_output_bf16(device const float *y [[buffer(0)]],
                                     device const float *post [[buffer(1)]],
                                     device const int *scatter [[buffer(2)]],
                                     device bfloat *output [[buffer(3)]],
                                     device const bfloat *epilogue [[buffer(4)]],
                                     constant TrellisParams &p [[buffer(5)]],
                                     constant TrellisTransformParams &tp [[buffer(6)]],
                                     uint2 group [[threadgroup_position_in_grid]],
                                     uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float blockValues[kTrellisMaxBlock];
  trellis_fwht_output_impl(y, post, scatter, output, epilogue, p, tp, group, tid, blockValues);
}

kernel void trellis_fwht_output_f32(device const float *y [[buffer(0)]],
                                    device const float *post [[buffer(1)]],
                                    device const int *scatter [[buffer(2)]],
                                    device float *output [[buffer(3)]],
                                    device const bfloat *epilogue [[buffer(4)]],
                                    constant TrellisParams &p [[buffer(5)]],
                                    constant TrellisTransformParams &tp [[buffer(6)]],
                                    uint2 group [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float blockValues[kTrellisMaxBlock];
  trellis_fwht_output_impl(y, post, scatter, output, epilogue, p, tp, group, tid, blockValues);
}

// The decode product of one output row per lane group. Each lane covers the
// K positions congruent to its lane index modulo 32; lane % 16 is therefore
// the lane's input column within every 16-column tile.
template <int bits, int columns>
kernel void trellis_product(device const uint *codes [[buffer(0)]],
                            device const half *scales [[buffer(1)]],
                            device const float *x [[buffer(2)]],
                            device float *y [[buffer(3)]],
                            constant TrellisParams &p [[buffer(4)]],
                            uint2 group [[threadgroup_position_in_grid]],
                            uint sg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
  const uint row = 4 * group.x + sg;
  const uint column = columns * group.y;
  const uint r = row % 16, c = lane % 16;
  const uint position = trellis_position(r, c);
  constexpr uint nw = 8 * bits, total = 256 * bits;
  const uint start = ((position + 1) * bits + total - 16) % total;
  const uint word = start / 32, offset = start % 32;
  float sum[columns] = {};
  for (uint k = lane; k < p.input_size; k += 32) {
    device const uint *tile = codes + (ulong(row / 16) * (p.input_size / 16) + k / 16) * nw;
    const uint hi = tile[word], lo = tile[(word + 1) % nw];
    const uint state = (offset <= 16 ? hi >> (16 - offset) : (hi << (offset - 16)) | (lo >> (48 - offset))) & 65535;
    const float w = trellis_value(state) * float(scales[k / 16]);
    for (int j = 0; j < columns; ++j)
      if (column + j < p.rows) sum[j] += w * x[ulong(column + j) * p.input_size + k];
  }
  for (int j = 0; j < columns; ++j) {
    const float value = simd_sum(sum[j]);
    if (lane == 0 && column + j < p.rows) y[ulong(column + j) * p.output_size + row] = value;
  }
}

#define TRELLIS_PRODUCT(B, C) \
template [[host_name("trellis_product_" #B "_" #C)]] \
kernel void trellis_product<B, C>(device const uint *, device const half *, device const float *, device float *, \
                                  constant TrellisParams &, uint2, uint, uint);
TRELLIS_PRODUCT(2, 1)
TRELLIS_PRODUCT(3, 1)
TRELLIS_PRODUCT(4, 1)
TRELLIS_PRODUCT(5, 1)
TRELLIS_PRODUCT(6, 1)
TRELLIS_PRODUCT(2, 4)
TRELLIS_PRODUCT(3, 4)
TRELLIS_PRODUCT(4, 4)
TRELLIS_PRODUCT(5, 4)
TRELLIS_PRODUCT(6, 4)
#undef TRELLIS_PRODUCT

// Eight consecutive states share one 64-bit window: the stream span of eight
// weights is 16 + 7 * bits <= 58 bits, so after one variable shift the
// per-weight extracts are compile-time constant shifts. Lane-invariant word
// indices hoist out of the tile loop; the third word is only read where the
// lane's window can reach it.
template <int bits>
struct TrellisDecode {
  uint i0, i1, i2, offset;
  bool needW2;
  inline void init(uint lane) {
    constexpr uint nw = 8 * bits, total = 256 * bits;
    const uint start = ((8 * lane + 1) * bits + total - 16) % total;
    const uint word = start / 32;
    offset = start % 32;
    i0 = word;
    i1 = (word + 1) % nw;
    i2 = (word + 2) % nw;
    needW2 = offset + 16 + 7 * bits > 64;
  }
  inline void decode(device const uint *tile, thread float *values) const {
    decodeWords(uint3(tile[i0], tile[i1], tile[i2]), values);
  }
  template <bool anchors32 = false>
  inline void decodeWords(uint3 w, thread float *values) const {
    if constexpr (anchors32) {
      static_assert(bits == 3);
      // TRL3 starts at offsets 3, 11, 19 or 27, so two words suffice.
      const uint first = (w.x << offset) | (w.y >> (32 - offset));
      const uint second = (first << (4 * bits)) | ((w.y << offset) >> (32 - 4 * bits));
#pragma unroll
      for (uint j = 0; j < 4; ++j) {
        values[j] = trellis_value((first >> (16 - j * bits)) & 65535u);
        values[4 + j] = trellis_value((second >> (16 - j * bits)) & 65535u);
      }
    } else {
      const ulong wide = (ulong(w.x) << 32) | ulong(w.y);
      ulong window;
      if (needW2) {
        window = offset ? (wide << offset) | (ulong(w.z) >> (32 - offset)) : wide;
      } else {
        window = offset ? wide << offset : wide;
      }
#pragma unroll
      for (uint j = 0; j < 8; ++j) values[j] = trellis_value(uint(window >> (64 - 16 - j * bits)) & 65535);
    }
  }
};

template <int bits>
inline void trellis_decode8(device const uint *tile, uint lane, thread float *values) {
  TrellisDecode<bits> state;
  state.init(lane);
  state.decode(tile, values);
}

// 16 output rows by 16 activation rows per threadgroup. This is the
// correctness-first prefill/verify tile; it keeps decoded weights and
// activations in threadgroup memory for simdgroup 8x8 products.
template <int bits>
kernel void trellis_product_mm(device const uint *codes [[buffer(0)]],
                               device const half *scales [[buffer(1)]],
                               device const float *x [[buffer(2)]],
                               device float *y [[buffer(3)]],
                               constant TrellisParams &p [[buffer(4)]],
                               uint2 group [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]],
                               uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr uint BK = 64;
  threadgroup float sw[16 * BK], sx[BK * 16], sy[256];
  const uint row0 = 16 * group.x, token0 = 16 * group.y;
  const uint rb = 8 * (sg % 2), tb = 8 * (sg / 2);
  simdgroup_matrix<float, 8, 8> acc(0.0f);
  constexpr uint nw = 8 * bits;
  const uint lane = tid % 32;
  TrellisDecode<bits> decodeState;
  decodeState.init(lane);
  for (uint first = 0; first < p.input_size; first += BK) {
    const uint tileK = first / 16 + sg;
    float v[8] = {};
    if (tileK < p.input_size / 16) {
      device const uint *tile = codes + (ulong(group.x) * (p.input_size / 16) + tileK) * nw;
      decodeState.decode(tile, v);
    }
    const float scale = tileK < p.input_size / 16 ? float(scales[tileK]) : 0;
#pragma unroll
    for (uint j = 0; j < 8; ++j) {
      const uint r = lane / 4 + 8 * (j / 4);
      const uint c = 2 * (lane % 4) + j % 2 + 8 * (j / 2 % 2);
      sw[r * BK + 16 * sg + c] = v[j] * scale;
    }
    for (uint q = tid; q < BK * 16; q += 128) {
      const uint k = first + q / 16, token = token0 + q % 16;
      sx[q] = token < p.rows && k < p.input_size ? x[ulong(token) * p.input_size + k] : 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (uint k = 0; k < BK; k += 8) {
      simdgroup_matrix<float, 8, 8> w, xv;
      simdgroup_load(w, sw + rb * BK + k, BK);
      simdgroup_load(xv, sx + k * 16 + tb, 16);
      simdgroup_multiply_accumulate(acc, w, xv, acc);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  simdgroup_store(acc, sy + rb * 16 + tb, 16);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint q = tid; q < 256; q += 128) {
    const uint token = token0 + q % 16;
    if (token < p.rows) y[ulong(token) * p.output_size + row0 + q / 16] = sy[q];
  }
}

#define TRELLIS_MM(B) \
template [[host_name("trellis_product_mm_" #B)]] \
kernel void trellis_product_mm<B>(device const uint *, device const half *, device const float *, device float *, \
                                  constant TrellisParams &, uint2, uint, uint);
TRELLIS_MM(2)
TRELLIS_MM(3)
TRELLIS_MM(4)
TRELLIS_MM(5)
TRELLIS_MM(6)
#undef TRELLIS_MM

// Verify-tile product for rows up to 8: 32 output rows by 8 tokens per
// threadgroup. The activation slab staged per K chunk is half the mm tile's,
// and twice the output height halves how many threadgroups re-read it.
kernel void trellis_zero(device float *y [[buffer(0)]],
                         constant TrellisParams &p [[buffer(1)]],
                         uint tid [[thread_position_in_grid]]) {
  if (tid < p.rows * p.output_size) y[tid] = 0;
}

template <int bits, bool anchors32 = false>
kernel void trellis_product_mm8(device const uint *codes [[buffer(0)]],
                                device const half *scales [[buffer(1)]],
                                device const float *x [[buffer(2)]],
                                device float *y [[buffer(3)]],
                                constant TrellisParams &p [[buffer(4)]],
                                uint3 group [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr uint BK = 64;
  threadgroup float sw[BK * 32], sy[256];
  const uint row0 = 32 * group.x, token0 = 8 * group.y;
  simdgroup_matrix<float, 8, 8> acc(0.0f);
  constexpr uint nw = 8 * bits;
  const uint lane = tid % 32;
  const uint rowTile = 2 * group.x + sg % 2;
  TrellisDecode<bits> decodeState;
  decodeState.init(lane);
  uint3 ahead0(0), ahead1(0);
  const uint tiles = p.input_size / 16;
  const uint kTilePair = 2 * (sg / 2);
  const bool fullK = tiles % (BK / 16) == 0;
  constexpr uint SPLIT = 4;
  const uint chunksTotal = (p.input_size + BK - 1) / BK;
  const uint per = (chunksTotal + SPLIT - 1) / SPLIT;
  const uint k0tile = group.z * per * (BK / 16);
  const uint firstTile = k0tile + kTilePair;
  if (rowTile < p.output_size / 16 && firstTile < tiles) {
    device const uint *tile0 = codes + (ulong(rowTile) * tiles + firstTile) * nw;
    ahead0 = uint3(tile0[decodeState.i0], tile0[decodeState.i1], tile0[decodeState.i2]);
    if (fullK || firstTile + 1 < tiles) {
      device const uint *tile1 = tile0 + nw;
      ahead1 = uint3(tile1[decodeState.i0], tile1[decodeState.i1], tile1[decodeState.i2]);
    }
  }
  const uint chunkEnd = min(chunksTotal, (group.z + 1) * per);
  for (uint chunk = group.z * per; chunk < chunkEnd; ++chunk) {
    const uint first = chunk * BK;
    // Eight weight tiles per chunk: row tiles sg % 2, K tiles first/16 + 2 *
    // (sg / 2) and one more, interleaved across the simdgroup pairs. The next
    // chunk's words load before the current chunk decodes.
    uint3 cur0 = ahead0, cur1 = ahead1;
    const uint nextTile = first / 16 + BK / 16 + kTilePair;
    if (chunk + 1 < chunkEnd && rowTile < p.output_size / 16 && (fullK || nextTile < tiles)) {
      device const uint *tile0 = codes + (ulong(rowTile) * tiles + nextTile) * nw;
      ahead0 = uint3(tile0[decodeState.i0], tile0[decodeState.i1], tile0[decodeState.i2]);
      if (fullK || nextTile + 1 < tiles) {
        device const uint *tile1 = tile0 + nw;
        ahead1 = uint3(tile1[decodeState.i0], tile1[decodeState.i1], tile1[decodeState.i2]);
      }
    }
#pragma unroll
    for (uint pass = 0; pass < 2; ++pass) {
      const uint kTileLocal = 2 * (sg / 2) + pass;
      const uint tileK = first / 16 + kTileLocal;
      float v[8] = {};
      if (tileK < tiles && rowTile < p.output_size / 16) decodeState.template decodeWords<anchors32>(pass ? cur1 : cur0, v);
      const float scale = tileK < tiles ? float(scales[tileK]) : 0;
#pragma unroll
      for (uint j = 0; j < 8; ++j) {
        const uint r = lane / 4 + 8 * (j / 4);
        const uint c = 2 * (lane % 4) + j % 2 + 8 * (j / 2 % 2);
        sw[(16 * kTileLocal + c) * 32 + 16 * (sg % 2) + r] = v[j] * scale;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // The token slab is eight rows: stage every k step's activation tile up
    // front so their loads overlap the decode work above instead of
    // serializing inside the accumulate chain.
    simdgroup_matrix<float, 8, 8> xv[BK / 8];
#pragma unroll
    for (uint k = 0; k < BK; k += 8)
      if (fullK || first + k < p.input_size) simdgroup_load(xv[k / 8], x + ulong(token0) * p.input_size + first + k, p.input_size);
#pragma unroll
    for (uint k = 0; k < BK; k += 8) {
      if (!fullK && first + k >= p.input_size) break;
      simdgroup_matrix<float, 8, 8> w;
      simdgroup_load(w, sw + k * 32 + 8 * sg, 32);
      simdgroup_multiply_accumulate(acc, xv[k / 8], w, acc);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  simdgroup_store(acc, sy + sg * 8, 32);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint q = tid; q < 256; q += 128) {
    const uint token = token0 + q / 32;
    if (token < p.rows && row0 + q % 32 < p.output_size)
      atomic_fetch_add_explicit((device atomic_float *)(y + ulong(token) * p.output_size + row0 + q % 32), sy[q], memory_order_relaxed);
  }
}

// Larger-batch product: 32 output rows by 16 tokens per threadgroup. The
// activation tile is the left operand, so its rows stream straight from
// device memory in the layout the input transform already writes; only the
// decoded weights pass through threadgroup memory, stored transposed.
template <int bits, int TT>
kernel void trellis_product_mm16x32(device const uint *codes [[buffer(0)]],
                                    device const half *scales [[buffer(1)]],
                                    device const float *x [[buffer(2)]],
                                    device float *y [[buffer(3)]],
                                    constant TrellisParams &p [[buffer(4)]],
                                    uint2 group [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]],
                                    uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr uint BK = 64;
  threadgroup float sw[BK * 32], sy[16 * 32];
  const uint row0 = 32 * group.x, token0 = TT * group.y;
  simdgroup_matrix<float, 8, 8> acc0[TT / 16], acc1[TT / 16];
#pragma unroll
  for (uint t = 0; t < TT / 16; ++t) acc0[t] = acc1[t] = simdgroup_matrix<float, 8, 8>(0.0f);
  constexpr uint nw = 8 * bits;
  const uint lane = tid % 32;
  const uint rowTile = 2 * group.x + sg % 2;
  TrellisDecode<bits> decodeState;
  decodeState.init(lane);
  uint3 ahead0(0), ahead1(0);
  const uint tiles = p.input_size / 16;
  const uint kTilePair = 2 * (sg / 2);
  const bool fullK = tiles % (BK / 16) == 0;
  if (rowTile < p.output_size / 16 && kTilePair < tiles) {
    device const uint *tile0 = codes + (ulong(rowTile) * tiles + kTilePair) * nw;
    ahead0 = uint3(tile0[decodeState.i0], tile0[decodeState.i1], tile0[decodeState.i2]);
    if (fullK || kTilePair + 1 < tiles) {
      device const uint *tile1 = tile0 + nw;
      ahead1 = uint3(tile1[decodeState.i0], tile1[decodeState.i1], tile1[decodeState.i2]);
    }
  }
  for (uint first = 0; first < p.input_size; first += BK) {
    uint3 cur0 = ahead0, cur1 = ahead1;
    const uint nextTile = first / 16 + BK / 16 + kTilePair;
    if (first + BK < p.input_size && rowTile < p.output_size / 16 && (fullK || nextTile < tiles)) {
      device const uint *tile0 = codes + (ulong(rowTile) * tiles + nextTile) * nw;
      ahead0 = uint3(tile0[decodeState.i0], tile0[decodeState.i1], tile0[decodeState.i2]);
      if (fullK || nextTile + 1 < tiles) {
        device const uint *tile1 = tile0 + nw;
        ahead1 = uint3(tile1[decodeState.i0], tile1[decodeState.i1], tile1[decodeState.i2]);
      }
    }
#pragma unroll
    for (uint pass = 0; pass < 2; ++pass) {
      const uint kTileLocal = 2 * (sg / 2) + pass;
      const uint tileK = first / 16 + kTileLocal;
      float v[8] = {};
      if (tileK < tiles && rowTile < p.output_size / 16) decodeState.decodeWords(pass ? cur1 : cur0, v);
      const float scale = tileK < tiles ? float(scales[tileK]) : 0;
#pragma unroll
      for (uint j = 0; j < 8; ++j) {
        const uint r = lane / 4 + 8 * (j / 4);
        const uint c = 2 * (lane % 4) + j % 2 + 8 * (j / 2 % 2);
        // transposed: k-major so the weight tile can be the right operand
        sw[(16 * kTileLocal + c) * 32 + 16 * (sg % 2) + r] = v[j] * scale;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // x rows beyond the activation count read finite scratch garbage; the
    // matching weights are zero past the K edge, so nothing they carry in
    // finite form reaches the output. acc lands transposed: token-major. All
    // activation tiles load ahead of the accumulate chain.
#pragma unroll
    for (uint k = 0; k < BK; k += 8) {
      if (!fullK && first + k >= p.input_size) break;
      simdgroup_matrix<float, 8, 8> w;
      simdgroup_load(w, sw + k * 32 + 8 * sg, 32);
#pragma unroll
      for (uint t = 0; t < TT / 16; ++t) {
        simdgroup_matrix<float, 8, 8> xv0, xv1;
        simdgroup_load(xv0, x + ulong(token0 + 16 * t) * p.input_size + first + k, p.input_size);
        simdgroup_load(xv1, x + ulong(token0 + 16 * t + 8) * p.input_size + first + k, p.input_size);
        simdgroup_multiply_accumulate(acc0[t], xv0, w, acc0[t]);
        simdgroup_multiply_accumulate(acc1[t], xv1, w, acc1[t]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
#pragma unroll
  for (uint t = 0; t < TT / 16; ++t) {
    simdgroup_store(acc0[t], sy + (16 * t) * 32 + sg * 8, 32);
    simdgroup_store(acc1[t], sy + (16 * t + 8) * 32 + sg * 8, 32);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint q = tid; q < 32 * TT; q += 128) {
    const uint token = token0 + q / 32;
    if (token < p.rows && row0 + q % 32 < p.output_size) y[ulong(token) * p.output_size + row0 + q % 32] = sy[q];
  }
}

#define TRELLIS_MM1632(B, T) \
template [[host_name("trellis_product_mm16x32_" #B "_" #T)]] \
kernel void trellis_product_mm16x32<B, T>(device const uint *, device const half *, device const float *, device float *, \
                                          constant TrellisParams &, uint2, uint, uint);
TRELLIS_MM1632(2, 16)
TRELLIS_MM1632(3, 16)
TRELLIS_MM1632(4, 16)
TRELLIS_MM1632(5, 16)
TRELLIS_MM1632(6, 16)
TRELLIS_MM1632(2, 32)
TRELLIS_MM1632(3, 32)
TRELLIS_MM1632(4, 32)
TRELLIS_MM1632(5, 32)
TRELLIS_MM1632(6, 32)
#undef TRELLIS_MM1632

#define TRELLIS_MM8(B) \
template [[host_name("trellis_product_mm8_" #B)]] \
kernel void trellis_product_mm8<B>(device const uint *, device const half *, device const float *, device float *, \
                                   constant TrellisParams &, uint3, uint, uint);
TRELLIS_MM8(2)
TRELLIS_MM8(3)
TRELLIS_MM8(4)
TRELLIS_MM8(5)
TRELLIS_MM8(6)
#undef TRELLIS_MM8

template [[host_name("trellis_product_mm8_anchors_3")]]
kernel void trellis_product_mm8<3, true>(device const uint *, device const half *, device const float *, device float *,
                                       constant TrellisParams &, uint3, uint, uint);

// Few-token product. One threadgroup covers a 16-row tile band: each lane
// batch-decodes 8 positions through trellis_decode8 and multiplies them into
// two float4 partial sums per token, partitioned over K in strides of four
// tiles. Decode work is shared across the token columns.
template <int bits, int columns>
kernel void trellis_product_mv(device const uint *codes [[buffer(0)]],
                               device const half *scales [[buffer(1)]],
                               device const float *x [[buffer(2)]],
                               device float *y [[buffer(3)]],
                               constant TrellisParams &p [[buffer(4)]],
                               uint group [[threadgroup_position_in_grid]],
                               uint sg [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
  const uint tileRow = group;
  const uint c = 2 * (lane % 4), r = 16 * tileRow + 2 * sg + (lane % 8) / 4;
  float4 sum0[columns], sum1[columns];
#pragma unroll
  for (int t = 0; t < columns; ++t) sum0[t] = sum1[t] = 0;
  TrellisDecode<bits> decodeState;
  decodeState.init(8 * sg + lane % 8);
  const uint tiles = p.input_size / 16;
  uint3 ahead(0);
  if (lane / 8 < tiles) {
    device const uint *tile = codes + (ulong(tileRow) * tiles + lane / 8) * (8 * bits);
    ahead = uint3(tile[decodeState.i0], tile[decodeState.i1], tile[decodeState.i2]);
  }
  for (uint k = lane / 8; k < tiles; k += 4) {
    const uint3 cur = ahead;
    const uint knext = k + 4;
    if (knext < tiles) {
      // issue the next tile's words before extracting the current one: the
      // memory latency overlaps the extraction arithmetic instead of
      // serializing ahead of it
      device const uint *tile = codes + (ulong(tileRow) * tiles + knext) * (8 * bits);
      ahead = uint3(tile[decodeState.i0], tile[decodeState.i1], tile[decodeState.i2]);
    }
    float v[8];
    decodeState.decodeWords(cur, v);
    const float s = float(scales[k]);
    const float4 w0 = float4(v[0], v[1], v[2], v[3]) * s;
    const float4 w1 = float4(v[4], v[5], v[6], v[7]) * s;
#pragma unroll
    for (int t = 0; t < columns; ++t) {
      const ulong base = ulong(min(uint(t), p.rows - 1)) * p.input_size + 16 * k + c;
      const float4 xv = float4(x[base], x[base + 1], x[base + 8], x[base + 9]);
      sum0[t] += w0 * xv;
      sum1[t] += w1 * xv;
    }
  }
#pragma unroll
  for (int t = 0; t < columns; ++t) {
    sum0[t] += simd_shuffle_down(sum0[t], 16);
    sum1[t] += simd_shuffle_down(sum1[t], 16);
    sum0[t] += simd_shuffle_down(sum0[t], 8);
    sum1[t] += simd_shuffle_down(sum1[t], 8);
    sum0[t] += simd_shuffle_down(sum0[t], 2);
    sum1[t] += simd_shuffle_down(sum1[t], 2);
    sum0[t] += simd_shuffle_down(sum0[t], 1);
    sum1[t] += simd_shuffle_down(sum1[t], 1);
    if (uint(t) < p.rows && lane < 8 && lane % 4 == 0) {
      y[ulong(t) * p.output_size + r] = sum0[t].x + sum0[t].y + sum0[t].z + sum0[t].w;
      y[ulong(t) * p.output_size + r + 8] = sum1[t].x + sum1[t].y + sum1[t].z + sum1[t].w;
    }
  }
}

#define TRELLIS_MV(B, C) \
template [[host_name("trellis_product_mv_" #B "_" #C)]] \
kernel void trellis_product_mv<B, C>(device const uint *, device const half *, device const float *, device float *, \
                                     constant TrellisParams &, uint, uint, uint);
TRELLIS_MV(2, 1)
TRELLIS_MV(3, 1)
TRELLIS_MV(4, 1)
TRELLIS_MV(5, 1)
TRELLIS_MV(6, 1)
TRELLIS_MV(2, 2)
TRELLIS_MV(3, 2)
TRELLIS_MV(4, 2)
TRELLIS_MV(5, 2)
TRELLIS_MV(6, 2)
TRELLIS_MV(2, 4)
TRELLIS_MV(3, 4)
TRELLIS_MV(4, 4)
TRELLIS_MV(5, 4)
TRELLIS_MV(6, 4)
TRELLIS_MV(2, 8)
TRELLIS_MV(3, 8)
TRELLIS_MV(4, 8)
TRELLIS_MV(5, 8)
TRELLIS_MV(6, 8)
#undef TRELLIS_MV

template <typename T>
inline void trellis_output_impl(device const float *y, device const float *post, device const int *pout,
                                device const int *scatter, device T *output, device const bfloat *epilogue,
                                constant TrellisParams &p, uint2 index) {
  if (index.x >= p.output_size || index.y >= p.rows) return;
  const uint source = (p.flags & 2) ? uint(pout[index.x]) : index.x;
  const uint local = index.x;
  const uint destination = (p.flags & 4) ? uint(scatter[p.out_offset + local]) : p.out_offset + local;
  float value = y[ulong(index.y) * p.output_size + source] * post[source];
  if (p.flags & 8) value += float(epilogue[ulong(index.y) * p.out_stride + destination]);
  if (p.flags & 16) {
    const float gate = float(epilogue[ulong(index.y) * p.out_stride + destination]);
    value *= gate / (1.0f + exp(-gate));
  }
  output[ulong(index.y) * p.out_stride + destination] = T(value);
}

kernel void trellis_output_bf16(device const float *y [[buffer(0)]],
                                device const float *post [[buffer(1)]],
                                device const int *pout [[buffer(2)]],
                                device const int *scatter [[buffer(3)]],
                                device bfloat *output [[buffer(4)]],
                                device const bfloat *epilogue [[buffer(5)]],
                                constant TrellisParams &p [[buffer(6)]],
                                uint2 index [[thread_position_in_grid]]) {
  trellis_output_impl(y, post, pout, scatter, output, epilogue, p, index);
}

kernel void trellis_output_f32(device const float *y [[buffer(0)]],
                               device const float *post [[buffer(1)]],
                               device const int *pout [[buffer(2)]],
                               device const int *scatter [[buffer(3)]],
                               device float *output [[buffer(4)]],
                               device const bfloat *epilogue [[buffer(5)]],
                               constant TrellisParams &p [[buffer(6)]],
                               uint2 index [[thread_position_in_grid]]) {
  trellis_output_impl(y, post, pout, scatter, output, epilogue, p, index);
}

// One threadgroup per gathered token. Decode the row into threadgroup memory,
// invert the input-side block transform there, scale it and write bf16.
template <int bits>
kernel void trellis_embedding(device const uint *tokens [[buffer(0)]],
                              device const uint *codes [[buffer(1)]],
                              device const half *scales [[buffer(2)]],
                              device const float *pre [[buffer(3)]],
                              device const float *post [[buffer(4)]],
                              device bfloat *output [[buffer(5)]],
                              constant TrellisEmbeddingParams &p [[buffer(6)]],
                              uint group [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]],
                              uint threads [[threads_per_threadgroup]]) {
  threadgroup float rowValues[kTrellisMaxEmbedding];
  if (group >= p.rows || p.input_size > kTrellisMaxEmbedding) return;
  const uint token = tokens[group];
  const uint tileRow = token / 16, rowInTile = token % 16;
  const uint columnTiles = p.input_size / 16;
  constexpr uint nw = 8 * bits;
  for (uint column = tid; column < p.input_size; column += threads) {
    const uint tileColumn = column / 16, inTile = column % 16;
    device const uint *tile = codes + (ulong(tileRow) * columnTiles + tileColumn) * nw;
    rowValues[column] = trellis_coded_value<bits>(tile, rowInTile, inTile) * float(scales[tileColumn]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (p.input_block > 1) {
    for (uint h = 1; h < p.input_block; h *= 2) {
      for (uint j = tid; j < p.input_size / 2; j += threads) {
        const uint within = j % (p.input_block / 2), block = j / (p.input_block / 2);
        const uint local = (within / h) * (2 * h) + within % h;
        const uint i = block * p.input_block + local;
        const float x = rowValues[i], y = rowValues[i + h];
        rowValues[i] = x + y;
        rowValues[i + h] = x - y;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = tid; i < p.input_size; i += threads) rowValues[i] *= p.input_scale;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float outputScale = post[token];
  for (uint column = tid; column < p.input_size; column += threads)
    output[ulong(group) * p.input_size + column] = bfloat(rowValues[column] * pre[column] * outputScale);
}

#define TRELLIS_EMBEDDING(B) \
template [[host_name("trellis_embedding_" #B)]] \
kernel void trellis_embedding<B>(device const uint *, device const uint *, device const half *, device const float *, \
                                 device const float *, device bfloat *, constant TrellisEmbeddingParams &, uint, uint, uint);
TRELLIS_EMBEDDING(2)
TRELLIS_EMBEDDING(3)
TRELLIS_EMBEDDING(4)
TRELLIS_EMBEDDING(5)
TRELLIS_EMBEDDING(6)
#undef TRELLIS_EMBEDDING
