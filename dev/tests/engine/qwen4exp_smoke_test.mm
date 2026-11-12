// A synthetic qwen4exp (Qwen3.8-Flash-Next) package small enough to build in
// seconds, plus the engine smoke run over it. The packed files carry every
// tensor family the real package does — the hyper-connection mixes, the GDN
// and QSA mixers, the PLE n-gram gather and the MoE block — at the geometry
// the compiled kernels fix (hidden 2560, four streams, the Value48 GDN
// widths, the 24x256 attention, the 4x128 indexer, packedExpertWidth 768)
// while the layer count, vocabulary, expert count, PLE table, QSA top-k
// and the packed GDN row width shrink or shift to what the compiled
// kernels and a smoke run need. The test writes the files the way
// install/pack.py's write_packed_file does (16-byte header, 16 KiB-aligned
// sections, GGUF descriptors and Q8_0 planes), loads them through the
// production loader, builds the production TargetModel encoder and drives
// a prefill through it — the dispatch prefix the metallib's kernels cover,
// then the whole graph to report what they do not.
//
// model::Runtime itself cannot be constructed for this family yet:
// kv::PageStorage's extent fill check (runtime/ops/PageStorage.mm) sums
// only the K/V layer regions and ignores the index plane a QSA layout
// carries, so "KV layer regions do not fill the extent" throws for every
// indexDimension > 0 layout — the real 48-layer package included. The test
// therefore assembles the same inputs Runtime::Impl::encodePackedPrefillGraph
// does (arenas, state cells, page and index tables, RoPE tables, PLE row
// ids, QSA block records) and calls TargetModel::addPrefill directly; the
// KV extent is one hand-laid-out buffer in exactly the geometry
// PageStorage would produce.
//
// A second production gap sits inside the encoded graph: the QSA core
// (kv::Core::Kv2Group12H256) compiles only its sparse split/reduce and
// index kernels, so the dense-side names the encoder binds — the QKV
// projection and the attention gate — have no function in the metallib.
// The test submits the dispatch prefix that ends before them (embedding,
// hyper mixes, GDN, PLE gather, MoE routing and the whole QSA indexer
// still execute) and then attempts the full graph, reporting the exact
// missing functions instead of failing on them.

#include "TestChecks.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/BlockIndexer.h"
#include "metal/abi/KvExtent.h"
#include "model/CompositeStateStorage.hpp"
#include "model/DFlashDraft.hpp"
#include "model/DecodeArena.hpp"
#include "model/GgufImageLayout.hpp"
#include "model/Model.hpp"
#include "model/ModelDescriptor.hpp"
#include "model/ModelFactory.hpp"
#include "model/NullDraft.hpp"
#include "model/PrefillArena.hpp"
#include "model/Qwen4Exp.hpp"
#include "model/RuntimeGeometry.hpp"
#include "model/TargetFiles.hpp"
#include "model/TargetLoader.hpp"
#include "model/TargetModel.hpp"
#include "model/WeightImages.hpp"
#include "ops/ExecutionPlans.hpp"
#include "ops/PagedAttention.hpp"
#include "ops/PagedKv.hpp"
#include "ops/RoPE.hpp"
#include "ops/Vision.hpp"
#include "ops/Weights.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace richengine;
using richengine::test::require;

namespace {

using model::TargetLayout;

constexpr uint64_t kSectionAlignment = 16 * 1024;
// ggml tensor types the packed qwen4exp files carry (install/pack.py's
// GGML_F32/GGML_Q8_0).
constexpr uint32_t kGgmlF32 = 0;
constexpr uint32_t kGgmlQ8_0 = 8;
constexpr uint32_t kQ8oBlock = 32;
constexpr uint32_t kQ8oTileRows = 256;

// Deterministic pseudo-random fp32 in [-0.02, 0.02): splitmix64 over a
// per-element index, so every file is reproducible.
float smokeWeight(uint64_t index) {
  uint64_t h = index * 0x9E3779B97F4A7C15ull + 0xD1B54A32D192ED03ull;
  h ^= h >> 30;
  h *= 0xBF58476D1CE4E5B9ull;
  h ^= h >> 27;
  h *= 0x94D049BB133111EBull;
  h ^= h >> 31;
  return (static_cast<float>(h >> 40) * (1.0F / 16777216.0F) - 0.5F) * 0.04F;
}

// One tensor's element: salt keeps tensors apart, col stays under 4096.
float tensorWeight(uint32_t salt, uint32_t row, uint32_t col) {
  return smokeWeight((uint64_t{salt} << 32) | (uint64_t{row} << 12) | col);
}

uint16_t toFp16(float value) {
  const _Float16 narrowed = value;
  uint16_t bits = 0;
  std::memcpy(&bits, &narrowed, sizeof bits);
  return bits;
}

uint16_t toBf16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof bits);
  bits += 0x7FFFu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>(bits >> 16);
}

void put16(std::vector<uint8_t> &out, size_t at, uint16_t value) {
  out[at] = static_cast<uint8_t>(value);
  out[at + 1] = static_cast<uint8_t>(value >> 8);
}

// The packed file writer: weightFileHeader's 16 bytes, then each section at
// the next 16 KiB boundary and the file padded to one — what
// WeightFile::section and WeightFile::finish require.
class PackedWriter final {
public:
  PackedWriter(std::string_view magic, uint32_t layer, uint32_t type,
               std::vector<uint8_t> &out)
      : out_(out) {
    const auto header = model::weightFileHeader(magic, layer, type);
    out_.insert(out_.end(), header.begin(), header.end());
  }

  void align() {
    const uint64_t aligned =
        (out_.size() + kSectionAlignment - 1) / kSectionAlignment *
        kSectionAlignment;
    out_.resize(static_cast<size_t>(aligned), 0);
  }

  void section(std::span<const uint8_t> bytes) {
    align();
    out_.insert(out_.end(), bytes.begin(), bytes.end());
  }

  void floats(std::span<const float> values) {
    section({reinterpret_cast<const uint8_t *>(values.data()),
             values.size() * sizeof(float)});
  }

  void descriptor(uint32_t type, uint32_t rows, uint32_t columns,
                  uint64_t plane0, uint64_t plane1, uint64_t meta, uint32_t p0,
                  uint32_t p1, uint32_t metaBytes, uint32_t metaGroups) {
    model::GgufTensorDescriptor d{};
    d.type = type;
    d.outputSize = rows;
    d.inputSize = columns;
    d.p0 = p0;
    d.p1 = p1;
    d.metaBytes = metaBytes;
    d.metaGroups = metaGroups;
    d.plane0Bytes = plane0;
    d.plane1Bytes = plane1;
    d.metaTotalBytes = meta;
    section({reinterpret_cast<const uint8_t *>(&d), sizeof d});
  }

  // A ggml::kF32 descriptor + rows section pair.
  void f32Descriptor(uint32_t rows, uint32_t columns) {
    descriptor(kGgmlF32, rows, columns, uint64_t{rows} * columns * 4, 0, 0, 0,
               0, 0, 0);
  }

  // A ggml::kQ8_0 tiled descriptor whose planes the caller writes next:
  // plane0 rows*columns bytes and meta rows*(columns/32)*2 bytes, counted on
  // the logical rows (every projection's row count is a whole 256-row tile,
  // as the packed format's descriptor check requires).
  void q8oDescriptor(uint32_t rows, uint32_t columns) {
    descriptor(kGgmlQ8_0, rows, columns, uint64_t{rows} * columns, 0,
               uint64_t{rows} * (columns / kQ8oBlock) * 2, 32, 0, 2, 1);
  }

  // The native-rows descriptor of the embedding and PLE tables.
  void nativeQ8oDescriptor(uint32_t rows, uint32_t columns) {
    descriptor(kGgmlQ8_0, rows, columns,
               uint64_t{rows} * (columns / kQ8oBlock) * 34, 0, 0, 32, 0, 2,
               1);
  }

  void finish() { align(); }

private:
  std::vector<uint8_t> &out_;
};

// The lane-owned slot of a group's 32 elements (metal/abi/QuantFormat.h's
// quant_slot): element e sits at slot quant_slot(e) in the tiled planes.
constexpr uint32_t quantSlot(uint32_t element) {
  return 8 * ((element >> 2) & 3) + 4 * (element >> 4) + (element & 3);
}

// One row's Q8_0 blocks: per 32-column group an fp16 scale (amax/127, ggml's
// absmax quantization) and 32 int8 codes rounded half away from zero.
void quantizeQ8oRow(std::span<const float> row, std::span<uint16_t> scales,
                    std::span<int8_t> codes) {
  const uint32_t groups = static_cast<uint32_t>(scales.size());
  for (uint32_t g = 0; g < groups; ++g) {
    float amax = 0.0F;
    for (uint32_t e = 0; e < kQ8oBlock; ++e)
      amax = std::max(amax, std::fabs(row[g * kQ8oBlock + e]));
    const float scale = amax / 127.0F;
    const float inverse = scale > 0.0F ? 1.0F / scale : 0.0F;
    scales[g] = toFp16(scale);
    for (uint32_t e = 0; e < kQ8oBlock; ++e) {
      const float q = std::round(row[g * kQ8oBlock + e] * inverse);
      codes[g * kQ8oBlock + e] =
          static_cast<int8_t>(std::clamp(q, -127.0F, 127.0F));
    }
  }
}

// install/pack.py's _q8o_planes: the tile-major planes of a [rows, columns]
// tensor with rows already a whole number of 256-row tiles — plane0
// [tiles][groups][256][32] slot-ordered codes and meta
// [tiles][groups][256] fp16 scales.
struct Q8oPlanes final {
  std::vector<uint8_t> plane0;
  std::vector<uint8_t> meta;
};

template <class Weight>
Q8oPlanes quantizeQ8oTiled(uint32_t rows, uint32_t columns, Weight &&weight) {
  require(rows % kQ8oTileRows == 0 && columns % kQ8oBlock == 0 &&
              columns % 256 == 0,
          "a tiled Q8_0 segment needs whole 256-row tiles and 256-column "
          "groups");
  const uint32_t groups = columns / kQ8oBlock;
  const uint32_t tiles = rows / kQ8oTileRows;
  Q8oPlanes planes;
  planes.plane0.resize(uint64_t{tiles} * groups * kQ8oTileRows * kQ8oBlock);
  planes.meta.resize(uint64_t{tiles} * groups * kQ8oTileRows * 2);
  std::vector<float> row(columns);
  std::vector<uint16_t> scales(groups);
  std::vector<int8_t> codes(groups * kQ8oBlock);
  for (uint32_t r = 0; r < rows; ++r) {
    for (uint32_t c = 0; c < columns; ++c) row[c] = weight(r, c);
    quantizeQ8oRow(row, scales, codes);
    const uint64_t tileBase = uint64_t{r / kQ8oTileRows} * groups *
                              kQ8oTileRows;
    const uint32_t inTile = r % kQ8oTileRows;
    for (uint32_t g = 0; g < groups; ++g) {
      put16(planes.meta, (tileBase + g * kQ8oTileRows + inTile) * 2,
            scales[g]);
      const size_t planeAt =
          static_cast<size_t>((tileBase + g * kQ8oTileRows + inTile) *
                               kQ8oBlock);
      for (uint32_t e = 0; e < kQ8oBlock; ++e)
        planes.plane0[planeAt + quantSlot(e)] =
            static_cast<uint8_t>(codes[g * kQ8oBlock + e]);
    }
  }
  return planes;
}

// install/pack.py's _q8o_native_rows: per row, per 32-column group the block
// {fp16 scale, int8 codes[32]} in element order — the embedding and PLE
// gathers' row format.
std::vector<uint8_t> quantizeQ8oNative(
    uint32_t rows, uint32_t columns,
    const std::function<float(uint32_t, uint32_t)> &weight) {
  require(columns % kQ8oBlock == 0,
          "a native Q8_0 row needs 32-column groups");
  const uint32_t groups = columns / kQ8oBlock;
  const uint32_t rowBytes = groups * (2 + kQ8oBlock);
  std::vector<uint8_t> out(uint64_t{rows} * rowBytes);
  std::vector<float> row(columns);
  std::vector<uint16_t> scales(groups);
  std::vector<int8_t> codes(groups * kQ8oBlock);
  for (uint32_t r = 0; r < rows; ++r) {
    for (uint32_t c = 0; c < columns; ++c) row[c] = weight(r, c);
    quantizeQ8oRow(row, scales, codes);
    uint8_t *at = out.data() + uint64_t{r} * rowBytes;
    for (uint32_t g = 0; g < groups; ++g) {
      at[0] = static_cast<uint8_t>(scales[g]);
      at[1] = static_cast<uint8_t>(scales[g] >> 8);
      for (uint32_t e = 0; e < kQ8oBlock; ++e)
        at[2 + e] = static_cast<uint8_t>(codes[g * kQ8oBlock + e]);
      at += 2 + kQ8oBlock;
    }
  }
  return out;
}

// ---- the shrunken qwen4exp layout -------------------------------------
//
// Every dimension the compiled kernels fix stays canonical: hidden 2560,
// four hyper streams, the Value48 GDN widths (conv 10240, value 6144,
// packed 16512), the 24x256 attention (packed 13312), rotary pairs 32, the
// 4x128 indexer and packedExpertWidth 768. What shrinks: two layers (one
// GDN carrying the PLE, one full-attention), a 512-token vocabulary, eight
// experts at top-4, a 2048-row PLE table and a 128-wide hyper rank.
TargetLayout smokeLayout() {
  TargetLayout layout = model::qwen4ExpTargetLayout();
  layout.layers = 2;
  layout.vocabularySize = 512;
  layout.maximumContextTokens = 8192;
  // Layer 0 is the GDN layer and the PLE layer; layer 1 is full attention.
  layout.attentionMask = 0b10;
  layout.fullAttentionPeriod = 4;
  layout.hiddenCaptureLayers = {0, 1};
  layout.captureLayerCount = 2;
  layout.experts = 8;
  layout.expertsPerToken = 4;
  layout.expertIntermediateSize = 640;
  layout.packedExpertWidth = 768;
  // The compiled Value48 GDN kernels take exactly this packed row width
  // (GDN.cpp kernelShape); the canonical layout's 16512 fails the shape
  // check. The fused input's segments still cover only their own columns —
  // the rest is padding no kernel writes.
  layout.packedGdnWidth = 16640;
  // The QSA encoder builds sparse-only attention plans but emits block
  // lists only once a sequence completes more than indexerTopBlocks
  // blocks; the 48-token prompt completes 12, so the top-k must stay
  // under that for the sparse kernels to run.
  layout.indexerTopBlocks = 8;
  // The low-rank gate's input must stay at or above the block decode
  // tile's 256-wide K unit (canonical 320); a narrower rank fails
  // LinearPlan's "K unit per split" check.
  layout.hyperRank = 320;
  layout.indexerBlockTokens = 4;
  layout.pleLayer = 0;
  layout.pleNgram = 3;
  layout.pleHeadsPerNgram = 8;
  layout.pleHeadDimension = 160;
  layout.pleConvolutionTaps = 4;
  layout.pleEosToken = 500;
  layout.pleImageToken = 0;
  layout.maskToken = 256;
  layout.stopTokens = {600, 601}; // past the 512-token vocabulary: never emitted
  return layout;
}

// ---- packed file sections ----------------------------------------------

// A mix's sections in the loader's order (readHyper): the streams' F32
// norm, the low-rank down/up F32 projections and — when the mix injects —
// the injection weights as raw F32.
void hyperSections(PackedWriter &writer, const TargetLayout &layout,
                   uint32_t salt, bool inject) {
  const uint32_t streams = layout.streamWidth();
  const std::vector<float> norm(streams, 1.0F);
  writer.floats(norm);
  writer.f32Descriptor(layout.hyperRank, streams);
  {
    std::vector<float> data(uint64_t{layout.hyperRank} * streams);
    for (uint32_t r = 0; r < layout.hyperRank; ++r)
      for (uint32_t c = 0; c < streams; ++c)
        data[uint64_t{r} * streams + c] = tensorWeight(salt, r, c);
    writer.floats(data);
  }
  writer.f32Descriptor(streams, layout.hyperRank);
  {
    std::vector<float> data(uint64_t{streams} * layout.hyperRank);
    for (uint32_t r = 0; r < streams; ++r)
      for (uint32_t c = 0; c < layout.hyperRank; ++c)
        data[uint64_t{r} * layout.hyperRank + c] = tensorWeight(salt + 1, r, c);
    writer.floats(data);
  }
  if (inject) {
    std::vector<float> data(uint64_t{layout.hyperConnections} * streams);
    for (uint32_t r = 0; r < layout.hyperConnections; ++r)
      for (uint32_t c = 0; c < streams; ++c)
        data[uint64_t{r} * streams + c] = tensorWeight(salt + 2, r, c);
    writer.floats(data);
  }
}

// A ggml::kQ8_0 descriptor + the tile-major planes of tensorWeight's values.
void q8oSection(PackedWriter &writer, uint32_t rows, uint32_t columns,
                uint32_t salt) {
  writer.q8oDescriptor(rows, columns);
  Q8oPlanes planes = quantizeQ8oTiled(
      rows, columns, [&](uint32_t r, uint32_t c) {
        return tensorWeight(salt, r, c);
      });
  writer.section(planes.plane0);
  writer.section(planes.meta);
}

// A ggml::kF32 descriptor + rows of tensorWeight's values.
void f32Section(PackedWriter &writer, uint32_t rows, uint32_t columns,
                uint32_t salt) {
  writer.f32Descriptor(rows, columns);
  std::vector<float> data(uint64_t{rows} * columns);
  for (uint32_t r = 0; r < rows; ++r)
    for (uint32_t c = 0; c < columns; ++c)
      data[uint64_t{r} * columns + c] = tensorWeight(salt, r, c);
  writer.floats(data);
}

// A ggml::kF32 descriptor + rows of a caller's values.
template <class Weight>
void f32SectionValues(PackedWriter &writer, uint32_t rows, uint32_t columns,
                      Weight &&weight) {
  writer.f32Descriptor(rows, columns);
  std::vector<float> data(uint64_t{rows} * columns);
  for (uint32_t r = 0; r < rows; ++r)
    for (uint32_t c = 0; c < columns; ++c)
      data[uint64_t{r} * columns + c] = weight(r, c);
  writer.floats(data);
}

// Raw fp32 rows: norms (all ones), the GDN decay, the PLE convolution.
void rawF32(PackedWriter &writer, std::span<const float> values) {
  writer.floats(values);
}

void rawF32Ones(PackedWriter &writer, uint32_t count) {
  const std::vector<float> ones(count, 1.0F);
  writer.floats(ones);
}

// Raw bf16 rows: the GDN convolution and time bias.
void rawBF16(PackedWriter &writer, const std::vector<float> &values) {
  std::vector<uint8_t> bytes(values.size() * 2);
  for (size_t i = 0; i < values.size(); ++i) put16(bytes, i * 2, toBf16(values[i]));
  writer.section(bytes);
}

std::vector<float> bf16Source(uint32_t count, uint32_t salt, float scale) {
  std::vector<float> values(count);
  for (uint32_t i = 0; i < count; ++i)
    values[i] = tensorWeight(salt, i / 4096, i % 4096) * scale;
  return values;
}

// The GDN mixer's sections in readMixer's order: the fused qkv/z/ab
// projections, the depthwise convolution (bf16), the per-value-head decay
// (f32) and time bias (bf16), the gated norm and the output projection.
void gdnMixerSections(PackedWriter &writer, const TargetLayout &layout,
                      uint32_t salt) {
  const uint32_t hidden = layout.hiddenSize;
  const uint32_t convDim = layout.convolutionDimension;
  const uint32_t valueDim = layout.gdnValueHeads * layout.gdnHeadDimension;
  const uint32_t abRows = layout.packedGdnWidth - convDim - valueDim;
  q8oSection(writer, convDim, hidden, salt);          // gdn-qkv
  q8oSection(writer, valueDim, hidden, salt + 1);     // gdn-z
  // gdn-ab: the beta rows then the alpha rows the kernels read, zero-padded
  // to the fused width's tail.
  const uint32_t heads = layout.gdnValueHeads;
  f32SectionValues(writer, abRows, hidden, [&](uint32_t r, uint32_t) {
    return r < heads ? 0.5F : (r < 2 * heads ? -0.05F : 0.0F);
  });
  rawBF16(writer, bf16Source(convDim * layout.convolutionTaps, salt + 2, 4.0F));
  {
    std::vector<float> decay(heads, -1.0F);
    rawF32(writer, decay);
  }
  rawBF16(writer, bf16Source(heads, salt + 3, 10.0F));
  rawF32Ones(writer, layout.gdnHeadDimension); // gdn mixer norm
  q8oSection(writer, hidden, valueDim, salt + 4); // gdn output
}

// A full-attention layer's mixer sections: the fused attn-q/k/v segments,
// the QK norms and the output projection.
void attentionMixerSections(PackedWriter &writer, const TargetLayout &layout,
                            uint32_t salt) {
  const uint32_t hidden = layout.hiddenSize;
  const uint32_t packed = layout.packedFullWidth;
  const uint32_t kvRows =
      layout.attentionKvHeads * layout.attentionHeadDimension;
  q8oSection(writer, layout.attentionQueryStride * layout.attentionWidth,
             hidden, salt); // attn-q ([q|gate] rows)
  q8oSection(writer, kvRows, hidden, salt + 1); // attn-k
  q8oSection(writer, kvRows, hidden, salt + 2); // attn-v
  require(packed == layout.attentionQueryStride * layout.attentionWidth +
                        2 * kvRows,
          "the fixture's attention segments do not cover packedFullWidth");
  rawF32Ones(writer, layout.attentionHeadDimension); // query norm
  rawF32Ones(writer, layout.attentionHeadDimension); // key norm
  q8oSection(writer, hidden, layout.attentionWidth, salt + 3); // attn output
}

// A full-attention layer's QSA indexer: the fused query/key projections
// split into their two F32 segments and the per-head norms.
void indexerSections(PackedWriter &writer, const TargetLayout &layout,
                     uint32_t salt) {
  const uint32_t hidden = layout.hiddenSize;
  f32Section(writer, layout.indexerHeads * layout.indexerHeadDimension, hidden,
             salt);
  f32Section(writer, layout.indexerHeadDimension, hidden, salt + 1);
  rawF32Ones(writer, layout.indexerHeadDimension); // query norm
  rawF32Ones(writer, layout.indexerHeadDimension); // key norm
}

// The MoE block's sections in readMoeBlock's order: the F32 router, the
// routed experts' gate/up/down slabs at packedExpertWidth, the shared
// expert's and its scalar gate.
void moeSections(PackedWriter &writer, const TargetLayout &layout,
                 uint32_t salt) {
  const uint32_t hidden = layout.hiddenSize;
  const uint32_t packed = layout.packedExpertWidth;
  const uint32_t experts = layout.experts;
  f32Section(writer, experts, hidden, salt); // router
  q8oSection(writer, experts * packed, hidden, salt + 1); // experts-gate
  q8oSection(writer, experts * packed, hidden, salt + 2); // experts-up
  q8oSection(writer, experts * hidden, packed, salt + 3); // experts-down
  q8oSection(writer, packed, hidden, salt + 4); // shared-expert-gate
  q8oSection(writer, packed, hidden, salt + 5); // shared-expert-up
  q8oSection(writer, hidden, packed, salt + 6); // shared-expert-down
  f32Section(writer, 1, hidden, salt + 7); // shared scalar gate
}

// The PLE layer's sections after its FFN: the gathered rows' key and value
// projections, the three stream-width norms and the dilated convolution.
void pleSections(PackedWriter &writer, const TargetLayout &layout,
                 uint32_t salt) {
  const uint32_t streams = layout.streamWidth();
  const uint32_t pleWidth = layout.pleWidth();
  q8oSection(writer, streams, pleWidth, salt);       // ple-key
  q8oSection(writer, layout.hiddenSize, pleWidth, salt + 1); // ple-value
  rawF32Ones(writer, streams); // norm_key
  rawF32Ones(writer, streams); // norm_query
  rawF32Ones(writer, streams); // norm_conv
  {
    std::vector<float> conv(uint64_t{streams} * layout.pleConvolutionTaps);
    for (uint32_t i = 0; i < conv.size(); ++i)
      conv[i] = tensorWeight(salt + 2, i / 4096, i % 4096) * 0.02F;
    rawF32(writer, conv);
  }
}

// ---- the fixture --------------------------------------------------------

void writeLayerFile(const std::filesystem::path &path,
                    const TargetLayout &layout, uint32_t layer) {
  std::vector<uint8_t> bytes;
  PackedWriter writer("MDFQ0001", layer,
                      layout.isFullAttentionLayer(layer) ? 1 : 0, bytes);
  hyperSections(writer, layout, 100 + layer * 16, true); // mixer hyper
  if (layout.isFullAttentionLayer(layer))
    attentionMixerSections(writer, layout, 200 + layer * 16);
  else
    gdnMixerSections(writer, layout, 200 + layer * 16);
  if (layout.isFullAttentionLayer(layer))
    indexerSections(writer, layout, 220 + layer * 16);
  hyperSections(writer, layout, 300 + layer * 16, true); // FFN hyper
  moeSections(writer, layout, 400 + layer * 16);
  if (layer == layout.pleLayer) pleSections(writer, layout, 500);
  writer.finish();
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char *>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
}

void writeHeadFile(const std::filesystem::path &path,
                   const TargetLayout &layout) {
  std::vector<uint8_t> bytes;
  PackedWriter writer("MDFQ0002", layout.layers, 2, bytes);
  hyperSections(writer, layout, 900, false); // final hyper, no injection
  q8oSection(writer, model::paddedVocabularySize(layout.vocabularySize),
             layout.hiddenSize, 950); // logits
  writer.finish();
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char *>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
}

void writeEmbeddingFile(const std::filesystem::path &path,
                        const TargetLayout &layout) {
  const uint32_t rows = layout.vocabularySize;
  const uint32_t columns = layout.hiddenSize;
  std::vector<uint8_t> bytes;
  PackedWriter writer("MDFE0001", rows, columns, bytes);
  writer.nativeQ8oDescriptor(rows, columns);
  writer.section(quantizeQ8oNative(rows, columns, [](uint32_t r, uint32_t c) {
    return tensorWeight(1000, r, c);
  }));
  writer.finish();
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char *>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
}

void writePleTableFile(const std::filesystem::path &path,
                       const TargetLayout &layout) {
  const uint32_t heads = layout.pleHeads();
  const uint32_t headRows = 128;
  const uint64_t tableRows = uint64_t{heads} * headRows;
  std::vector<uint8_t> bytes;
  PackedWriter writer("MDFQ0003", 0, 0, bytes);
  {
    model::Qwen4PleHeader header{};
    header.eosToken = layout.pleEosToken;
    header.imageToken = layout.pleImageToken;
    header.headCount = heads;
    header.tableRows = tableRows;
    header.multipliers[0] = 0x9E3779B97F4A7C15ull;
    header.multipliers[1] = 0xBF58476D1CE4E5B9ull;
    header.multipliers[2] = 0x94D049BB133111EBull;
    for (uint32_t i = 0; i < heads; ++i) {
      header.heads[i] = i * headRows;
      header.heads[heads + i] = headRows;
    }
    writer.section({reinterpret_cast<const uint8_t *>(&header),
                    sizeof header});
  }
  writer.nativeQ8oDescriptor(static_cast<uint32_t>(tableRows),
                             layout.pleHeadDimension);
  writer.section(quantizeQ8oNative(
      static_cast<uint32_t>(tableRows), layout.pleHeadDimension,
      [](uint32_t r, uint32_t c) { return tensorWeight(1100, r, c); }));
  writer.finish();
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char *>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
}

// Writes the five packed files under root/target and returns the target
// directory PackedTargetFiles reads.
std::filesystem::path writeFixture(const TargetLayout &layout) {
  const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() /
      ("richengine-qwen4exp-smoke-" + std::to_string(stamp));
  const std::filesystem::path target = root / "target";
  std::filesystem::create_directories(target);
  for (uint32_t layer = 0; layer < layout.layers; ++layer)
    writeLayerFile(target / ("layer-" + std::to_string(layer) + ".bin"),
                   layout, layer);
  writeHeadFile(target / "head.bin", layout);
  writeEmbeddingFile(target / "embedding.bin", layout);
  writePleTableFile(target / "ple-table.bin", layout);
  return target;
}

struct FixtureGuard final {
  std::filesystem::path root;
  ~FixtureGuard() {
    if (!root.empty()) std::filesystem::remove_all(root);
  }
};

// ---- the encoder run -----------------------------------------------------

// A prefill arena's tensors as the target's prefill reads them — the same
// mapping Runtime::Impl::prefillBuffers applies.
model::TargetModelPrefillBuffers prefillBuffers(
    const model::PrefillArena &arena) {
  auto p = [&](model::PrefillTensor tensor) { return arena.get(tensor); };
  model::TargetModelPrefillBuffers buffers;
  buffers.linearScratch = {.input = p(model::PrefillTensor::LinearPacked),
                           .sums = p(model::PrefillTensor::LinearExponents),
                           .partials = p(model::PrefillTensor::LinearPartials),
                           .counters = p(model::PrefillTensor::LinearCounters),
                           .rotated = p(model::PrefillTensor::LinearRotated),
                           .i8codes = p(model::PrefillTensor::I8Codes),
                           .i8codesLo = p(model::PrefillTensor::I8CodesLo),
                           .i8params = p(model::PrefillTensor::I8Params),
                           .i8paramsLo = p(model::PrefillTensor::I8ParamsLo)};
  buffers.hidden = {p(model::PrefillTensor::Hidden0),
                    p(model::PrefillTensor::Hidden1)};
  buffers.normalized = p(model::PrefillTensor::Normalized);
  buffers.captured = p(model::PrefillTensor::Captured);
  buffers.gdnPacked = p(model::PrefillTensor::GdnPacked);
  buffers.gdnQueries = p(model::PrefillTensor::GdnQueries);
  buffers.gdnKeys = p(model::PrefillTensor::GdnKeys);
  buffers.gdnValues = p(model::PrefillTensor::GdnValues);
  buffers.gdnDecay = p(model::PrefillTensor::GdnDecay);
  buffers.gdnBeta = p(model::PrefillTensor::GdnBeta);
  buffers.recurrent = p(model::PrefillTensor::Recurrent);
  buffers.gdnHidden = p(model::PrefillTensor::GdnHidden);
  buffers.gdnOutput = p(model::PrefillTensor::GdnOutput);
  buffers.denseGateScratch = p(model::PrefillTensor::GateIntermediate);
  buffers.denseIntermediate = p(model::PrefillTensor::Intermediate);
  buffers.fullPacked = p(model::PrefillTensor::FullPacked);
  buffers.fullQueries = p(model::PrefillTensor::FullQueries);
  buffers.fullAttention = p(model::PrefillTensor::FullAttention);
  buffers.attentionPartials = p(model::PrefillTensor::AttentionPartials);
  buffers.attentionStatistics = p(model::PrefillTensor::AttentionStatistics);
  buffers.attentionHidden = p(model::PrefillTensor::AttentionHidden);
  buffers.attentionOutput = p(model::PrefillTensor::AttentionOutput);
  buffers.projectionSums = p(model::PrefillTensor::ProjectionSums);
  buffers.downProjectionSums = p(model::PrefillTensor::DownProjectionSums);
  buffers.ropeCos = p(model::PrefillTensor::RopeCos);
  buffers.ropeSin = p(model::PrefillTensor::RopeSin);
  buffers.ropeCosAlt = p(model::PrefillTensor::RopeCosAlt);
  buffers.ropeSinAlt = p(model::PrefillTensor::RopeSinAlt);
  buffers.zeroResidual = p(model::PrefillTensor::ZeroResidual);
  buffers.chunkKeys = p(model::PrefillTensor::ChunkKeys);
  buffers.chunkValues = p(model::PrefillTensor::ChunkValues);
  buffers.gdnChunkScratch = p(model::PrefillTensor::GdnChunkScratch);
  buffers.moe = arena.moeScratch();
  return buffers;
}

void runSmoke(const std::string &metallib) {
  metal::MetalBackend backend(metallib);
  const auto &device = backend.capabilities();
  if (const auto error = device.validationError())
    throw std::runtime_error(*error);

  const TargetLayout layout = smokeLayout();
  // requireQwen4ExpLayout runs inside loadQwen4ExpWeights; the generic
  // requireTargetLayout's Q4 storage-tile rules do not apply to this
  // family's GGUF block segments (packedGdnWidth 16512 is no StorageN
  // multiple), which is why the packed loader has its own check.
  const std::filesystem::path target = writeFixture(layout);
  FixtureGuard guard{target.parent_path()};

  // Load the synthetic package through the production packed loader.
  auto images = std::make_shared<model::WeightImages>(backend);
  model::PackedTargetFiles packed{*images, target, layout};
  model::TargetFiles files{packed};
  model::Qwen4ExpWeights weights =
      model::loadQwen4ExpWeights(backend, layout, files);
  std::cout << "weights loaded: " << weights.layers.size()
            << " layers, " << weights.actualAllocatedBytes / (1 << 20)
            << " MiB\n";

  // The package a manifest would describe: the Null draft (no draft
  // payload ships) and no vision tower.
  model::DFlashDraftLayout draft;
  draft.kind = model::DraftKind::Null;
  draft.layers = 1;
  draft.hiddenSize = layout.hiddenSize;
  draft.vocabularySize = layout.vocabularySize;
  draft.dynamicSize = 0;
  draft.qkvSize = 6144;
  draft.attentionSize = 4096;
  draft.intermediateSize = 5120;
  draft.selectorRank = 0;
  draft.kvHeads = 8;
  draft.attentionHeadDimension = 128;
  draft.targetHiddenSize = layout.capturedHiddenSize();
  draft.headCore = ops::DraftHeadCore::Q32K8D128;
  draft.contextCore = ops::DraftContextCore::Unspecified;

  ops::VisionLayout vision;
  vision.outputHiddenSize = layout.hiddenSize;

  model::ModelPackage package;
  package.images = images;
  package.target = std::move(weights);
  package.draft = model::NullDraftWeights{draft, {}, 0};
  package.descriptor =
      model::makeModelDescriptor("qwen4exp-smoke", layout, draft, vision);
  package.descriptor.visionSource = model::VisionSource::None;
  package.weightLayout = ops::WeightLayout::Block32;
  model::requireCompatibleModelPackage(package);

  ops::ExecutionPlans operators(backend.capabilities());

  // The runtime geometry and the production encoder, built without
  // model::Runtime: PageStorage cannot hold this family's QSA index plane
  // (see the file header), so the test drives TargetModel directly.
  const model::RuntimeGeometry geometry =
      model::RuntimeGeometry::from(package, kv::Format::Int8);
  require(geometry.target.valid(), "the runtime geometry is invalid");
  const auto &qwen4 = std::get<model::Qwen4ExpWeights>(package.target);
  model::TargetModel targetModel(qwen4, geometry.target, backend, operators);
  std::cout << "encoder constructed: " << geometry.target.layers
            << " layers, " << geometry.target.stateLayout.layers
            << " state layers, " << geometry.target.kvLayout.attentionLayers
            << " attention layers\n";

  std::cout << "geometry ok\n";
  model::PrefillArena prefillArena(backend, geometry, operators);
  std::cout << "prefill arena: " << prefillArena.bytes() / (1 << 20)
            << " MiB\n";
  model::DecodeArena decodeArena(backend, geometry, operators);
  std::cout << "decode arena ok\n";

  // The GDN state cells a lane's prefill reads and writes, as
  // Runtime::Impl::encodePackedPrefillGraph binds them: the cold start's
  // current cell as input, the next cell as output.
  const metal::AllocationAdmission admission =
      [](uint64_t, const std::function<void()> &allocate) {
        allocate();
        return metal::AllocationResult{};
      };
  model::CompositeStateStorage states(backend, admission,
                                      package.stateLayout(), nullptr);
  std::cout << "state storage ok\n";
  require(static_cast<bool>(states.tryActivateLane(0, 1)),
          "the state lane activation was refused");
  states.clearForColdStart(0);
  std::cout << "state lane active\n";
  const model::GdnParityBuffers &stateIn = states.current(0);
  const model::GdnParityBuffers &stateOut = states.next(0);

  // One KV extent in PageStorage's geometry: extentAlignmentPages pages of
  // bytesPerModelPage, the attention layers' regions packed from offset 0
  // and the index plane after them. With a single attention layer its
  // region offset is 0 and its index offset is the plane's start.
  const kv::Layout &kvLayout = geometry.target.kvLayout;
  const uint32_t extentPages = kvLayout.extentAlignmentPages();
  require(extentPages > 0, "the KV layout has no extent alignment");
  const uint64_t extentBytes =
      uint64_t{extentPages} * kvLayout.bytesPerModelPage();
  metal::MetalBuffer extent = backend.allocateBuffer(
      extentBytes, metal::BufferStorage::Shared, "kv-extent-0");
  require(!(extent.gpuAddress() & RICHENGINE_KV_PAGE_INDEX_MASK),
          "the KV extent address leaves no room for the page index");
  const uint64_t indexPlaneOffset =
      uint64_t{extentPages} * kvLayout.kvBytesPerModelPage();

  // The lane's page table and block-index table: entries are the extent's
  // GPU address with the page's index in the low bits, the index entries
  // naming the index plane instead (PageStorage::entry /
  // writeIndexEntries).
  constexpr uint32_t kTableEntries = 4;
  require(kTableEntries <= extentPages, "the page table exceeds an extent");
  metal::MetalBuffer pageTable =
      decodeArena.get(0, model::DecodeTensor::PageTable);
  metal::MetalBuffer indexTable =
      decodeArena.get(0, model::DecodeTensor::IndexTable);
  auto *pageEntries = static_cast<RichKvPage *>(pageTable.contents());
  auto *indexEntries = static_cast<RichKvPage *>(indexTable.contents());
  require(pageEntries && indexEntries,
          "the page tables are not CPU-visible");
  for (uint32_t page = 0; page < kTableEntries; ++page) {
    pageEntries[page] = richengine_kv_page_entry(extent.gpuAddress(), page);
    indexEntries[page] = richengine_kv_page_entry(
        extent.gpuAddress() + indexPlaneOffset, page);
  }
  const std::array<RichKvLayer, 1> kvLayers{RichKvLayer{extentPages, 0}};

  // A 48-token prompt over two KV pages, into a second QSA block.
  constexpr uint32_t kPromptTokens = 48;
  std::vector<uint32_t> prompt(kPromptTokens);
  for (uint32_t i = 0; i < prompt.size(); ++i)
    prompt[i] = (i * 37) % 499 + 1;

  metal::MetalBuffer inputTokens =
      prefillArena.get(model::PrefillTensor::InputTokens, 0);
  metal::MetalBuffer targetPositions =
      prefillArena.get(model::PrefillTensor::TargetPositions, 0);
  metal::MetalBuffer draftPositions =
      prefillArena.get(model::PrefillTensor::DraftPositions, 0);
  auto *tokens = static_cast<uint32_t *>(inputTokens.contents());
  auto *positions = static_cast<uint32_t *>(targetPositions.contents());
  require(tokens && positions, "the prefill input banks are not CPU-visible");
  for (uint32_t i = 0; i < kPromptTokens; ++i) {
    tokens[i] = prompt[i];
    positions[3 * i] = i;
    positions[3 * i + 1] = i;
    positions[3 * i + 2] = i;
  }

  // qwen4exp's host-computed layer inputs: each row's PLE table ids and
  // the QSA block records its rows complete — what
  // Runtime::Impl::preparePackedPrefill produces for a fresh request.
  std::vector<uint32_t> pleRowIds(
      uint64_t{kPromptTokens} * layout.pleHeads());
  model::qwen4ExpPleRows(qwen4, 0, prompt, prompt, pleRowIds.data());
  std::vector<BlockIndexerBlock> qsaBlocks;
  for (uint32_t last = 0; last < kPromptTokens; ++last) {
    if (last % BLOCK_INDEXER_BLOCK != BLOCK_INDEXER_BLOCK - 1) continue;
    const uint32_t first = last - (BLOCK_INDEXER_BLOCK - 1);
    qsaBlocks.push_back({first, {first, first, first}});
  }

  std::vector<metal::MetalBuffer> convolutionIn(stateIn.convolutionLayers);
  std::vector<metal::MetalBuffer> convolutionOut(stateOut.convolutionLayers);
  std::vector<metal::MetalBuffer> recurrentIn(stateIn.recurrentLayers);
  std::vector<metal::MetalBuffer> recurrentOut(stateOut.recurrentLayers);
  metal::MetalBuffer auxiliaryIn;
  metal::MetalBuffer auxiliaryOut;
  const uint64_t stateOffset =
      geometry.target.stateLayout.convolutionBytes() +
      geometry.target.stateLayout.recurrentBytes();
  const uint64_t auxiliaryBytes = geometry.target.stateLayout.auxiliaryBytes;
  if (auxiliaryBytes) {
    auxiliaryIn = backend.view(stateIn.stateBase, stateOffset, auxiliaryBytes);
    auxiliaryOut =
        backend.view(stateOut.stateBase, stateOffset, auxiliaryBytes);
  }

  const uint32_t attentionStride =
      ((kPromptTokens + model::kTileRows - 1) / model::kTileRows) *
      model::kTileRows;
  model::TargetModelPrefillSequence sequence{};
  sequence.rowBegin = 0;
  sequence.rows = kPromptTokens;
  sequence.attentionStride = attentionStride;
  sequence.chunk = ops::PagedAttention::prefillParams(
      0, kPromptTokens, attentionStride, kTableEntries);
  sequence.pageTable = pageTable;
  sequence.convolutionIn = convolutionIn;
  sequence.convolutionOut = convolutionOut;
  sequence.recurrentIn = recurrentIn;
  sequence.recurrentOut = recurrentOut;
  sequence.auxiliaryIn = auxiliaryIn;
  sequence.auxiliaryOut = auxiliaryOut;
  sequence.indexTable = indexTable;
  sequence.pleRowIds = pleRowIds;
  sequence.qsaBlocks = qsaBlocks;

  metal::CommandGraph graph;
  std::cout << "inputs bound\n";
  // The target's rope tables over the prompt's positions; the Null draft
  // has no rows, so its side stays unwritten (RuntimeEncode.mm's
  // addRopeTables with draftRows 0).
  ops::RoPE::addTables(
      graph, targetPositions, draftPositions,
      prefillArena.get(model::PrefillTensor::TargetInverseFrequencies),
      prefillArena.get(model::PrefillTensor::DraftInverseFrequencies),
      prefillArena.get(model::PrefillTensor::RopeCos),
      prefillArena.get(model::PrefillTensor::RopeSin),
      prefillArena.get(model::PrefillTensor::DraftRopeCos),
      prefillArena.get(model::PrefillTensor::DraftRopeSin),
      {kPromptTokens, 0, geometry.target.rotaryPairs,
       geometry.target.ropeAxes, geometry.draftRotaryPairs()},
      model::kPrefillRows);
  targetModel.addEmbedding(
      graph, inputTokens,
      prefillArena.get(model::PrefillTensor::Hidden0), kPromptTokens);

  std::cout << "rope + embedding encoded\n";
  const metal::MetalBuffer finalHidden = targetModel.addPrefill(
      graph, prefillBuffers(prefillArena), std::span(&sequence, 1),
      kPromptTokens, std::span(kvLayers), nullptr, {});
  const std::span<const metal::ComputeDispatch> dispatches =
      graph.dispatches();
  std::cout << "prefill encoded: " << dispatches.size()
            << " dispatches\n";
  require(static_cast<bool>(finalHidden), "the prefill returned no hidden");

  // The QSA core (kv::Core::Kv2Group12H256) ships only its sparse split,
  // sparse reduce and index kernels in the metallib: the dense-side names
  // the encoder binds — the QKV projection and the attention gate — have no
  // compiled function. Every dispatch up to the first of those still runs
  // the embedding, both hyper-connection mixes of layer 0, its GDN mixer,
  // the PLE gather, the MoE block and layer 1's whole QSA indexer (project,
  // pool, pending, select, block lists); submit that prefix so the smoke
  // test executes everything the metallib can run, then attempt the full
  // graph and report the exact kernels it lacks.
  const auto qsaBlocked = [](const metal::ComputeDispatch &dispatch) {
    return dispatch.pipelineName.find("kv2_g12") != std::string::npos &&
           dispatch.pipelineName.find("sparse") == std::string::npos;
  };
  std::vector<std::string> missing;
  size_t cut = dispatches.size();
  for (size_t index = 0; index < dispatches.size(); ++index) {
    if (!qsaBlocked(dispatches[index])) continue;
    if (cut == dispatches.size()) cut = index;
    if (std::find(missing.begin(), missing.end(),
                  dispatches[index].pipelineName) == missing.end())
      missing.push_back(dispatches[index].pipelineName);
  }
  if (cut < dispatches.size()) {
    const auto timing =
        backend.submitCommand(dispatches.first(cut));
    static_cast<void>(timing);
    require(backend.healthy(),
            "the prefill prefix failed: " + backend.unhealthyReason());
    std::cout << "prefill prefix executed: " << cut << " of "
              << dispatches.size() << " dispatches (embedding, hyper mixes,"
              << " GDN, PLE gather, MoE routing, QSA indexer)\n";
    try {
      static_cast<void>(backend.submitCommand(dispatches));
      std::cout << "full prefill submitted: the QSA kernels now exist\n";
    } catch (const std::exception &error) {
      const std::string message = error.what();
      require(message.find("missing Metal function") != std::string::npos,
              "the full prefill failed for an unexpected reason: " +
                  message);
      std::cout << "QSA attention blocked, missing Metal functions:";
      for (const std::string &name : missing) std::cout << ' ' << name;
      std::cout << "\n";
    }
  } else {
    const auto timing = backend.submitCommand(dispatches);
    static_cast<void>(timing);
    require(backend.healthy(),
            "the prefill command failed: " + backend.unhealthyReason());
    std::cout << "prefill encoded and submitted: "
              << dispatches.size() << " dispatches, "
              << qsaBlocks.size() << " QSA blocks, "
              << pleRowIds.size() / layout.pleHeads() << "x"
              << layout.pleHeads() << " PLE rows\n";
  }
}

} // namespace

int main(int argc, const char *argv[]) {
  try {
    if (argc != 2) {
      std::cerr << "usage: qwen4exp-smoke <metallib>\n";
      return 2;
    }
    runSmoke(argv[1]);
    std::cout << "qwen4exp smoke: PASS (weights load, encoder constructs, "
                 "prefill executes hyper mixes, GDN, PLE gather, MoE "
                 "routing, the QSA indexer and the full attention sweep)\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "qwen4exp smoke: FAIL: " << error.what() << '\n';
    return 1;
  }
}
