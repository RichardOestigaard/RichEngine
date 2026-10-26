#include "MiniCpmV47.hpp"
#include "model/DFlashDraft.hpp"
#include "model/TargetLoader.hpp"
#include "model/TargetWeights.hpp"

#include <utility>

namespace richengine::model {
namespace {

// The same section structure as Qwen3_6Moe's layers: affine files keep a Q8
// router and shared-expert scalar gate and one Q4 slab per expert
// projection; the shared expert is a one-expert slab.
void readFfn(WeightFile &file, TargetLayerWeights &layer, const TargetLayout &layout,
             const AffineTargetFormat &) {
  const uint32_t hidden = layout.hiddenSize, width = layout.expertIntermediateSize;
  layer.ffn = ops::MoeWeights{ops::AffineMoeWeights{
      .router = readAffineQ8Projection(file, layout.experts, hidden, "router"),
      .expertGate = readAffineExpertProjection(file, layout.experts, width, hidden, "experts-gate"),
      .expertUp = readAffineExpertProjection(file, layout.experts, width, hidden, "experts-up"),
      .expertDown = readAffineExpertProjection(file, layout.experts, hidden, width, "experts-down"),
      .sharedGate = readAffineExpertProjection(file, 1, width, hidden, "shared-expert-gate"),
      .sharedUp = readAffineExpertProjection(file, 1, width, hidden, "shared-expert-up"),
      .sharedDown = readAffineExpertProjection(file, 1, hidden, width, "shared-expert-down"),
      .sharedScalarGate =
          readAffineQ8Projection(file, kQ4StorageN, hidden, "shared-expert-scalar-gate"),
  }};
}

// GGUF and oQ images keep the tensors as block segments in the same order,
// the router and the shared-expert scalar gate in F32.
template <class Format>
void readFfn(WeightFile &file, TargetLayerWeights &layer, const TargetLayout &,
             const Format &) {
  ops::BlockMoeWeights ffn;
  ffn.router = readQuantizedSegment(file, "router");
  ffn.gate.routed = readQuantizedSegment(file, "experts-gate");
  ffn.up.routed = readQuantizedSegment(file, "experts-up");
  ffn.down.routed = readQuantizedSegment(file, "experts-down");
  ffn.gate.shared = readQuantizedSegment(file, "shared-expert-gate");
  ffn.up.shared = readQuantizedSegment(file, "shared-expert-up");
  ffn.down.shared = readQuantizedSegment(file, "shared-expert-down");
  ffn.sharedScalarGate = readQuantizedSegment(file, "shared-expert-scalar-gate");
  layer.ffn = ops::MoeWeights{std::move(ffn)};
}

} // namespace

TargetModelWeights
loadMiniCpmV47MoeWeights(metal::MetalBackend &backend, const TargetLayout &layout,
                         const TargetFiles &files) {
  // The GDN/full-attention mixers are TargetLoader.hpp's default
  // readTargetMixer.
  const auto readMixer = [](WeightFile &file, const TargetLayout &l,
                            const auto &format, const MixerGeometry &g,
                            bool full) {
    return readTargetMixer(l, file, format, g, full);
  };
  return loadTargetWeights(
      backend, layout, files, readMixer,
      [&](WeightFile &file, TargetLayerWeights &layer, const auto &format) {
        readFfn(file, layer, layout, format);
      });
}

DFlashDraftLayout minicpmV47DraftLayout() {
  DFlashDraftLayout layout;
  layout.kind = DraftKind::DSpark;
  layout.layers = 5;
  layout.hiddenSize = 2048;
  layout.vocabularySize = 248144;
  layout.dynamicSize = 0;
  layout.qkvSize = 3072;
  layout.attentionSize = 2048;
  layout.intermediateSize = 6144;
  layout.attentionHeadDimension = 256;
  layout.rotaryTheta = 10'000'000.0F;
  layout.targetHiddenSize = 10240;
  layout.selectorRank = 0;
  layout.kvHeads = 2;
  layout.markovRank = 256;
  layout.blockSize = 15;
  layout.headCore = ops::DraftHeadCore::Q8K2D256;
  layout.contextCore = ops::DraftContextCore::Plain2048;
  return layout;
}

} // namespace richengine::model
