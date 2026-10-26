#include "model/GgufTarget.hpp"
#include "model/GgufPreparation.hpp"

namespace richengine::model {

// The GGUF geometry of a target's layout. The GGUF schema the arch implies
// names the metadata keys and tensor schema the image planner reads:
// "llama" the dense target, "granite" Granite 4.2's llama tensors plus the
// fixed attention multiplier and the unit embedding/residual/logit scales,
// "lfm2"/"lfm2moe" the LFM2 targets — tied embeddings and the shared-
// expert-free sigmoid MoE with leading dense-FFN layers — and the hybrids
// the empty arch that resolves to "qwen35"/"qwen35moe" on the MoE split.
gguf::TargetGeometry ggufTargetGeometry(const TargetLayout &layout) {
  if (layout.packedOnly)
    throw std::invalid_argument("Gemma 4 targets load from packed files only");
  gguf::TargetGeometry geometry;
  geometry.layers = layout.layers;
  geometry.hiddenSize = layout.hiddenSize;
  geometry.vocabularySize = layout.vocabularySize;
  geometry.intermediateSize = layout.intermediateSize;
  geometry.gdnKeyHeads = layout.gdnKeyHeads;
  geometry.gdnValueHeads = layout.gdnValueHeads;
  geometry.gdnHeadDimension = layout.gdnHeadDimension;
  geometry.convolutionDimension = layout.convolutionDimension;
  // The conv-only mixers declare their taps; the GDN hybrids' geometry
  // keeps the zero that defaults to kGdnConvolutionTaps.
  if (!layout.gdnHeadDimension) geometry.convolutionTaps = layout.convolutionTaps;
  geometry.attentionWidth = layout.attentionWidth;
  geometry.attentionKvHeads = layout.attentionKvHeads;
  geometry.attentionHeadDimension = layout.attentionHeadDimension;
  geometry.rotaryPairs = layout.rotaryPairs;
  geometry.rotaryTheta = layout.rotaryTheta;
  geometry.fullAttentionPeriod = layout.fullAttentionPeriod;
  geometry.attentionMask = layout.attentionMask;
  if (layout.rmsEpsilon) geometry.rmsEpsilon = layout.rmsEpsilon;
  geometry.attentionScale = layout.attentionScale;
  // The packed query rows interleave a gate row each on the hybrids alone.
  geometry.attentionQueryGate = layout.attentionQueryStride == 2;
  geometry.attentionQkNorm = layout.attentionQkNorm;
  if (layout.ffnKind == FfnKind::SparseMoe) {
    geometry.experts = layout.experts;
    geometry.expertsPerToken = layout.expertsPerToken;
    geometry.expertIntermediateSize = layout.expertIntermediateSize;
    geometry.leadingDenseLayers = layout.denseLayers;
    geometry.sharedExpert = layout.moeSharedExpert;
  }
  geometry.arch = std::string(layout.ggufArch);
  // The LFM2 targets tie their output head to the token embedding.
  geometry.tiedOutput = layout.tiedOutput;
  return geometry;
}
std::filesystem::path findTargetGguf(const std::filesystem::path &directory) {
  std::filesystem::path found;
  std::error_code error;
  for (const auto &entry : std::filesystem::directory_iterator(directory, error)) {
    if (entry.path().extension() != ".gguf") continue;
    if (!found.empty()) throw GgufError("target directory holds more than one GGUF: " + directory.string());
    found = entry.path();
  }
  if (error) throw GgufError("cannot list target directory: " + directory.string());
  if (found.empty()) throw GgufError("target directory holds no GGUF: " + directory.string());
  return found;
}

GgufTargetLoader::GgufTargetLoader(metal::MetalBackend &backend, WeightImages &images,
                                   const std::filesystem::path &path, const gguf::TargetGeometry &geometry)
    : backend_(backend), images_(images), planned_(std::make_shared<Planned>(path)) {
  const GgufFile file(planned_->source);
  rotation_ = file.rotation();
  planned_->images = gguf::planImages(file, geometry);
}

WeightFile GgufTargetLoader::open(size_t index) {
  const gguf::Image &plan = planned_->images[index];
  return images_.load({"target/" + plan.name, plan.magic, plan.layer, plan.type, plan.bytes,
                       [&backend = backend_, planned = planned_, index](std::span<uint8_t>,
                                                                         const metal::MetalBuffer &buffer) {
                         writeGgufImage(backend, planned->source, buffer, planned->images[index]);
                         planned->source.checkUnchanged();
                       }});
}

WeightFile GgufTargetLoader::layer(uint32_t index) {
  if (index >= planned_->images.size() - 2) throw GgufError("target layer is out of range");
  return open(index);
}

WeightFile GgufTargetLoader::head() { return open(planned_->images.size() - 2); }

WeightFile GgufTargetLoader::embedding() { return open(planned_->images.size() - 1); }

} // namespace richengine::model
