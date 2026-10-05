#include "model/GgufTarget.hpp"
#include "model/Dense.hpp"
#include "model/Granite.hpp"
#include "model/GgufPreparation.hpp"
#include "model/Lfm2.hpp"
#include "model/Lfm2Moe.hpp"

namespace richengine::model {

gguf::TargetGeometry ggufTargetGeometry(const DenseLayout &layout) {
  gguf::TargetGeometry geometry;
  geometry.arch = "llama";
  geometry.layers = layout.layers;
  geometry.hiddenSize = layout.hiddenSize;
  geometry.vocabularySize = layout.vocabularySize;
  geometry.intermediateSize = layout.intermediateSize;
  geometry.attentionWidth = layout.attentionWidth;
  geometry.attentionKvHeads = layout.attentionKvHeads;
  geometry.attentionHeadDimension = layout.attentionHeadDimension;
  geometry.rotaryPairs = layout.rotaryPairs;
  geometry.rotaryTheta = layout.rotaryTheta;
  geometry.fullAttentionPeriod = 1; // every layer attends
  geometry.attentionQueryGate = false;
  geometry.attentionQkNorm = false;
  return geometry;
}

// Granite 4.2 ("granite" arch): llama tensors plus the fixed attention
// multiplier and the unit embedding/residual/logit scales.
gguf::TargetGeometry ggufTargetGeometry(const GraniteLayout &layout) {
  gguf::TargetGeometry geometry;
  geometry.arch = "granite";
  geometry.layers = layout.layers;
  geometry.hiddenSize = layout.hiddenSize;
  geometry.vocabularySize = layout.vocabularySize;
  geometry.intermediateSize = layout.intermediateSize;
  geometry.attentionWidth = layout.attentionWidth;
  geometry.attentionKvHeads = layout.attentionKvHeads;
  geometry.attentionHeadDimension = layout.attentionHeadDimension;
  geometry.rotaryPairs = layout.rotaryPairs;
  geometry.rotaryTheta = layout.rotaryTheta;
  geometry.fullAttentionPeriod = 1;
  geometry.rmsEpsilon = layout.rmsEpsilon;
  geometry.attentionScale = layout.attentionScale;
  geometry.attentionQueryGate = false;
  geometry.attentionQkNorm = false;
  return geometry;
}

gguf::TargetGeometry ggufTargetGeometry(const Lfm2Layout &layout) {
  gguf::TargetGeometry geometry;
  geometry.arch = "lfm2";
  geometry.layers = layout.layers;
  geometry.hiddenSize = layout.hiddenSize;
  geometry.vocabularySize = layout.vocabularySize;
  geometry.intermediateSize = layout.intermediateSize;
  geometry.attentionMask = Lfm2Layout::attentionMask;
  geometry.convolutionDimension = layout.convolutionDimension;
  geometry.convolutionTaps = Lfm2Layout::convolutionTaps;
  geometry.rmsEpsilon = layout.rmsEpsilon;
  geometry.attentionWidth = layout.attentionWidth;
  geometry.attentionKvHeads = layout.attentionKvHeads;
  geometry.attentionHeadDimension = layout.attentionHeadDimension;
  geometry.rotaryPairs = layout.rotaryPairs;
  geometry.rotaryTheta = layout.rotaryTheta;
  geometry.attentionQueryGate = false;
  geometry.attentionQkNorm = true;
  geometry.tiedOutput = true;
  return geometry;
}

// The LFM2-MoE target ("lfm2moe"): LFM2's mixers and tied head, a dense FFN
// on the leading denseLayers and the shared-expert-free sigmoid MoE block
// on every later layer, conv or attention.
gguf::TargetGeometry ggufTargetGeometry(const Lfm2MoeLayout &layout) {
  gguf::TargetGeometry geometry;
  geometry.arch = "lfm2moe";
  geometry.layers = layout.layers;
  geometry.hiddenSize = layout.hiddenSize;
  geometry.vocabularySize = layout.vocabularySize;
  geometry.intermediateSize = layout.intermediateSize;
  geometry.attentionMask = Lfm2MoeLayout::attentionMask;
  geometry.convolutionDimension = layout.convolutionDimension;
  geometry.convolutionTaps = Lfm2MoeLayout::convolutionTaps;
  geometry.rmsEpsilon = layout.rmsEpsilon;
  geometry.attentionWidth = layout.attentionWidth;
  geometry.attentionKvHeads = layout.attentionKvHeads;
  geometry.attentionHeadDimension = layout.attentionHeadDimension;
  geometry.rotaryPairs = layout.rotaryPairs;
  geometry.rotaryTheta = layout.rotaryTheta;
  geometry.attentionQueryGate = false;
  geometry.attentionQkNorm = true;
  geometry.tiedOutput = true;
  geometry.experts = layout.experts;
  geometry.expertsPerToken = layout.expertsPerToken;
  geometry.expertIntermediateSize = layout.expertIntermediateSize;
  geometry.leadingDenseLayers = Lfm2MoeLayout::denseLayers;
  geometry.sharedExpert = false;
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
