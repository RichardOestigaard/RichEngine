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
  if (layout.packedOnly && layout.ggufArch != "maxwell-moe")
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
  geometry.globalAttentionMask = layout.globalAttentionMask;
  geometry.altRotaryTheta = layout.altRotaryTheta;
  geometry.slidingWindowTokens = layout.slidingWindowTokens;
  geometry.attentionGateRows = layout.attentionGateRows;
  geometry.sandwichNorm = layout.sandwichNorm;
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
  if (layout.ggufArch == "maxwell-moe") {
    // The Maxwell (maxwell-moe) extras: the MBSA indexer, the mHC modules
    // and HC head, and the MoE's selection bias beside its wider ungated
    // shared expert.
    geometry.indexerHeads = layout.indexerHeads;
    geometry.indexerHeadDimension = layout.indexerHeadDimension;
    geometry.indexerBlockTokens = layout.indexerBlockTokens;
    geometry.indexerPrefixBlocks = layout.indexerPrefixBlocks;
    geometry.indexerLocalBlocks = layout.indexerLocalBlocks;
    geometry.indexerRemoteBlocks = layout.indexerRemoteBlocks;
    geometry.mhcMixWidth = layout.mhcMixWidth;
    geometry.mhcStreams = layout.hyperConnections;
    geometry.mhcSinkhornIterations = layout.mhcSinkhornIterations;
    geometry.mhcHead = layout.mhcHead;
    // layout.moeSharedExpert is false because the shared expert runs
    // beside the routed block, but the GGUF carries its shexp tensors.
    geometry.sharedExpert = true;
    geometry.expertBias = true;
    geometry.sharedIntermediateSize = layout.sharedIntermediateSize;
    // The one MTP block the maxwell-moe GGUFs ship (nextn_predict_layers;
    // requireMaxwellMetadata accounts for it in block_count).
    geometry.mtpLayers = 1;
  }
  if (layout.ggufArch == "nemotron_h_moe") {
    // The nemotron extras: the MoE-only blocks' mask, the Mamba-2
    // geometry and the sigmoid router's per-expert bias beside its
    // ungated shared expert (expertBias carries both facts — the MoE
    // block routes the shared slot at unit weight).
    geometry.moeMask = layout.moeMask;
    geometry.mambaHeads = layout.mambaHeads;
    geometry.mambaHeadDimension = layout.mambaHeadDimension;
    geometry.mambaGroups = layout.mambaGroups;
    geometry.mambaStateSize = layout.mambaStateSize;
    geometry.mambaInnerSize = layout.mambaInnerSize;
    geometry.mambaTimeStepRank = layout.mambaTimeStepRank;
    geometry.mambaInputRows = layout.packedGdnWidth;
    geometry.expertBias = true;
    geometry.sharedIntermediateSize = layout.sharedIntermediateSize;
    // The sigmoid router's scaling factor and, Bolt's LatentMoE only, the
    // latent width the routed experts read and write (zero on Lightning,
    // whose experts span the hidden size).
    geometry.expertWeightsScale = layout.moeRoutedScale;
    geometry.moeLatentSize = layout.moeLatentSize;
  }
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
  planned_->geometry = geometry;
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

WeightFile GgufTargetLoader::mtp() {
  const GgufFile file(planned_->source);
  const gguf::Image plan = gguf::planMtpImage(file, planned_->geometry);
  return images_.load({"target/" + plan.name, plan.magic, plan.layer, plan.type, plan.bytes,
                       [&backend = backend_, planned = planned_, plan](std::span<uint8_t>,
                                                                       const metal::MetalBuffer &buffer) {
                         writeGgufImage(backend, planned->source, buffer, plan);
                         planned->source.checkUnchanged();
                       }});
}

} // namespace richengine::model
