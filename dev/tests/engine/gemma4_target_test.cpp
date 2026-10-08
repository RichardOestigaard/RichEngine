// The Gemma 4 26B-A4B host geometry: the dual local/global attention
// layouts, the per-layer KV binding, the padded MoE shape and the packed
// loader's projection sizing, all without a Metal device.

#include "TestChecks.hpp"
#include "model/Gemma4Moe.hpp"
#include "model/ModelFactory.hpp"
#include "model/TargetLoader.hpp"
#include "model/RuntimeArenas.hpp"
#include "ops/ExecutionPlans.hpp"
#include "ops/KernelNames.hpp"
#include "ops/PagedAttention.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

using namespace richengine;
using richengine::test::require;

constexpr model::Gemma4MoeLayout layout{};

void checkLayout() {
  // Five full-attention layers, every sixth; the rest are sliding local.
  uint64_t mask = 0;
  for (uint32_t layer = 0; layer < layout.layers; ++layer)
    if (layout.isGlobalAttentionLayer(layer)) mask |= uint64_t{1} << layer;
  require(mask == layout.globalLayerMask() &&
              layout.isGlobalAttentionLayer(5) &&
              layout.isGlobalAttentionLayer(29) &&
              !layout.isGlobalAttentionLayer(0) &&
              !layout.isGlobalAttentionLayer(4),
          "the global layer mask does not match every sixth layer");
  require(layout.packedWidthAt(0) == 8192 &&
              layout.packedWidthAt(5) == 9216 &&
              layout.maximumPackedWidth() == 9216 &&
              layout.attentionWidthAt(0) == 4096 &&
              layout.attentionWidthAt(5) == 8192,
          "the packed/attention widths diverge from 16x256/16x512+2x512");
  require(layout.headDimensionAt(5) == 512 && layout.rotaryPairsAt(0) == 128 &&
              layout.rotaryPairsAt(5) == 64 &&
              layout.hiddenCaptureLayers.size() == 6 &&
              layout.capturedHiddenSize() == 6 * 2816,
          "the captured hidden or rotary geometry is off");
  require(layout.packedExpertWidth == 768 &&
              layout.expertIntermediateSize == 704,
          "the routed experts' physical width is not the padded 768");
}

void checkKvLayouts() {
  const kv::Layout pool = layout.kvLayout();
  require(pool.valid() && pool.attentionLayers == 30 &&
              pool.altLayerMask == layout.globalLayerMask() &&
              pool.altKvHeads == 2 && pool.altHeadDimension == 512 &&
              pool.kvHeads == 8 && pool.headDimension == 256,
          "the pool layout lost its alternate geometry");
  require(!pool.isAltLayer(4) && pool.isAltLayer(5),
          "the alternate mask does not mark the global layers");
  const kv::Layout local = pool.layerLayout(0);
  const kv::Layout global = pool.layerLayout(5);
  require(local.kvHeads == 8 && local.headDimension == 256,
          "the sliding layers' per-layer layout is off");
  require(global.kvHeads == 2 && global.headDimension == 512,
          "a global layer does not hold its 2x512 page geometry");
  // The global layers' pages carry their own (smaller) geometry.
  kv::Layout expected = pool;
  const uint64_t want = 25 * pool.bytesPerLayerPageAt(0) +
                        5 * pool.bytesPerLayerPageAt(5);
  require(pool.bytesPerModelPage() == want &&
              pool.bytesPerLayerPageAt(5) < pool.bytesPerLayerPageAt(0),
          "the dual-geometry pool's page bytes do not follow the regions");
  static_cast<void>(expected);
}

model::Gemma4MoeWeights fakeWeights() {
  const auto projection = [](uint32_t n, uint32_t k) {
    return ops::Projection(n, k, ops::AffineWeights{});
  };
  model::Gemma4MoeWeights weights;
  weights.finalNorm = {};
  weights.logitsProjection =
      projection(layout.vocabularySize, layout.hiddenSize);
  weights.layers.resize(layout.layers);
  for (uint32_t layer = 0; layer < layout.layers; ++layer) {
    auto &w = weights.layers[layer];
    model::AttentionMixerWeights mixer;
    mixer.inputProjection =
        projection(layout.packedWidthAt(layer), layout.hiddenSize);
    mixer.outputProjection =
        projection(layout.hiddenSize, layout.attentionWidthAt(layer));
    w.mixer = mixer;
    // The padded expert slabs: output/input carry the physical 768.
    w.expertGate = {{}, layout.experts, layout.packedExpertWidth,
                    layout.hiddenSize, 0};
    w.expertUp = w.expertGate;
    w.expertDown = {{}, layout.experts, layout.hiddenSize,
                    layout.packedExpertWidth, 0};
    w.sharedGate = projection(layout.packedSharedWidth, layout.hiddenSize);
    w.sharedUp = w.sharedGate;
    w.sharedDown = projection(layout.hiddenSize, layout.packedSharedWidth);
  }
  return weights;
}

void checkGeometry() {
  const model::Gemma4MoeWeights weights = fakeWeights();
  const model::TargetModelGeometry geometry = model::targetModelGeometry(weights);
  require(geometry.gemmaMoe && geometry.altAttentionMask == layout.globalLayerMask() &&
              geometry.attentionHeadDimension == 256 &&
              geometry.altHeadDimension == 512 &&
              geometry.altRotaryPairs == 64 && geometry.rotaryPairs == 128,
          "the target geometry lost Gemma's alternate attention");
  require(geometry.logitSoftcap == 30.0F &&
              std::abs(geometry.embeddingScale - std::sqrt(2816.0F)) < 1e-3F,
          "the softcap or embedding scale is missing");
  const ops::MoeShape moe = geometry.moeShape();
  require(moe.valid() && moe.hiddenSize == 2816 && moe.experts == 128 &&
              moe.expertsPerToken == 8 && moe.expertIntermediateSize == 768 &&
              moe.routesPerToken() == 8 && !moe.sharedExpert,
          "the padded MoE shape does not validate");
  const kv::Layout local = geometry.layerKvLayout(false);
  const kv::Layout global = geometry.layerKvLayout(true);
  require(local.windowTokens == 1024 && local.kvHeads == 8 &&
              global.windowTokens == 0 && global.kvHeads == 2,
          "the per-layer KV layouts lost their window or geometry");
  const auto hasShape = [](const auto &shapes, ops::ProjectionShape want) {
    return std::find(shapes.begin(), shapes.end(), want) != shapes.end();
  };
  require(hasShape(geometry.prefillProjections, {8192, 2816}) &&
              hasShape(geometry.prefillProjections, {9216, 2816}),
          "the packed QKV projections of both widths are not prefill-planned");
  const auto &first = weights.layers[0];
  require(first.expertGate.outputSize == layout.packedExpertWidth &&
              first.expertDown.inputSize == layout.packedExpertWidth &&
              first.expertGate.experts == 128 &&
              first.sharedGate.layout() == ops::WeightLayout::Affine64,
          "the expert slabs or shared expert lost their layout kind");
}

void checkAttentionPlans() {
  DeviceCapabilities device;
  device.appleGpuFamily = 10;
  device.gpuCoreCount = 20;
  const ops::ExecutionPlans plans(device);
  const model::TargetModelGeometry geometry =
      model::targetModelGeometry(fakeWeights());
  const kv::Layout local = geometry.layerKvLayout(false);
  const kv::Layout global = geometry.layerKvLayout(true);
  require(local.windowTokens == 1024 && local.scoreScale == 1.0F &&
              global.windowTokens == 0 && global.kvHeads == 2,
          "the layer layouts lost the window or global geometry");
  for (const kv::Format format : {kv::Format::Int8, kv::Format::BFloat16}) {
    kv::Layout l = local, g = global;
    l.format = format;
    g.format = format;
    const ops::PrefillAttentionPlan prefillLocal = plans.prefillAttention(64, 16, l);
    const ops::PrefillAttentionPlan prefillGlobal = plans.prefillAttention(64, 16, g);
    require(prefillLocal.splitPipeline.find("_swa_h256") != std::string::npos &&
                prefillGlobal.splitPipeline.find("_hd512") != std::string::npos,
                "the prefill splits are not the sliding 256 and 512 kernels");
    require(prefillLocal.windowTokens == 1024 &&
                prefillGlobal.windowTokens == 0 &&
                prefillGlobal.scoreScale == 1.0F,
                "the window or global scale did not reach the plan");
    const ops::PrefillAttentionPlan canvasLocal =
        ops::PagedAttention::prefillCanvasPlan(64, 16, l);
    const ops::PrefillAttentionPlan canvasGlobal =
        ops::PagedAttention::prefillCanvasPlan(64, 16, g);
    require(canvasLocal.splitPipeline.find("_split_canvas_swa_h256") !=
                    std::string::npos &&
                canvasGlobal.splitPipeline.find("_split_canvas_hd512") !=
                    std::string::npos &&
                canvasLocal.reducePipeline ==
                    ops::kPrefillAttentionReduceCanvasGemmaH256 &&
                canvasGlobal.reducePipeline ==
                    ops::kPrefillAttentionReduceCanvasGemmaHd512,
            "canvas reduction kernels lost full-canvas visibility");
    const std::array<uint32_t, 2> histories{40, 40};
    const ops::VerifyAttentionPlan verifyLocal = plans.verifyAttention(2, 16, l, histories);
    const ops::VerifyAttentionPlan verifyGlobal = plans.verifyAttention(2, 16, g, histories);
    require(verifyLocal.splitPipeline.find("_split_swa_h256") != std::string::npos &&
                verifyGlobal.splitPipeline.find("_split_gemma_hd512") != std::string::npos,
                "the verify splits are not the sliding 256 and global 512 kernels");
    require(prefillLocal.splits > 0 && prefillGlobal.splits > 0 &&
                prefillLocal.workspace.partialsBytes > 0 &&
                prefillGlobal.workspace.partialsBytes > 0 &&
                verifyLocal.workspace.partialsBytes > 0 &&
                verifyGlobal.workspace.partialsBytes > 0,
                "an attention plan has no split workspace");
  }
}

void checkPackage() {
  model::ModelPackage package;
  model::Gemma4MoeWeights weights = fakeWeights();
  model::DFlashDraftLayout draft;
  draft.kind = model::DraftKind::Null;
  draft.layers = 1;
  draft.hiddenSize = layout.hiddenSize;
  draft.vocabularySize = layout.vocabularySize;
  draft.qkvSize = 6144;
  draft.attentionSize = 4096;
  draft.kvHeads = 8;
  draft.intermediateSize = 4224;
  draft.targetHiddenSize = layout.capturedHiddenSize();
  draft.slidingWindow = 1024;
  draft.selectorRank = 0;
  ops::VisionLayout vision;
  vision.outputHiddenSize = layout.hiddenSize;
  package.descriptor = model::makeModelDescriptor("gemma4 test", layout, draft, vision);
  require(package.descriptor.valid(), "the Gemma 4 descriptor is not valid");
  package.target = std::move(weights);
  model::NullDraftWeights draftWeights{};
  draftWeights.layout = draft;
  package.draft = draftWeights;
  const model::RuntimeGeometry geometry =
      model::RuntimeGeometry::from(package, kv::Format::Int8);
  require(geometry.target.gemmaMoe && geometry.target.layers == 30 &&
              geometry.target.gdnLayers == 0 && geometry.target.experts == 128 &&
              geometry.target.attentionWidth == 4096,
          "the runtime geometry lost the Gemma target");
  require(geometry.target.layerKvLayout(true).headDimension == 512,
          "the runtime geometry lost the global head dimension");
  DeviceCapabilities device;
  device.appleGpuFamily = 10;
  device.gpuCoreCount = 20;
  const ops::ExecutionPlans plans(device);
  const auto sizes = model::prefillTensorBytes(geometry, plans);
  // The MoE workspace holds expert tiles of the padded 704-wide block and
  // the attention splits' workspace of the wider global geometry.
  require(sizes[uint32_t(model::moeScratchTensor<model::PrefillTensor>(0))] > 0 &&
              sizes[uint32_t(model::PrefillTensor::AttentionPartials)] > 0 &&
              sizes[uint32_t(model::PrefillTensor::RopeCosAlt)] > 0,
          "the Gemma arenas did not size the MoE, split or alt-RoPE buffers");
  const auto decodeSizes = model::decodeTensorBytes(geometry, plans);
  require(decodeSizes[uint32_t(model::DecodeTensor::RopeCosAlt)] > 0 &&
              decodeSizes[uint32_t(model::DecodeTensor::GemmaResidual)] > 0,
          "the decode arena lost the alt RoPE or Gemma residual buffers");
}

} // namespace

int main() {
  try {
    checkLayout();
    checkKvLayouts();
    checkGeometry();
    checkAttentionPlans();
    checkPackage();
    std::cout << "gemma4 target: PASS\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
