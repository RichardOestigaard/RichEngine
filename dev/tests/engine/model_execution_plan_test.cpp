#include "TestChecks.hpp"
#include "model/ModelFactory.hpp"
#include "model/TargetLoader.hpp"
#include "model/MiniCpmV47.hpp"
#include "model/Qwen3_6Moe.hpp"
#include "model/Qwen3_8.hpp"
#include "model/RuntimeArenas.hpp"
#include "ops/DraftAttention.hpp"

#include "metal/abi/QuantFormat.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {

using namespace richengine;

using richengine::test::require;

model::ModelPackage package(const model::TargetLayout &layout) {
  model::ModelPackage result;
  model::TargetModelWeights target;
  target.layout = layout;
  model::DFlashDraftLayout draft;
  if (layout.arch == "qwen3_6moe") {
    draft.layers = 6;
    draft.hiddenSize = 2048;
    draft.dynamicSize = 512;
    draft.intermediateSize = 6144;
    draft.targetHiddenSize = target.layout.capturedHiddenSize();
  } else if (layout.arch == "minicpm_v47_moe") {
    draft = model::minicpmV47DraftLayout();
  }
  const auto projection = [](uint32_t n, uint32_t k) {
    return ops::Projection(n, k, ops::AffineWeights{});
  };
  target.logitsProjection =
      projection(model::paddedVocabularySize(layout.vocabularySize),
                 layout.hiddenSize);
  target.layers.resize(layout.layers);
  for (uint32_t i = 0; i < layout.layers; ++i) {
    auto &layer = target.layers[i];
    if (layout.isFullAttentionLayer(i)) {
      model::AttentionMixerWeights attention;
      attention.inputProjection = projection(layout.packedFullWidth, layout.hiddenSize);
      attention.outputProjection = projection(layout.hiddenSize, layout.attentionWidth);
      layer.mixer = std::move(attention);
    } else {
      model::GdnMixerWeights gdn;
      gdn.inputProjection = projection(layout.packedGdnWidth, layout.hiddenSize);
      gdn.outputProjection = projection(layout.hiddenSize, layout.attentionWidth);
      layer.mixer = std::move(gdn);
    }
    if (layout.ffnKind == model::FfnKind::Dense) {
      auto &ffn = layer.ffn.emplace<model::DenseFfnWeights>();
      ffn.gateProjection = projection(layout.intermediateSize, layout.hiddenSize);
      ffn.upProjection = ffn.gateProjection;
      ffn.downProjection = projection(layout.hiddenSize, layout.intermediateSize);
    } else {
      // The shared MoE block every sparse layer carries.
      layer.ffn = ops::MoeWeights{};
    }
  }
  ops::VisionLayout vision;
  vision.outputHiddenSize = target.layout.hiddenSize;
  result.descriptor = model::makeModelDescriptor(
      "operator workspace test", target.layout, draft, vision);
  result.target = std::move(target);
  result.draft = richengine::model::DFlashDraftWeights{};
  std::get<richengine::model::DFlashDraftWeights>(result.draft).layout = draft;
  return result;
}

void checkMixedLayouts() {
  auto mixed = package(model::qwen38TargetLayout());
  auto &target = std::get<model::Qwen3_8Weights>(mixed.target);
  auto &up = std::get<model::DenseFfnWeights>(target.layers.front().ffn).upProjection;
  up = ops::Projection(up.outputSize, up.inputSize,
                       ops::BlockWeights{{ops::QuantizedSegment::planes(GGUF_FMT_Q4K, up.outputSize,
                                                                        up.inputSize, {}, {}, {})}});
  bool mismatchRejected = false;
  try { static_cast<void>(model::targetModelGeometry(target)); }
  catch (const model::WeightStoreError &) { mismatchRejected = true; }
  require(mismatchRejected, "incompatible fused gate/up layouts reached execution");
  std::get<model::DenseFfnWeights>(target.layers.front().ffn).gateProjection = up;
  require(target.logitsProjection.layout() == ops::WeightLayout::Affine64,
          "mixed fixture must keep an affine vocabulary head");
  for (uint32_t family : {9U, 10U, 11U}) {
    DeviceCapabilities device;
    device.appleGpuFamily = family;
    device.gpuCoreCount = 16;
    ops::ExecutionPlans plans(device);
    const auto geometry = model::RuntimeGeometry::from(mixed, kv::Format::Int8);
    const auto head = target.logitsProjection.shape();
    const auto containsHead = [&](const auto &shapes) {
      return std::find(shapes.begin(), shapes.end(), head) != shapes.end();
    };
    require(!containsHead(geometry.target.prefillProjections) &&
                containsHead(geometry.target.decodeProjections),
            "vocabulary head must reserve workspace only in decode");
    const auto scratch = model::DecodeArena::linearScratchSize(geometry, plans);
    for (uint32_t lanes = 1; lanes <= model::kLaneCount; ++lanes) {
      const auto plan = plans.linear().plan({{up.outputSize, up.inputSize}, lanes * model::kDecodeRows,
          ops::LinearPhase::Decode, ops::LinearEpilogue::GateUp}, up);
      const auto required = plan.scratchSize();
      require(scratch.input >= required.input && scratch.sums >= required.sums &&
                  scratch.partials >= required.partials && scratch.counters >= required.counters,
              "affine head hid a block-quantized layer's scratch requirement");
      require(model::DecodeArena::gateScratchBytes(geometry, plans) >= plan.gateScratchBytes(),
              "mixed gate/up workspace is too small");
    }
    const auto sizes = model::prefillTensorBytes(geometry, plans);
    for (uint32_t rows : {1U, 8U, 17U, 32U}) {
      const auto required = plans.linear().plan({{up.outputSize, up.inputSize}, rows,
          ops::LinearPhase::Prefill, ops::LinearEpilogue::None}, up).scratchSize();
      require(sizes[uint32_t(model::PrefillTensor::LinearPartials)] >= required.partials &&
                  sizes[uint32_t(model::PrefillTensor::LinearCounters)] >= required.counters,
              "mixed short-prefill split scratch is too small");
    }
  }
  // Every MoE block of a target shares one layout, which the geometry's one
  // MoE shape records: no source mixes them.
  auto sparse = package(model::qwen36MoeTargetLayout());
  auto &moe = std::get<model::Qwen3_6MoeWeights>(sparse.target);
  require(model::targetModelGeometry(moe).moeShape().weightLayout == ops::WeightLayout::Affine64,
          "the MoE shape lost the blocks' layout");
  for (auto &layer : moe.layers) layer.ffn = ops::MoeWeights{ops::BlockMoeWeights{}};
  require(model::targetModelGeometry(moe).moeShape().weightLayout == ops::WeightLayout::Block32 &&
              moe.logitsProjection.layout() == ops::WeightLayout::Affine64,
          "the MoE shape must follow the expert layers, not the head");
  moe.layers.back().ffn = ops::MoeWeights{ops::AffineMoeWeights{}};
  bool mixedRejected = false;
  try { static_cast<void>(model::targetModelGeometry(moe)); }
  catch (const model::WeightStoreError &) { mixedRejected = true; }
  require(mixedRejected, "a target mixing MoE layouts reached execution");
}

// One decode arena serves every lane count, and on Apple10 and later a
// Split128 plan's partials grow with the rows. The arena must hold every
// lane's plan of every affine target and draft projection at the measured
// core counts.
void checkLaneScratch(const model::ModelPackage &package) {
  const auto geometry = model::RuntimeGeometry::from(package, kv::Format::Int8);
  const auto &d = geometry.draft;
  std::vector<ops::LinearMatrix> matrices{
      {d.dynamicSize, d.hiddenSize}, {d.qkvSize, d.hiddenSize}, {d.contextKvSize(), d.hiddenSize},
      {d.hiddenSize, d.attentionSize},
      {d.intermediateSize, d.hiddenSize}, {d.hiddenSize, d.intermediateSize},
      {d.selectorRank, d.hiddenSize}, {d.hiddenSize, d.targetHiddenSize}};
  for (const auto &p : geometry.target.decodeProjections)
    if (p.layout == ops::WeightLayout::Affine64) matrices.push_back({p.outputSize, p.inputSize});
  for (uint32_t family : {10U, 11U})
    for (uint32_t cores : {12U, 20U, 40U}) {
      DeviceCapabilities device;
      device.appleGpuFamily = family;
      device.gpuCoreCount = cores;
      const ops::ExecutionPlans plans(device);
      const auto scratch = model::DecodeArena::linearScratchSize(geometry, plans);
      for (const auto matrix : matrices)
        if (matrix.outputSize && matrix.inputSize)
          for (uint32_t lanes = 1; lanes <= model::kLaneCount; ++lanes)
          for (auto epilogue : {ops::LinearEpilogue::None, ops::LinearEpilogue::Residual,
                                ops::LinearEpilogue::GateUp}) {
            const auto need = plans.linear().plan({matrix, lanes * model::kDecodeRows,
                ops::LinearPhase::Decode, epilogue}).scratchSize();
            require(scratch.partials >= need.partials && scratch.counters >= need.counters,
                    "decode arena scratch below a lane's affine plan");
          }
    }
}

// Arenas are sized from the projections the weights hold, so each must have
// sizes; an empty one would drop its workspace from the bound silently.
void checkUnsizedProjection() {
  auto broken = package(model::qwen38TargetLayout());
  std::get<model::DenseFfnWeights>(
      std::get<model::Qwen3_8Weights>(broken.target).layers.back().ffn).downProjection = ops::Projection();
  bool rejected = false;
  try { static_cast<void>(model::RuntimeGeometry::from(broken, kv::Format::Int8)); }
  catch (const std::invalid_argument &) { rejected = true; }
  require(rejected, "a target projection without sizes reached arena sizing");
}

// The GDN value rows are sized with attentionWidth, so a layout whose value
// heads span another width is refused before loading. Arena sizing checks
// the GDN shape, whose packed rows must also hold the two gates of every
// value head.
void checkGdnWidths() {
  const auto sparse = package(model::qwen36MoeTargetLayout());
  const auto layoutRejected = [](const model::TargetLayout &layout) {
    try { model::requireTargetLayout(layout); }
    catch (const model::WeightStoreError &) { return true; }
    return false;
  };
  const auto geometryRejected = [&](const model::TargetLayout &layout) {
    auto broken = sparse;
    std::get<model::Qwen3_6MoeWeights>(broken.target).layout = layout;
    try { static_cast<void>(model::RuntimeGeometry::from(broken, kv::Format::Int8)); }
    catch (const std::invalid_argument &) { return true; }
    return false;
  };
  const model::TargetLayout shipped = model::qwen36MoeTargetLayout();
  require(!layoutRejected(shipped) && !geometryRejected(shipped),
          "the shipped sparse layout was refused");
  auto narrowValues = shipped;
  narrowValues.gdnValueHeads = narrowValues.gdnKeyHeads;
  narrowValues.convolutionDimension = 3 * narrowValues.gdnKeyHeads * narrowValues.gdnHeadDimension;
  require(layoutRejected(narrowValues),
          "a GDN value width other than attentionWidth was accepted");
  auto withoutGates = shipped;
  withoutGates.packedGdnWidth = shipped.convolutionDimension + shipped.attentionWidth;
  require(geometryRejected(withoutGates), "packed GDN rows without the gates reached arena sizing");
}

// The chain verify's packed lane stride follows the draft's trained block
// depth, rounded up to the decode kernels' eight-row tile and capped at the
// verify capacity: a block-8 draft verifies eight rows (its seven proposals
// plus the anchor), a deeper block keeps sixteen, and a draftless model
// takes the full capacity.
void checkVerifyRows() {
  model::DFlashDraftLayout null{};
  require(null.verifyRows() == model::ExecutionLimits::targetVerifyRows,
          "a draftless model must verify the full capacity");
  model::DFlashDraftLayout dflash2{model::DraftKind::DFlashTwo};
  dflash2.blockSize = 8;
  // The 16-compiled runtime keeps the full verify stride: narrowing it to
  // the trained block is opt-in (RICHENGINE_VERIFY_NARROW,
  // DFlashDraft.hpp). proposalLimit and the draft's own query rows still
  // follow the block.
  require(dflash2.verifyRows() == model::ExecutionLimits::targetVerifyRows &&
              dflash2.proposalLimit() == 7 && dflash2.queryRows() == 8,
          "a block-8 draft must verify its trained depth");
  dflash2.blockSize = 16;
  require(dflash2.verifyRows() == model::ExecutionLimits::targetVerifyRows,
          "a block-16 draft keeps the full verify depth");
  model::DFlashDraftLayout block9{model::DraftKind::DFlashV1};
  block9.blockSize = 9;
  require(block9.verifyRows() == model::ExecutionLimits::targetVerifyRows &&
              block9.proposalLimit() == 8,
          "a nine-row verify depth rounds up to the next eight-row tile");
  model::DFlashDraftLayout block7{model::DraftKind::DFlashTwo};
  block7.blockSize = 7;
  require(block7.verifyRows() == model::ExecutionLimits::targetVerifyRows &&
              block7.proposalLimit() == 6 && block7.queryRows() == 8,
          "a block-7 draft's six proposals and anchor fit the first tile");
  // MiniCPM-V 4.7's DSpark: a 15-proposal block verifies the full stride.
  const auto dspark = model::minicpmV47DraftLayout();
  require(dspark.kind == model::DraftKind::DSpark && dspark.blockSize == 15 &&
              dspark.verifyRows() == model::ExecutionLimits::targetVerifyRows &&
              dspark.proposalLimit() == 15 &&
              dspark.queryRows() == model::ExecutionLimits::targetVerifyRows,
          "the MiniCPM DSpark draft must keep the full verify stride");
}

// MiniCPM-V 4.7 keeps the 35B MoE geometry with a vocabulary (248144) that
// is not a tile multiple: the packed head and each logits row span round to
// it (248320), the token ids stay within the real vocabulary, and the
// draft's 8x2x256 attention head pattern plans.
void checkMiniCpm() {
  auto sparse = package(model::minicpmV47MoeTargetLayout());
  const auto geometry = model::RuntimeGeometry::from(sparse, kv::Format::Int8);
  require(geometry.target.vocabularySize == 248144 &&
              geometry.target.paddedVocabularySize == 248320,
          "the MiniCPM vocabulary's padded storage must round to a tile");
  const auto &draft = geometry.draft;
  require(draft.hiddenSize == 2048 && draft.kvHeads == 2 &&
              draft.attentionHeadDimension == 256 && draft.layers == 5 &&
              draft.targetHiddenSize == 5 * 2048,
          "the MiniCPM DSpark draft geometry changed");
  // The head-kernel dispatch shape: 8 query heads over 2 kv heads of 256
  // (the _q8k2d256 cores). A packed forward keeps the full 16-row stride.
  const auto shape = draft.attentionShape();
  require(shape.queryHeads == 8 && shape.kvHeads == 2 &&
              shape.headDimension == 256 && shape.qkvSize == 3072,
          "the MiniCPM draft attention shape changed");
  (void)ops::DraftAttention::plan(shape, 1, model::ExecutionLimits::targetVerifyRows);
}

} // namespace

int main() {
  try {
    checkUnsizedProjection();
    checkGdnWidths();
    checkMixedLayouts();
    checkVerifyRows();
    checkMiniCpm();
    const auto dense = package(model::qwen38TargetLayout());
    const auto sparse = package(model::qwen36MoeTargetLayout());
    const auto minicpm = package(model::minicpmV47MoeTargetLayout());
    checkLaneScratch(dense);
    checkLaneScratch(sparse);
    checkLaneScratch(minicpm);
    std::cout << "model execution plans: PASS (three paired geometries)\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
