#include "DSplineDraft.hpp"
#include "Checked.hpp"
#include "DraftCheckpoint.hpp"
#include "metal/abi/Dspline.h"
#include "ops/KernelNames.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace richengine::model {
namespace {

// bf16 decode/encode for the load-time table folds below (the same
// ties-to-even rounding AffinePreparation's nearestBfloat16 uses).
float dsplineBfloatValue(uint16_t bits) {
  return std::bit_cast<float>(uint32_t(bits) << 16);
}

uint16_t dsplineBfloatBits(float value) {
  const uint32_t bits = std::bit_cast<uint32_t>(value);
  return static_cast<uint16_t>((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
}

// The sampled refinement's norm kernel stages a mask row's sixteen
// gated senders in threadgroup memory; bridge ranks past its bound take
// the plain walk.
constexpr uint32_t kDsplineNormMaxRank = 512;

void requireLayout(const DFlashDraftLayout &layout) {
  if (layout.kind != DraftKind::DSpline)
    throw WeightStoreError("a DSpline draft requires a DSpline layout");
  if (!layout.layers || layout.layers > 32 || !layout.hiddenSize ||
      !layout.vocabularySize || !layout.qkvSize || !layout.attentionSize ||
      !layout.intermediateSize || !layout.attentionHeadDimension ||
      !(layout.rotaryTheta > 0.0F) || !layout.targetHiddenSize ||
      !layout.kvHeads || !layout.bridgeCodeSize || !layout.bridgeRank ||
      !layout.bridgeWriteLayers) {
    throw WeightStoreError("DSpline draft layout contains a zero dimension");
  }
  if (layout.layers >= 32 &&
      layout.bridgeWriteLayers >> layout.layers)
    throw WeightStoreError("DSpline bridge write layers exceed the draft");
  if (!(layout.bridgeWriteLayers & (uint32_t{1} << (layout.layers - 1))))
    throw WeightStoreError(
        "a DSpline draft writes after its last layer (the refinement's "
        "stash and the fused final norm both come of that write)");
  validateQ4Layout(layout.qkvSize, layout.hiddenSize);
  validateQ4Layout(layout.hiddenSize, layout.attentionSize);
  validateQ4Layout(layout.intermediateSize, layout.hiddenSize);
  validateQ4Layout(layout.hiddenSize, layout.intermediateSize);
  validateQ4Layout(layout.hiddenSize, layout.targetHiddenSize);
}

// The checkpoint's format sentinels — draft_block_causal_version,
// last_write_refine_version, independent_write_version and
// terminal_disabled_version — land in model.bin's tail as 8-byte scalars.
// This is the value every released checkpoint carries; the section's
// bytes are checked when the image buffer is CPU-visible, and presence and
// shape are always checked by the section itself.
void requireVersion(metal::MetalBuffer section, std::string_view name) {
  if (!section || section.sizeBytes() != sizeof(uint64_t))
    throw WeightStoreError("DSpline checkpoint version section is malformed");
  if (section.storage() != metal::BufferStorage::Shared ||
      !section.contents()) {
    fprintf(stderr,
            "warning: DSpline %.*s version is not CPU-readable; skipping "
            "its value check\n",
            static_cast<int>(name.size()), name.data());
    return;
  }
  uint64_t value = 0;
  std::memcpy(&value, section.contents(), sizeof(value));
  if (value != 1)
    throw WeightStoreError("unsupported DSpline " + std::string(name));
}

// The terminal-disabled checkpoint's out.weight must be all zeros — the
// write path it would feed does not exist (sglang's DSpineCodeBridge
// refuses the same nonzero terminal projection).
void requireTerminalDisabled(metal::MetalBuffer section) {
  if (section.storage() != metal::BufferStorage::Shared ||
      !section.contents()) {
    fprintf(
        stderr,
        "warning: DSpline terminal projection is not CPU-readable; skipping "
        "its zero check\n");
    return;
  }
  const auto *words = static_cast<const uint16_t *>(section.contents());
  const uint64_t count = section.sizeBytes() / sizeof(uint16_t);
  for (uint64_t i = 0; i < count; ++i) {
    // A bf16 -0.0 reads 0x8000; it is zero.
    if (words[i] & 0x7FFF)
      throw WeightStoreError(
          "DSpline terminal-disabled checkpoint has a nonzero "
          "terminal projection");
  }
}

} // namespace

DSplineDraft::DSplineDraft(const DSplineDraftWeights &weights,
                           metal::MetalBackend &backend,
                           const ops::ExecutionPlans &operators)
    : weights_(weights), backend_(backend), operators_(operators),
      plain_(weights.plain, backend, operators) {
  // RICHENGINE_DSPINE_MMTEST: run dspline_bridge_matmul on synthetic data
  // once and dump inputs/outputs for offline comparison (diagnostic only).
  if (std::getenv("RICHENGINE_DSPINE_MMTEST")) {
    const DFlashDraftLayout &layout = weights_.layout;
    const uint32_t rows = 16;
    const uint32_t hidden = layout.hiddenSize;
    const uint32_t code = layout.bridgeCodeSize;
    const uint32_t rank = layout.bridgeRank;
    auto alloc = [&](uint64_t bytes) {
      metal::MetalBuffer b = backend_.allocateBuffer(
          bytes, metal::BufferStorage::Shared, "dspline-mmtest");
      std::memset(b.contents(), 0, bytes);
      return b;
    };
    metal::MetalBuffer in = alloc(uint64_t{rows} * hidden * 2);
    metal::MetalBuffer out = alloc(uint64_t{rows} * 2 * code * 2);
    {
      auto *iv = static_cast<uint16_t *>(in.contents());
      for (uint64_t i = 0; i < uint64_t{rows} * hidden; ++i)
        iv[i] = static_cast<uint16_t>(0x3F00 + (i % 7)); // ~0.5 + wiggle
    }
    metal::CommandGraph graph;
    const DsplineMatmulParams read{rows, code, hidden, code};
    graph.add("dspline_bridge_matmul",
              {in, backend_.view(weights_.earlyReadWeights, 0,
                                 uint64_t{code} * hidden * 2),
               weights_.readWeight, weights_.readWeight, out},
              read, {code / 64, (rows + 15) / 16, 1});
    static_cast<void>(backend_.submitCommand(graph.dispatches()));
    auto dump = [](const char *name, const metal::MetalBuffer &b) {
      FILE *f =
          std::fopen((std::string("/tmp/dst_") + name + ".bin").c_str(), "wb");
      if (f) {
        std::fwrite(b.contents(), 1, b.sizeBytes(), f);
        std::fclose(f);
      }
    };
    dump("in", in);
    dump("out", out);
    fprintf(stderr, "dspline mmtest: in=%lluB out=%lluB\n",
            (unsigned long long)in.sizeBytes(),
            (unsigned long long)out.sizeBytes());
    (void)rank;
  }
}

void DSplineDraft::addContextPrefill(
    metal::CommandGraph &graph, DFlashPrefillBuffers buffers, uint32_t rows,
    std::span<const DFlashPrefillSpan> spans) const {
  plain_.addContextPrefill(graph, std::move(buffers), rows, spans);
}

void DSplineDraft::addBridgeWrite(metal::CommandGraph &graph,
                                  const DFlashDecodeBuffers &buffers,
                                  uint32_t layer, uint32_t rows) const {
  const DFlashDraftLayout &layout = weights_.layout;
  const uint32_t hidden = layout.hiddenSize;
  const uint32_t code = layout.bridgeCodeSize;
  const uint32_t rank = layout.bridgeRank;
  static std::once_flag weightDumpOnce;
  std::call_once(weightDumpOnce, [&] {
    if (!std::getenv("RICHENGINE_DSPINE_WDUMP"))
      return;
    auto dumpBuffer = [](const char *name, const metal::MetalBuffer &buffer,
                         uint64_t bytes = 0) {
      void *contents = buffer.contents();
      if (!contents)
        return;
      const uint64_t size =
          bytes && bytes < buffer.sizeBytes() ? bytes : buffer.sizeBytes();
      FILE *f =
          std::fopen((std::string("/tmp/dsw_") + name + ".bin").c_str(), "wb");
      if (!f)
        return;
      std::fwrite(contents, 1, size, f);
      std::fclose(f);
    };
    dumpBuffer("early", weights_.earlyReadWeights);
    dumpBuffer("readw", weights_.readWeight);
    dumpBuffer("featw", weights_.featureWriteWeights);
    dumpBuffer("gatew", weights_.gateWeight);
    dumpBuffer("predg", weights_.predecessorGateWeight);
    dumpBuffer("prevw", weights_.prevWeight);
    dumpBuffer("gateb", weights_.gateBias);
    dumpBuffer("codebook", weights_.codebook, 1u << 22);
    dumpBuffer("finorm", weights_.plain.finalNorm.buffer);
  });
  if (!code || !rank || code % 64 || rank % 64 || hidden % 64)
    throw std::invalid_argument("DSpline bridge dims are not 64-aligned");
  if (!buffers.inputTokens || !buffers.projected ||
      !buffers.intermediate)
    throw std::invalid_argument("DSpline bridge buffers are unbound");
  // The code path's scratch lives in the decode pass's idle rows: the
  // normalized read input in `projected` (unused in decode) and the
  // projected/codes/packed/message fields in `intermediate`, free
  // between the down projection that just read it and the next layer's
  // gate/up that rewrites it. Matmul output rows pad to the 16-row tile.
  // The delta field is dead space — the fused write epilogue updates
  // hidden in place — but the scratch layout keeps its span so the
  // packed/rms offsets addSelection recomputes stay put.
  const uint32_t tileRows = (rows + 15) & ~15u;
  const uint64_t field = tileRows;
  const uint64_t oProjected = 0;
  const uint64_t oCodes = oProjected + field * code;
  const uint64_t oPacked = oCodes + field * code;
  const uint64_t oMessage = oPacked + field * 3 * rank;
  const uint64_t oDelta = oMessage + field * rank;
  const uint64_t rmsOffset = (oDelta + field * hidden) * sizeof(uint16_t);
  if (rmsOffset + uint64_t{rows} * sizeof(float) >
      buffers.intermediate.sizeBytes()) {
    throw std::invalid_argument("DSpline bridge scratch exceeds the "
                                "draft intermediate buffer");
  }
  const auto span = [&](uint64_t element, uint64_t elements) {
    return backend_.view(buffers.intermediate, element * sizeof(uint16_t),
                         elements * sizeof(uint16_t));
  };
  const metal::MetalBuffer codeInput = buffers.projected;
  const metal::MetalBuffer projected = span(oProjected, field * code);
  const metal::MetalBuffer codes = span(oCodes, field * code);
  const metal::MetalBuffer packed = span(oPacked, field * 3 * rank);
  const metal::MetalBuffer message = span(oMessage, field * rank);
  const metal::MetalBuffer rms = backend_.view(
      buffers.intermediate, rmsOffset, uint64_t{rows} * sizeof(float));
  const metal::MetalBuffer &hiddenOut = buffers.hidden[(layer & 1) ^ 1];
  const metal::MetalBuffer readWeight =
      layer + 1 == layout.layers
          ? weights_.readWeight
          : backend_.view(weights_.earlyReadWeights,
                          uint64_t{layer} * code * hidden * sizeof(uint16_t),
                          uint64_t{code} * hidden * sizeof(uint16_t));
  const metal::MetalBuffer featureWrite = backend_.view(
      weights_.featureWriteWeights,
      uint64_t{layer} * hidden * rank * sizeof(uint16_t),
      uint64_t{hidden} * rank * sizeof(uint16_t));
  const DsplineBridgeParams bridge{rows, layout.queryRows(), hidden, code,
                                   rank, 1e-6F};
  graph.add("dspline_bridge_prepare", {hiddenOut, codeInput, rms}, bridge,
            {rows, 1, 1});
  const DsplineMatmulParams read{rows, code, hidden, code};
  graph.add("dspline_bridge_matmul",
            {codeInput, readWeight, readWeight, readWeight, projected},
            read, {code / 64, tileRows / 16, 1});
  graph.add("dspline_bridge_codes",
            {projected, weights_.codebook, buffers.inputTokens, codes},
            bridge, {rows, 1, 1});
  // The packed [gate; predecessor_gate; prev] projection and the
  // predecessor-shifted sigmoid message it feeds fuse into one dispatch
  // when every 16-row tile base lands on a block anchor (16 % block ==
  // 0 — query rows of 8 or 16): a threadgroup then needs no predecessor
  // row another tile computed. Other block strides keep the separate
  // matmul and message dispatches.
  if (!(16 % layout.queryRows())) {
    const DsplinePackParams pack{rows, code, rank, layout.queryRows()};
    graph.add("dspline_bridge_pack",
              {codes, weights_.gateWeight, weights_.predecessorGateWeight,
               weights_.prevWeight, weights_.gateBias, packed, message},
              pack, {rank / 64, tileRows / 16, 1});
  } else {
    const DsplineMatmulParams pack{rows, 3 * rank, code, rank};
    graph.add("dspline_bridge_matmul",
              {codes, weights_.gateWeight, weights_.predecessorGateWeight,
               weights_.prevWeight, packed},
              pack, {3 * rank / 64, tileRows / 16, 1});
    graph.add("dspline_bridge_message",
              {packed, weights_.gateBias, message}, bridge,
              {((uint64_t{rows} * rank) + 255) / 256, 1, 1});
  }
  // The feature write fused with its elementwise tail: the delta
  // projection's epilogue applies the RMS-scaled write-back in place
  // (dspline_bridge_finish's math on the bf16-rounded delta) and, on the
  // last write, emits the serial refinement's raw * final_norm stash into
  // the dead `normalized` rows — the second head pass's unnormalized base
  // logits input. One dispatch replaces the delta matmul,
  // dspline_refine_stash and dspline_bridge_finish.
  const DsplineWriteParams write{rows, hidden, rank, hidden,
                                 layout.queryRows(),
                                 layer + 1 == layout.layers ? 1u : 0u};
  graph.add("dspline_bridge_write",
            {message, featureWrite, featureWrite, featureWrite, hiddenOut,
             weights_.plain.finalNorm.buffer, buffers.normalized, rms},
            write, {hidden / 64, tileRows / 16, 1});
}

void DSplineDraft::addDecode(
    metal::CommandGraph &graph, DFlashDecodeBuffers buffers,
    const ops::Projection &vocabularyProjection,
    std::span<const uint32_t> cacheLengths,
    const metal::MetalBuffer &devicePack) const {
  const uint32_t lanes = static_cast<uint32_t>(cacheLengths.size());
  if (!lanes || lanes > ExecutionLimits::maximumBatchWidth ||
      buffers.persistentKeys.size() != weights_.layout.layers ||
      buffers.persistentValues.size() != weights_.layout.layers) {
    throw std::invalid_argument("invalid draft decode batch");
  }
  // The plain draft's layer schedule with the trained code-bridge write
  // after every layer in layout.bridgeWriteLayers: the write takes the
  // layer output (the plain loop's hidden[next]) and updates it in place,
  // so the next iteration's input norm — and the final norm after the
  // last write — read the written residual stream exactly as the bridge's
  // fused norm produces it. The last write's stash the serial refinement
  // (code_bridge_second_pass, layout.bridgeHeadTopk candidates) consumes
  // is still deferred — the proposals come of the plain selector.
  const DFlashDraftLayout &layout = weights_.layout;
  const uint32_t rows = lanes * layout.queryRows();
  const auto attentionPlan =
      operators_.draftAttention(layout.attentionShape(), lanes, layout.queryRows());
  const ops::Linear &linear = operators_.linear();
  const ops::LinearScratch &scratch = buffers.linearScratch;

  // RICHENGINE_DSPLINE_LAYERS=N: run the first N backbone layers then the
  // head — divergence bisection aid (selection falls back to plain).
  static const uint32_t layerLimit = [] {
    const char *e = std::getenv("RICHENGINE_DSPLINE_LAYERS");
    return e ? static_cast<uint32_t>(std::atoi(e)) + 1u : 0u;
  }();
  const uint32_t runLayers =
      layerLimit ? std::min(layerLimit - 1, layout.layers) : layout.layers;
  graph.beginBakedSpan();
  for (uint32_t layer = 0; layer < runLayers; ++layer) {
    const uint32_t current = layer & 1;
    const uint32_t next = current ^ 1;
    const bool causal = (layout.causalLayers >> layer) & 1;
    const DFlashV1DraftLayerWeights &weights = weights_.plain.layers[layer];
    const ops::LinearPlan qkvPlan = linear.decodePlan(weights.qkvProjection, lanes, ops::LinearEpilogue::None, nullptr, layout.queryRows());
    const ops::PreparedInput attentionNormalized = ops::Normalization::addRms(
        graph, buffers.hidden[current], weights.inputNorm, buffers.normalized,
        layout.hiddenSize, rows, scratch, qkvPlan.input());
    linear.add(graph,
               {.input = buffers.normalized, .output = buffers.proposalQkv, .scratch = scratch,
                .prepared = attentionNormalized},
               weights.qkvProjection, qkvPlan);
    ops::DraftAttention::addPrepare(
        graph,
        {buffers.proposalQkv, buffers.attention, weights.queryNorm,
         weights.keyNorm, buffers.ropeCos, buffers.ropeSin, buffers.queryKeys,
         buffers.queryValues},
        attentionPlan);
    ops::DraftAttention::addDecode(
        graph,
        {buffers.attention, buffers.persistentKeys[layer],
         buffers.persistentValues[layer], buffers.queryKeys,
         buffers.queryValues},
        cacheLengths, attentionPlan, causal, devicePack);
    ops::DraftAttention::addReorder(graph, buffers.attention,
                                    buffers.proposalQkv, attentionPlan);
    linear.add(graph, {.input = buffers.proposalQkv, .output = buffers.residual,
                       .residual = buffers.hidden[current], .scratch = scratch},
               weights.outputProjection,
               linear.decodePlan(weights.outputProjection, lanes,
                                 ops::LinearEpilogue::Residual, nullptr,
                                 layout.queryRows()));
    const ops::LinearPlan mlpPlan = linear.decodePlan(weights.upProjection, lanes, ops::LinearEpilogue::None, nullptr, layout.queryRows());
    const ops::PreparedInput mlpNormalized = ops::Normalization::addRms(
        graph, buffers.residual, weights.postAttentionNorm, buffers.normalized,
        layout.hiddenSize, rows, scratch, mlpPlan.input());
    linear.add(graph,
               {.input = buffers.normalized, .output = buffers.intermediate, .gateScratch = buffers.gateScratch,
                .scratch = scratch,
                .prepared = mlpNormalized},
               weights.upProjection,
               linear.decodePlan(weights.upProjection, lanes, ops::LinearEpilogue::GateUp, &weights.gateProjection, layout.queryRows()),
               &weights.gateProjection);
    linear.add(graph, {.input = buffers.intermediate, .output = buffers.hidden[next],
                       .residual = buffers.residual, .scratch = scratch},
               weights.downProjection,
               linear.decodePlan(weights.downProjection, lanes,
                                 ops::LinearEpilogue::Residual, nullptr,
                                 layout.queryRows()));
    if ((layout.bridgeWriteLayers >> layer) & 1)
      addBridgeWrite(graph, buffers, layer, rows);
  }

  // The final norm feeds the shared vocabulary head; the plain selector
  // works on the head's logits directly and needs no projection of its own.
  const ops::LinearPlan headPlan = linear.decodePlan(vocabularyProjection, lanes, ops::LinearEpilogue::None, nullptr, layout.queryRows());
  const ops::PreparedInput finalHidden = ops::Normalization::addRms(
      graph, buffers.hidden[runLayers & 1], weights_.plain.finalNorm,
      buffers.finalHidden, layout.hiddenSize, rows, scratch, headPlan.input());
  linear.add(
      graph, {.input = buffers.finalHidden, .output = buffers.logits, .scratch = scratch, .prepared = finalHidden},
      vocabularyProjection, headPlan);
  // Gemma's softcap applies to the draft's proposal logits too.
  if (layout.logitSoftcap > 0.0F) {
    const uint32_t count = rows * layout.vocabularySize;
    graph.addTail(std::string(ops::kDecodeLogitSoftcap), {buffers.logits},
                  layout.logitSoftcap, count,
                  {(count + 255) / 256, 1, 1}, {256, 1, 1});
  }
  graph.endBakedSpan();
  // The serial refinement's window into this decode's buffers: the
  // intermediate scratch keeps the last write's packed q columns and rms
  // through selection, `normalized` holds the staged raw*final_norm rows
  // the second head pass reads, and the head projection itself repeats
  // for the unnormalized base logits.
  stash_.intermediate = buffers.intermediate;
  stash_.rawnorm = buffers.normalized;
  stash_.scratch = buffers.linearScratch;
  stash_.head = &vocabularyProjection;
  stash_.rows = rows;
  stash_.valid = runLayers == layout.layers;
}

void DSplineDraft::addSelection(
    metal::CommandGraph &graph, const ops::DraftSelectorBuffers &buffers,
    std::span<const uint32_t> anchors,
    std::span<const ops::SamplingPolicy> policies, uint32_t treeMask) const {
  const DFlashDraftLayout &layout = weights_.layout;
  const uint32_t lanes = static_cast<uint32_t>(anchors.size());
  const uint32_t positions = layout.queryRows() - 1;
  uint32_t samplingMask = 0;
  for (uint32_t lane = 0; lane < lanes; ++lane)
    if (policies[lane].samples())
      samplingMask |= uint32_t{1} << lane;
  const bool sampled = samplingMask != 0;
  // The serial-table refinement covers the chain case — greedy lanes
  // walk its argmax, sampled lanes the renormalized table's filtered
  // draw; anything else — a comb-tree batch, a refinement width other
  // than the selector's candidate count, a bridge rank past the norm
  // kernel's bound, tables that could not build, or a selection without
  // its decode's stash — takes the plain walk.
  static const bool plainOnly =
      std::getenv("RICHENGINE_DSPINE_PLAIN") != nullptr;
  const bool sampledTables =
      weights_.serialGram && weights_.serialRW &&
      layout.bridgeRank <= kDsplineNormMaxRank;
  if (plainOnly || treeMask || !stash_.valid ||
      !weights_.serialTsv ||
      !weights_.serialTg || !lanes ||
      layout.bridgeHeadTopk != RICHENGINE_DRAFT_CANDIDATES ||
      stash_.rows != lanes * layout.queryRows() ||
      (sampled && !sampledTables)) {
    static bool logged = false;
    if (!logged) {
      logged = true;
      fprintf(stderr, "dspline refine fallback: tree=%u sampled=%d stash=%d tsv=%d tg=%d topk=%u/%u rows=%u want=%u\n",
              treeMask, (int)sampled, (int)stash_.valid, (bool)weights_.serialTsv, (bool)weights_.serialTg,
              layout.bridgeHeadTopk, (unsigned)RICHENGINE_DRAFT_CANDIDATES,
              stash_.rows, lanes * layout.queryRows());
    }
    plain_.addSelection(graph, buffers, anchors, policies, treeMask);
    return;
  }
  const ops::DraftSelectorWorkspace workspace =
      ops::DraftSelector::workspace(RICHENGINE_DRAFT_PROPOSAL_TOKENS);
  // The top-16 pass's shard partials are global — lanes*positions*shards*K
  // floats in all, one row's worth per proposal row in row order. Each
  // lane's tail block in partialValues follows them: the
  // positions*K*K fp32 score table, then for a batch that samples the
  // positions*K refined-RMS denominators; the shared [rows][rank] bf16
  // raw projection the norm kernel reads comes after every lane's
  // block. `unary` holds the argmax idx rows exactly.
  const uint64_t partials =
      uint64_t{lanes} * RICHENGINE_DRAFT_PROPOSAL_TOKENS *
      RICHENGINE_DRAFT_SAMPLING_SHARDS * RICHENGINE_DRAFT_CANDIDATES;
  const uint64_t laneScores =
      uint64_t{positions} * RICHENGINE_DRAFT_CANDIDATES *
      RICHENGINE_DRAFT_CANDIDATES;
  const uint64_t laneTail =
      laneScores +
      (sampled ? uint64_t{positions} * RICHENGINE_DRAFT_CANDIDATES : 0);
  const uint64_t rmsrefBase = partials + laneScores;
  const uint64_t rwBase = partials + uint64_t{lanes} * laneTail;
  const uint64_t rwFloats =
      sampled ? uint64_t{stash_.rows} * layout.bridgeRank / 2 : 0;
  const uint64_t scoresLane = workspace.partialValuesBytes / sizeof(float);
  if (rwBase + rwFloats > uint64_t{lanes} * scoresLane) {
    plain_.addSelection(graph, buffers, anchors, policies, treeMask);
    return;
  }
  // The top-16 over the plain logits picks each mask row's candidates —
  // the same shard pass the plain walk runs.
  const ops::Linear &linear = operators_.linear();
  const uint32_t queryRows = layout.queryRows();
  const uint32_t logitsStride =
      paddedVocabularySize(layout.vocabularySize);
  graph.addTail(std::string(ops::kDraftSelectTop16Sharded),
                {buffers.logits, buffers.partialIds,
                 buffers.partialValues},
                DraftTop16Params{layout.vocabularySize, logitsStride},
                queryRows,
                {uint64_t{lanes} * RICHENGINE_DRAFT_PROPOSAL_TOKENS *
                     RICHENGINE_DRAFT_SAMPLING_SHARDS,
                 1, 1});
  // The unnormalized base logits: dspline_refine_rows reads each mask
  // row's logits at its merged top-16 candidate columns only, so an
  // affine head gathers just those columns (the top-16 ids the first
  // head pass's logits already produced) instead of repeating the whole
  // [rows][hidden] x [hidden][vocabulary] projection. The writes land in
  // the same logits slots; every column the rescoring never reads keeps
  // the first pass's stale value. A head of another layout, an input
  // that is not a whole quant-group count, or one past the gather
  // kernel's 128-group bound still runs the full second pass.
  const uint32_t code = layout.bridgeCodeSize;
  const uint32_t rank = layout.bridgeRank;
  const uint32_t hidden = layout.hiddenSize;
  if (stash_.head->layout() == ops::WeightLayout::Affine64 &&
      stash_.head->inputSize == hidden && !(hidden % 64) &&
      hidden / 64 <= 128) {
    const ops::AffineWeights &affine = stash_.head->affine();
    const DsplineGatherParams gather{positions, queryRows, hidden,
                                     logitsStride};
    graph.add("dspline_head_gather",
              {stash_.rawnorm, affine.weights, affine.scales,
               affine.biases, buffers.partialIds, buffers.partialValues,
               buffers.logits},
              gather, {uint64_t{lanes} * positions, 1, 1});
  } else {
    const ops::LinearPlan headPlan =
        linear.decodePlan(*stash_.head, lanes, ops::LinearEpilogue::None,
                          nullptr, queryRows);
    linear.add(graph,
               {.input = stash_.rawnorm,
                .output = buffers.logits,
                .scratch = stash_.scratch},
               *stash_.head, headPlan);
  }
  // The last write's stash fields inside the intermediate scratch: the
  // packed q columns at tileRows*2*code and the fp32 rms after delta.
  const uint64_t field = stash_.rows;
  const uint64_t packedOffset = field * 2 * code * sizeof(uint16_t);
  const uint64_t rmsOffset =
      (field * 2 * code + field * 4 * rank + field * hidden) *
      sizeof(uint16_t);
  const metal::MetalBuffer packed = backend_.view(
      stash_.intermediate, packedOffset,
      field * 3 * rank * sizeof(uint16_t));
  const metal::MetalBuffer rms = backend_.view(
      stash_.intermediate, rmsOffset,
      uint64_t{stash_.rows} * sizeof(float));
  DsplineRefineParams refine{lanes,
                             positions,
                             queryRows,
                             rank,
                             logitsStride,
                             static_cast<uint32_t>(laneTail),
                             static_cast<uint32_t>(partials),
                             static_cast<uint32_t>(rmsrefBase),
                             static_cast<uint32_t>(rwBase),
                             hidden,
                             samplingMask,
                             layout.rmsEpsilon,
                             {},
                             {},
                             {},
                             {}};
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    const ops::SamplingPolicy &policy = policies[lane];
    if (!policy.samples())
      continue;
    refine.temperature[lane] = policy.temperature;
    // Upstream's keep is rank < top_k with top_k >= K disabling; the
    // policy's 0 or a top-k past the candidates means keep them all.
    refine.top_k[lane] =
        policy.topK && policy.topK < RICHENGINE_DRAFT_CANDIDATES
            ? policy.topK
            : RICHENGINE_DRAFT_CANDIDATES;
    refine.top_p[lane] = policy.topP;
    refine.min_p[lane] = policy.minP;
  }
  graph.add("dspline_refine_rows",
            {buffers.partialIds, buffers.partialValues, buffers.logits,
             weights_.serialTsv, weights_.serialTg, packed, rms,
             weights_.gateBias, buffers.draftInput, buffers.candidates,
             buffers.unary, buffers.partialValues},
            refine, {uint64_t{lanes} * positions, 1, 1}, {256, 1, 1});
  if (sampled) {
    // The sampled lanes' normalization terms: the staged raw*final_norm
    // rows through the norm-divided write table give raw @ write_w (the
    // raw x delta cross term), and dspline_refine_norm folds it with the
    // write Gram into every predecessor lane's refined RMS.
    graph.add("dspline_bridge_matmul",
              {stash_.rawnorm, weights_.serialRW, weights_.serialRW,
               weights_.serialRW,
               backend_.view(buffers.partialValues, rwBase * sizeof(float),
                             uint64_t{stash_.rows} * rank *
                                 sizeof(uint16_t))},
              DsplineMatmulParams{stash_.rows, rank, hidden, rank},
              {rank / 64, (stash_.rows + 15) / 16, 1});
    graph.add("dspline_refine_norm",
              {buffers.candidates, buffers.draftInput, packed, rms,
               weights_.serialTsv, weights_.gateBias, weights_.serialGram,
               buffers.partialValues},
              refine, {uint64_t{lanes} * positions, 1, 1}, {256, 1, 1});
  }
  graph.add("dspline_refine_walk",
            {buffers.unary, buffers.candidates, buffers.partialValues, rms,
             buffers.uniforms, buffers.proposedTokens,
             buffers.proposalProbabilities, buffers.confidence},
            refine, {lanes, 1, 1}, {32, 1, 1});
}

void DSplineDraft::addContextCommit(
    metal::CommandGraph &graph, DFlashContextBuffers buffers,
    std::span<const uint32_t> startPositions) const {
  plain_.addContextCommit(graph, std::move(buffers), startPositions);
}

namespace {

// Reads a DSpline draft's files in their section order: each layer (the
// plain draft's sections), then model.bin — fc, the norms, the bridge
// tensors and the format sentinels.
template <class Files>
DSplineDraftWeights readDraft(metal::MetalBackend &backend, Files &files,
                              const DFlashDraftLayout &layout) {
  const uint64_t allocationBaseline = backend.memoryStats().allocatedBytes;
  DSplineDraftWeights result;
  result.layout = layout;
  result.plain.layout = layout;
  result.plain.layers.reserve(layout.layers);
  const uint64_t headNormBytes = checkedMultiply<WeightStoreError>(
      layout.attentionHeadDimension, kBFloat16Bytes,
      "draft head norm bytes");

  for (uint32_t layerIndex = 0; layerIndex < layout.layers; ++layerIndex) {
    WeightFile file = files.layer(layerIndex);
    DFlashV1DraftLayerWeights layer;
    layer.inputNorm = readNorm(file, layout.hiddenSize, false, "input-norm");
    layer.inputNorm.rmsEpsilon = layout.rmsEpsilon;
    layer.qkvProjection = readProjection(
        file, layout.qkvSize, layout.hiddenSize, "qkv");
    layer.queryNorm = file.section(headNormBytes, "query-norm");
    layer.keyNorm = file.section(headNormBytes, "key-norm");
    layer.outputProjection = readProjection(
        file, layout.hiddenSize, layout.attentionSize,
        "attention-output");
    layer.postAttentionNorm =
        readNorm(file, layout.hiddenSize, false, "post-attention-norm");
    layer.postAttentionNorm.rmsEpsilon = layout.rmsEpsilon;
    layer.gateProjection = readProjection(
        file, layout.intermediateSize, layout.hiddenSize, "mlp-gate");
    layer.upProjection = readProjection(
        file, layout.intermediateSize, layout.hiddenSize, "mlp-up");
    layer.downProjection = readProjection(
        file, layout.hiddenSize, layout.intermediateSize, "mlp-down");
    file.finish();
    result.files.push_back(file.record());
    result.plain.layers.push_back(std::move(layer));
  }

  {
    WeightFile file = files.model();
    result.plain.contextProjection = readProjection(
        file, layout.hiddenSize, layout.targetHiddenSize,
        "context-projection");
    result.plain.hiddenNorm =
        readNorm(file, layout.hiddenSize, false, "hidden-norm");
    result.plain.hiddenNorm.rmsEpsilon = layout.rmsEpsilon;
    result.plain.finalNorm =
        readNorm(file, layout.hiddenSize, false, "final-norm");
    result.plain.finalNorm.rmsEpsilon = layout.rmsEpsilon;
    const auto section =
        [&file](uint64_t rows, uint64_t columns, const char *name) {
          return file.section(
              checkedMultiply<WeightStoreError>(
                  checkedMultiply<WeightStoreError>(rows, columns,
                                                    "bridge elements"),
                  kBFloat16Bytes, "bridge bytes"),
              name);
        };
    result.codebook = section(layout.vocabularySize, layout.bridgeCodeSize,
                              "bridge-codebook");
    result.earlyReadWeights = section(
        uint64_t{layout.layers - 1} * layout.bridgeCodeSize,
        layout.hiddenSize, "bridge-early-read-weights");
    result.readWeight =
        section(layout.bridgeCodeSize, layout.hiddenSize, "bridge-read-weight");
    result.featureWriteWeights = section(
        uint64_t{layout.layers} * layout.hiddenSize, layout.bridgeRank,
        "bridge-feature-write-weights");
    result.gateWeight = section(layout.bridgeRank, layout.bridgeCodeSize,
                                "bridge-gate-weight");
    result.predecessorGateWeight = section(
        layout.bridgeRank, layout.bridgeCodeSize,
        "bridge-predecessor-gate-weight");
    result.prevWeight = section(layout.bridgeRank, layout.bridgeCodeSize,
                                "bridge-prev-weight");
    result.gateBias =
        section(1, layout.bridgeRank, "bridge-gate-bias");
    // The terminal projection stays in the file so a packed and a
    // checkpoint draft read the same tail; a nonzero one is refused.
    requireTerminalDisabled(
        section(layout.bridgeCodeSize, layout.bridgeRank,
                "bridge-terminal-projection"));
    requireVersion(file.section(sizeof(uint64_t), "bridge-independent-write-version"),
                   "independent_write_version");
    requireVersion(file.section(sizeof(uint64_t), "bridge-terminal-disabled-version"),
                   "terminal_disabled_version");
    requireVersion(file.section(sizeof(uint64_t), "draft-block-causal-version"),
                   "draft_block_causal_version");
    requireVersion(file.section(sizeof(uint64_t), "last-write-refine-version"),
                   "last_write_refine_version");
    file.finish();
    result.files.push_back(file.record());
  }

  result.plain.actualAllocatedBytes = result.actualAllocatedBytes =
      metal::allocationDelta(allocationBaseline,
                             backend.memoryStats().allocatedBytes);
  return result;
}

// The refinement's tables (upstream build_serial_tables):
//   TSV[v] = [predecessor_gate; prev] @ bf16(sqrt(code) * codebook[v])
//   TG[v]  = bf16(head[v] * final_norm) @ feature_write_weights[last]
// both bf16 row-major, built once on the GPU in 256-tile-aligned chunks:
// dspline_bridge_codes reproduces the anchor substitution's scaled codes,
// dspline_bridge_matmul runs both projections, and
// dspline_table_dequant unpacks the affine Q4 head's tiled planes. The
// feature write transposes host-side — the image sections are
// CPU-visible. Returns false when the head's layout or the checkpoint's
// refinement width cannot feed the build; selection then stays plain.
bool buildSerialTables(metal::MetalBackend &backend,
                       DSplineDraftWeights &result,
                       const ops::Projection *head) {
  const DFlashDraftLayout &layout = result.layout;
  const uint32_t hidden = layout.hiddenSize;
  const uint32_t code = layout.bridgeCodeSize;
  const uint32_t rank = layout.bridgeRank;
  const uint64_t vocabulary = layout.vocabularySize;
  if (!head || head->layout() != ops::WeightLayout::Affine64 ||
      head->inputSize != hidden || head->outputSize < vocabulary ||
      vocabulary % kQ4StorageN || code % 64 || rank % 64 || hidden % 64 ||
      layout.bridgeHeadTopk != RICHENGINE_DRAFT_CANDIDATES ||
      !result.plain.finalNorm.buffer ||
      !result.plain.finalNorm.buffer.contents() ||
      !result.featureWriteWeights.contents() ||
      !result.codebook.contents()) {
    return false;
  }
  const ops::AffineWeights &affine = head->affine();
  // The last write's feature projection, transposed to the matmul's
  // [output][input] layout, on the CPU: 5120x512 bf16.
  metal::MetalBuffer writeT = backend.allocateBuffer(
      uint64_t{rank} * hidden * sizeof(uint16_t),
      metal::BufferStorage::Shared, "dspline-writeT");
  {
    const auto *source = static_cast<const uint16_t *>(
        result.featureWriteWeights.contents()) +
        uint64_t{layout.layers - 1} * hidden * rank;
    auto *target = static_cast<uint16_t *>(writeT.contents());
    for (uint64_t r = 0; r < rank; ++r)
      for (uint64_t h = 0; h < hidden; ++h)
        target[r * hidden + h] = source[h * rank + r];
  }
  // The codes kernel's anchor substitution takes the token ids it gathers;
  // a block of one makes every row an anchor row of its own id.
  metal::MetalBuffer ids =
      backend.allocateBuffer(vocabulary * sizeof(uint32_t),
                             metal::BufferStorage::Shared, "dspline-ids");
  {
    auto *target = static_cast<uint32_t *>(ids.contents());
    for (uint64_t i = 0; i < vocabulary; ++i) target[i] = uint32_t(i);
  }
  result.serialTsv = backend.allocateBuffer(
      vocabulary * 2 * rank * sizeof(uint16_t),
      metal::BufferStorage::Private, "dspline-tsv");
  result.serialTg = backend.allocateBuffer(
      vocabulary * rank * sizeof(uint16_t),
      metal::BufferStorage::Private, "dspline-tg");
  constexpr uint32_t chunk = 4096;
  metal::MetalBuffer scaled = backend.allocateBuffer(
      uint64_t{chunk} * code * sizeof(uint16_t),
      metal::BufferStorage::Private, "dspline-scaled");
  metal::MetalBuffer headRows = backend.allocateBuffer(
      uint64_t{chunk} * hidden * sizeof(uint16_t),
      metal::BufferStorage::Private, "dspline-headrows");
  const auto view = [&backend](const metal::MetalBuffer &base, uint64_t offset,
                             uint64_t bytes) {
    return backend.view(base, offset, bytes);
  };
  metal::CommandGraph graph;
  for (uint64_t start = 0; start < vocabulary; start += chunk) {
    const uint32_t rows =
        static_cast<uint32_t>(std::min<uint64_t>(chunk, vocabulary - start));
    const uint32_t tileRows = (rows + 15) & ~15u;
    // TSV: the scaled codebook rows (block 1 — every row its own anchor)
    // through the packed [predecessor_gate; prev] projection. The anchor
    // substitution gathers by absolute token id, so the codebook binds
    // unviewed while the ids view starts the chunk's ids at row 0.
    const metal::MetalBuffer tsv =
        view(result.serialTsv, start * 2 * rank * sizeof(uint16_t),
             uint64_t{rows} * 2 * rank * sizeof(uint16_t));
    graph.add("dspline_bridge_codes",
              {scaled, result.codebook,
               view(ids, start * sizeof(uint32_t),
                    uint64_t{rows} * sizeof(uint32_t)),
               scaled},
              DsplineBridgeParams{rows, 1, hidden, code, rank, 1e-6F},
              {rows, 1, 1});
    graph.add("dspline_bridge_matmul",
              {scaled, result.predecessorGateWeight, result.prevWeight,
               result.prevWeight, tsv},
              DsplineMatmulParams{rows, 2 * rank, code, rank},
              {2 * rank / 64, tileRows / 16, 1});
    // TG: the affine head's rows, unpacked and folded with the final norm,
    // through the transposed last feature write.
    const metal::MetalBuffer tg =
        view(result.serialTg, start * rank * sizeof(uint16_t),
             uint64_t{rows} * rank * sizeof(uint16_t));
    graph.add("dspline_table_dequant",
              {view(affine.weights, start * hidden / 2,
                    uint64_t{rows} * hidden / 2),
               view(affine.scales, start * hidden / 32,
                    uint64_t{rows} * hidden / 32),
               view(affine.biases, start * hidden / 32,
                    uint64_t{rows} * hidden / 32),
               result.plain.finalNorm.buffer, headRows},
              DsplineTableParams{rows, hidden},
              {(uint64_t{rows} * hidden + 255) / 256, 1, 1});
    graph.add("dspline_bridge_matmul",
              {headRows, writeT, writeT, writeT, tg},
              DsplineMatmulParams{rows, rank, hidden, rank},
              {rank / 64, tileRows / 16, 1});
  }
  // The sampled lanes' normalization tables (upstream serial_rs_table's
  // refined-RMS denominators): serialGram = writeT @ writeT^T, the write
  // Gram scoring a gated sender's ||delta||^2, and serialRW = writeT's
  // rows divided by the final norm weight, so the staged raw*final_norm
  // rows matmul into raw @ write_w for the raw x delta cross term.
  result.serialRW = backend.allocateBuffer(
      uint64_t{rank} * hidden * sizeof(uint16_t),
      metal::BufferStorage::Shared, "dspline-rw");
  {
    const auto *norm = static_cast<const uint16_t *>(
        result.plain.finalNorm.buffer.contents());
    const auto *source =
        static_cast<const uint16_t *>(writeT.contents());
    auto *target =
        static_cast<uint16_t *>(result.serialRW.contents());
    for (uint64_t r = 0; r < rank; ++r)
      for (uint64_t h = 0; h < hidden; ++h) {
        const float weight = dsplineBfloatValue(norm[h]);
        target[r * hidden + h] =
            weight != 0.0F ? dsplineBfloatBits(
                                 dsplineBfloatValue(source[r * hidden + h]) /
                                 weight)
                           : uint16_t(0);
      }
  }
  result.serialGram = backend.allocateBuffer(
      uint64_t{rank} * rank * sizeof(uint16_t),
      metal::BufferStorage::Private, "dspline-gram");
  graph.add("dspline_bridge_matmul",
            {writeT, writeT, writeT, writeT, result.serialGram},
            DsplineMatmulParams{rank, rank, hidden, rank},
            {rank / 64, rank / 16, 1});
  static_cast<void>(backend.submitCommand(graph.dispatches()));
  return true;
}

} // namespace

WeightFile PackedDSplineDraftFiles::layer(uint32_t index) const {
  const std::string filename = "layer-" + std::to_string(index) + ".bin";
  return images.load(packedImage(directory / filename, "draft/" + filename,
                                 kDSplineDraftMagic, index, 0));
}

WeightFile PackedDSplineDraftFiles::model() const {
  return images.load(packedImage(directory / "model.bin", "draft/model.bin",
                                 kDSplineDraftMagic, layout.layers, 1));
}

DSplineDraftWeights loadDSplineDraftWeights(
    metal::MetalBackend &backend, const DSplineDraftFiles &files,
    DFlashDraftLayout layout, const ops::Projection *vocabularyHead) {
  requireLayout(layout);
  DSplineDraftWeights result =
      std::holds_alternative<std::reference_wrapper<DSplineCheckpointLoader>>(
          files)
          ? readDraft(backend,
                      std::get<std::reference_wrapper<DSplineCheckpointLoader>>(
                          files)
                          .get(),
                      layout)
          : readDraft(backend, std::get<PackedDSplineDraftFiles>(files),
                      layout);
  buildSerialTables(backend, result, vocabularyHead);
  return result;
}

} // namespace richengine::model
