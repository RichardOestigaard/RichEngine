// The qwen4exp target's packed loader and its layer encoder
// (model/Qwen4Exp.hpp). The vendored upstream port under runtime/qwen4/
// is the reference: its loader order (Qwen4Exp.cpp) is this file's packed
// section order and its layer bodies (Qwen4ExpTarget.cpp) are the
// encoder's call order.
#include "model/Qwen4Exp.hpp"
#include "model/TargetLayerEncoder.hpp"
#include "model/TargetLoader.hpp"
#include "model/TargetModelGemmaHelpers.hpp"
#include "model/WeightImages.hpp"
#include "model/WeightStore.hpp"
#include "metal/abi/GDN.h"
#include "ops/BlockIndexer.hpp"
#include "ops/Embedding.hpp"
#include "ops/HyperConnection.hpp"
#include "ops/NgramEmbedding.hpp"

#include <algorithm>
#include <any>
#include <cmath>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace richengine::model {
namespace {

// The packed qwen4exp files' magics: the PLE table file's (the layers' and
// head's are the preset's layerMagic/headMagic).
constexpr std::string_view kPleTableMagic = "MDFQ0003";

// The qwen4exp additions to requireTargetLayout's checks: the
// hyper-connections, the QSA indexer and the PLE layer are set and
// consistent, the recurrent schedule leaves the PLE layer a GDN layer.
// The shared check's Q4 storage-tile rules do not apply: the packed
// expert rows are block segments (640 is no StorageN width).
void requireQwen4ExpLayout(const TargetLayout &layout) {
  const auto zero = [](auto... dimensions) {
    return ((dimensions == 0) || ...);
  };
  if (layout.ffnKind != FfnKind::SparseMoe ||
      zero(layout.maximumContextTokens, layout.layers, layout.hiddenSize,
           layout.vocabularySize, layout.packedGdnWidth,
           layout.packedFullWidth, layout.convolutionDimension,
           layout.gdnKeyHeads, layout.gdnValueHeads,
           layout.gdnHeadDimension, layout.attentionWidth,
           layout.attentionQueryHeads, layout.attentionKvHeads,
           layout.attentionHeadDimension, layout.rotaryPairs,
           layout.fullAttentionPeriod, layout.experts,
           layout.expertsPerToken, layout.expertIntermediateSize,
           layout.hyperConnections, layout.hyperRank, layout.indexerHeads,
           layout.indexerHeadDimension, layout.indexerTopBlocks,
           layout.indexerBlockTokens, layout.pleHeadsPerNgram,
           layout.pleHeadDimension, layout.pleConvolutionTaps) ||
      !(layout.rotaryTheta > 0.0F) || layout.pleNgram < 2 ||
      layout.pleNgram > kQwen4PleMaxNgram ||
      layout.pleLayer >= layout.layers ||
      layout.isFullAttentionLayer(layout.pleLayer) ||
      layout.gdnValueHeads % layout.gdnKeyHeads ||
      layout.convolutionDimension !=
          (2 * layout.gdnKeyHeads + layout.gdnValueHeads) *
              layout.gdnHeadDimension ||
      layout.gdnValueHeads * layout.gdnHeadDimension !=
          layout.attentionWidth ||
      layout.attentionWidth !=
          layout.attentionQueryHeads * layout.attentionHeadDimension ||
      layout.packedFullWidth !=
          layout.attentionQueryStride * layout.attentionWidth +
              2 * layout.attentionKvHeads * layout.attentionHeadDimension ||
      layout.expertsPerToken > layout.experts ||
      std::ranges::any_of(
          std::span(layout.hiddenCaptureLayers)
              .first(layout.captureLayerCount),
          [&](uint32_t layer) { return layer >= layout.layers; }) ||
      !layout.kvLayout().valid() || !layout.gdnStateLayout().valid())
    throw WeightStoreError("qwen4exp target layout is inconsistent");
}

// One mix's sections in the image's order (upstream hyperImage): the
// streams' norm, the gate's down and up projections and — for a mix whose
// block injects — the injection weights.
Qwen4HyperWeights readHyper(WeightFile &file, const BlockTargetFormat &format,
                            const TargetLayout &layout, bool inject,
                            std::string_view label) {
  const uint32_t streams = layout.streamWidth();
  Qwen4HyperWeights hyper;
  hyper.norm = format.norm(file, streams, label);
  hyper.down = format.projection(file, layout.hyperRank, streams, label);
  hyper.up = format.projection(file, streams, layout.hyperRank, label);
  if (inject)
    hyper.inject = file.section(
        uint64_t{layout.hyperConnections} * streams * sizeof(float), label);
  return hyper;
}

// A full-attention layer's QSA indexer after its mixer's sections: the fp32
// query and key projections and their per-head norms.
Qwen4IndexerWeights readIndexer(WeightFile &file,
                                const BlockTargetFormat &format,
                                const TargetLayout &layout) {
  Qwen4IndexerWeights indexer;
  indexer.query =
      format.projection(file, layout.indexerHeads * layout.indexerHeadDimension,
                        layout.hiddenSize, "indexer-query");
  indexer.key = format.projection(file, layout.indexerHeadDimension,
                                  layout.hiddenSize, "indexer-key");
  // The indexer scores in fp32 (upstream ops::BlockIndexer).
  indexer.query.destination = ops::FloatOutput::Float32;
  indexer.key.destination = ops::FloatOutput::Float32;
  indexer.queryNorm =
      format.norm(file, layout.indexerHeadDimension, "indexer-query-norm");
  indexer.keyNorm =
      format.norm(file, layout.indexerHeadDimension, "indexer-key-norm");
  return indexer;
}

// The MoE block's sections, in the qwen3_6moe order: router, the routed
// experts' gate, up and down, the shared expert's, and its scalar gate.
ops::MoeWeights readMoeBlock(WeightFile &file) {
  ops::BlockMoeWeights block;
  block.router = readQuantizedSegment(file, "router");
  block.gate.routed = readQuantizedSegment(file, "experts-gate");
  block.up.routed = readQuantizedSegment(file, "experts-up");
  block.down.routed = readQuantizedSegment(file, "experts-down");
  block.gate.shared = readQuantizedSegment(file, "shared-expert-gate");
  block.up.shared = readQuantizedSegment(file, "shared-expert-up");
  block.down.shared = readQuantizedSegment(file, "shared-expert-down");
  block.sharedScalarGate =
      readQuantizedSegment(file, "shared-expert-scalar-gate");
  return ops::MoeWeights{std::move(block)};
}

// The PLE layer's tensors after its FFN's: the gathered rows' key and value
// projections, the three stream-width norms and the dilated convolution.
Qwen4PleWeights readPle(WeightFile &file, const BlockTargetFormat &format,
                        const TargetLayout &layout) {
  const uint32_t streams = layout.streamWidth();
  Qwen4PleWeights ple;
  ple.key = format.projection(file, streams, layout.pleWidth(), "ple-key");
  ple.value =
      format.projection(file, layout.hiddenSize, layout.pleWidth(), "ple-value");
  ple.keyNorm = format.norm(file, streams, "ple-key-norm");
  ple.queryNorm = format.norm(file, streams, "ple-query-norm");
  ple.convolutionNorm = format.norm(file, streams, "ple-convolution-norm");
  ple.convolution = file.section(
      uint64_t{streams} * layout.pleConvolutionTaps * sizeof(float),
      "ple-convolution");
  return ple;
}

// The PLE hash the packed table's header states, checked against the
// layout (upstream readPleHash checks the same of a GGUF's metadata).
Qwen4PleHash readPleHash(WeightFile &file, const TargetLayout &layout,
                         Qwen4ExpWeights &weights) {
  const metal::MetalBuffer section =
      file.section(sizeof(Qwen4PleHeader), "ple-hash");
  const auto *header = static_cast<const Qwen4PleHeader *>(section.contents());
  if (!header)
    throw WeightStoreError("the packed PLE hash header is not CPU-visible");
  const uint32_t heads = layout.pleHeads();
  Qwen4PleHash hash;
  if (header->headCount != heads || !header->tableRows)
    throw WeightStoreError("the packed PLE hash does not match the layout");
  std::copy(std::begin(header->multipliers), std::end(header->multipliers),
            hash.multipliers.begin());
  hash.tableRows = header->tableRows;
  hash.headOffsets.assign(header->heads, header->heads + heads);
  hash.headVocabularies.assign(header->heads + heads, header->heads + 2 * heads);
  for (uint32_t head = 0; head < heads; ++head)
    if (!hash.headVocabularies[head] ||
        uint64_t{hash.headOffsets[head]} + hash.headVocabularies[head] >
            hash.tableRows)
      throw WeightStoreError(
          "a PLE head's rows lie outside the packed table");
  if (header->eosToken != layout.pleEosToken ||
      (header->imageToken && header->imageToken != layout.pleImageToken))
    throw WeightStoreError("the packed PLE hash does not match the layout");
  weights.pleEosToken = header->eosToken;
  weights.pleImageToken = header->imageToken;
  return hash;
}

// The PLE tensors of a step's rows (upstream Qwen4Scratch's PLE fields):
// the gathered rows' table ids — CPU-computed for a prefill chunk
// (qwen4ExpPleRows), GPU-hashed for a verify step (ngram_embedding_hash) —
// the gathered rows themselves, their key and value projections and the
// gate's gated values and convolution inputs, whose tail the verify
// commit's history writer reads.
struct Qwen4PleScratch final {
  metal::MetalBuffer pleIndices;
  metal::MetalBuffer pleRows;
  metal::MetalBuffer pleValue;
  metal::MetalBuffer pleGate;
  metal::MetalBuffer pleGated;
  metal::MetalBuffer pleConvolution;
};

// The words of a row's QSA block bitmap over the served window, and a row's
// scores: one per block its bitmap covers (upstream maskWords/scoreStride).
uint32_t qsaMaskWords(const TargetLayout &layout) {
  return (layout.maximumContextTokens / BLOCK_INDEXER_BLOCK +
          BLOCK_INDEXER_TILE_SLACK + 31) /
         32;
}
uint32_t qsaScoreStride(const TargetLayout &layout) {
  return qsaMaskWords(layout) * 32;
}
// The entries of one eight-row tile's block list (upstream listStride).
uint32_t qsaListStride(const TargetLayout &layout) {
  return kv::kSparseTileRows * (layout.indexerTopBlocks + 1);
}
// The blocks whose last row is one of `rows` rows from `position` (upstream
// blockCount).
uint32_t qsaBlockCount(uint64_t position, uint32_t rows) {
  return static_cast<uint32_t>((position + rows) / BLOCK_INDEXER_BLOCK -
                               position / BLOCK_INDEXER_BLOCK);
}
// The verify lane's record slots in the records scratch.
constexpr uint32_t kQsaLaneRecords =
    (ExecutionLimits::targetVerifyRows + BLOCK_INDEXER_BLOCK - 1) /
    BLOCK_INDEXER_BLOCK;
// A prefill chunk's rows the QSA selection scores at once (upstream
// kPrefillScoreRows; one window, so no stretch adjustment).
constexpr uint32_t kPrefillScoreRows = 256;

// The hyper-connected step scratch of a prefill chunk or a verify step
// (upstream Qwen4Steps' per-phase tensors): the fp32 residual streams, the
// mix intermediates (normalized/low/gate rows and the injection weights),
// the branch target and the QSA indexer's projections, pooled-key sources,
// scores, bitmaps, block lists and block records.
struct Qwen4StepScratch final {
  metal::MetalBuffer streams;
  metal::MetalBuffer normalized;
  metal::MetalBuffer low;
  metal::MetalBuffer gate;
  metal::MetalBuffer weights;
  metal::MetalBuffer branch;
  metal::MetalBuffer zeros;
  metal::MetalBuffer indexerQuery;
  metal::MetalBuffer indexerKey;
  metal::MetalBuffer qsaQueries;
  metal::MetalBuffer qsaKeys;
  metal::MetalBuffer qsaScores;
  metal::MetalBuffer qsaMask;
  metal::MetalBuffer qsaLists;
  metal::MetalBuffer qsaListCounts;
  metal::MetalBuffer records;
};

Qwen4StepScratch stepScratch(metal::MetalBackend &backend,
                             const TargetLayout &layout, uint64_t rows,
                             bool decode) {
  const auto buffer = [&](uint64_t bytes, const char *label) {
    return backend.allocateBuffer(bytes, metal::BufferStorage::Shared, label);
  };
  const ops::HyperShape hs{layout.hiddenSize, layout.hyperConnections,
                           layout.hyperRank};
  const uint32_t streams = layout.streamWidth();
  const uint32_t queryWidth =
      layout.indexerHeads * layout.indexerHeadDimension;
  const uint32_t maskWords = qsaMaskWords(layout);
  const uint32_t listStride = qsaListStride(layout);
  const uint64_t lanes = ExecutionLimits::maximumBatchWidth;
  Qwen4StepScratch scratch;
  scratch.streams =
      buffer(ops::HyperConnection::streamBytes(hs, rows), "qwen4-streams");
  scratch.normalized =
      buffer(rows * streams * sizeof(uint16_t), "qwen4-normalized");
  scratch.low =
      buffer(rows * layout.hyperRank * sizeof(uint16_t), "qwen4-low");
  scratch.gate = buffer(rows * streams * sizeof(uint16_t), "qwen4-gate");
  scratch.weights =
      buffer(rows * layout.hyperConnections * sizeof(float),
             "qwen4-weights");
  scratch.branch =
      buffer(rows * layout.hiddenSize * sizeof(uint16_t), "qwen4-branch");
  scratch.zeros =
      buffer(rows * layout.hiddenSize * sizeof(uint16_t), "qwen4-zeros");
  scratch.indexerQuery =
      buffer(rows * queryWidth * sizeof(float), "qwen4-indexer-query");
  scratch.indexerKey = buffer(
      rows * layout.indexerHeadDimension * sizeof(float), "qwen4-indexer-key");
  scratch.qsaQueries =
      buffer(rows * queryWidth * sizeof(uint16_t), "qwen4-qsa-queries");
  // A decode step keeps every attention layer's raw keys (its commit reads
  // them after the last); a prefill chunk's are the layer it runs.
  scratch.qsaKeys = buffer(
      (decode ? uint64_t{layout.attentionLayerCount()} * lanes *
                    ExecutionLimits::targetVerifyRows
              : rows) *
          layout.indexerHeadDimension * sizeof(uint16_t),
      "qwen4-qsa-keys");
  const uint64_t scoreRows =
      decode ? lanes * ExecutionLimits::targetVerifyRows
             : uint64_t{kPrefillScoreRows};
  scratch.qsaScores = buffer(scoreRows * qsaScoreStride(layout) * sizeof(float),
                             "qwen4-qsa-scores");
  scratch.qsaMask =
      buffer(rows * maskWords * sizeof(uint32_t), "qwen4-qsa-mask");
  scratch.qsaLists = buffer(
      (rows * (listStride / kv::kSparseTileRows) + lanes * listStride) *
          sizeof(uint32_t),
      "qwen4-qsa-lists");
  scratch.qsaListCounts =
      buffer((rows + lanes) * sizeof(uint32_t), "qwen4-qsa-list-counts");
  scratch.records = buffer(
      (decode ? lanes * kQsaLaneRecords
              : rows / BLOCK_INDEXER_BLOCK + lanes) *
          sizeof(BlockIndexerBlock),
      "qwen4-qsa-records");
  std::memset(scratch.zeros.contents(), 0, scratch.zeros.sizeBytes());
  return scratch;
}

// The qwen4exp layer encoder's per-pass prefill state, carried in
// PrefillStep::extension: each sequence's sparse attention plan (the
// block-list kernels, which take a zero-stride AttentionBlocks for rows
// that keep every block) and the record offset its blocks wrote into the
// scratch in beginPrefill.
struct Qwen4PrefillExtension final {
  std::vector<ops::PrefillAttentionPlan> sparseAttention;
  std::vector<uint32_t> recordOffsets;
};

// The same for a verify step: the lanes' sparse plan.
struct Qwen4VerifyExtension final {
  std::optional<ops::VerifyAttentionPlan> sparseAttention;
};

Qwen4PleScratch pleScratch(metal::MetalBackend &backend,
                           const TargetLayout &layout, uint64_t rows) {
  const auto buffer = [&](uint64_t bytes, const char *label) {
    return backend.allocateBuffer(bytes, metal::BufferStorage::Shared, label);
  };
  Qwen4PleScratch scratch;
  scratch.pleIndices = buffer(
      rows * layout.pleHeads() * sizeof(uint32_t), "qwen4-ple-indices");
  scratch.pleRows = buffer(rows * layout.pleWidth() * sizeof(uint16_t),
                           "qwen4-ple-rows");
  scratch.pleValue = buffer(rows * layout.hiddenSize * sizeof(uint16_t),
                            "qwen4-ple-value");
  scratch.pleGate = buffer(rows * layout.streamWidth() * sizeof(uint16_t),
                           "qwen4-ple-gate");
  scratch.pleGated = buffer(rows * layout.streamWidth() * sizeof(float),
                            "qwen4-ple-gated");
  scratch.pleConvolution =
      buffer(rows * layout.streamWidth() * sizeof(uint16_t),
             "qwen4-ple-convolution");
  return scratch;
}

} // namespace

Qwen4ExpWeights loadQwen4ExpWeights(metal::MetalBackend &backend,
                                    const TargetLayout &layout,
                                    const TargetFiles &files) {
  requireQwen4ExpLayout(layout);
  // The qwen4exp tensor set has no GGUF image plan or affine loader yet;
  // packed files only (the layout declares packedOnly, so the factory
  // never routes another source here).
  const auto *packed = std::get_if<PackedTargetFiles>(&files);
  if (!packed)
    throw WeightStoreError("qwen4exp loads from packed files only");
  const BlockTargetFormat format;
  const uint64_t allocationBaseline = backend.memoryStats().allocatedBytes;
  Qwen4ExpWeights result;
  result.layout = layout;
  result.layers.reserve(layout.layers);
  result.extras.reserve(layout.layers);
  for (uint32_t index = 0; index < layout.layers; ++index) {
    const bool fullAttention = layout.isFullAttentionLayer(index);
    WeightFile file = packed->layer(index);
    TargetLayerWeights &layer = result.layers.emplace_back();
    Qwen4ExpLayerExtras &extra = result.extras.emplace_back();
    extra.mixerHyper = readHyper(file, format, layout, true, "mixer-hyper");
    layer.mixer =
        readMixer(file, format, layout.mixerGeometry(), fullAttention);
    if (fullAttention) extra.indexer = readIndexer(file, format, layout);
    extra.ffnHyper = readHyper(file, format, layout, true, "ffn-hyper");
    layer.ffn = readMoeBlock(file);
    if (index == layout.pleLayer)
      extra.ple = readPle(file, format, layout);
    file.finish();
    result.files.push_back(file.record());
  }
  {
    WeightFile file = packed->head();
    result.finalHyper = readHyper(file, format, layout, false, "final-hyper");
    result.logitsProjection = format.projection(
        file, paddedVocabularySize(layout.vocabularySize), layout.hiddenSize,
        "logits");
    // bf16 logits would round near-ties together: their spacing is 0.125
    // at logits of 16 to 32.
    result.logitsProjection.destination = ops::FloatOutput::Float32;
    file.finish();
    result.files.push_back(file.record());
  }
  {
    WeightFile file = packed->embedding();
    result.tokenEmbedding =
        format.embedding(file, layout.vocabularySize, layout.hiddenSize);
    file.finish();
    result.files.push_back(file.record());
  }
  {
    WeightFile file = packed->images.load(
        packedImage(packed->directory / "ple-table.bin",
                    "target/ple-table.bin", kPleTableMagic, 0, 0));
    result.pleHash = readPleHash(file, layout, result);
    result.pleTable = readBlockEmbedding(
        file, static_cast<uint32_t>(result.pleHash.tableRows),
        layout.pleHeadDimension, "ple-table");
    file.finish();
    result.files.push_back(file.record());
  }
  result.manifestFingerprintSha256 = weightManifestFingerprint(result.files);
  result.actualAllocatedBytes = metal::allocationDelta(
      allocationBaseline, backend.memoryStats().allocatedBytes);
  return result;
}

TargetModelGeometry targetModelGeometry(const Qwen4ExpWeights &weights) {
  TargetModelGeometry geometry =
      targetModelGeometry(static_cast<const TargetModelWeights &>(weights));
  // The extras' projections the base pass does not see: both mixes' gates,
  // each indexer's projections, the PLE layer's and the final mix's.
  const auto include = [&](const ops::Projection &projection) {
    geometry.prefillProjections.push_back(projection.shape());
    geometry.decodeProjections.push_back(projection.shape());
  };
  for (const Qwen4ExpLayerExtras &extra : weights.extras) {
    include(extra.mixerHyper.down);
    include(extra.mixerHyper.up);
    if (extra.indexer) {
      include(extra.indexer->query);
      include(extra.indexer->key);
    }
    include(extra.ffnHyper.down);
    include(extra.ffnHyper.up);
    if (extra.ple) {
      include(extra.ple->key);
      include(extra.ple->value);
    }
  }
  include(weights.finalHyper.down);
  include(weights.finalHyper.up);
  for (auto *shapes :
       {&geometry.prefillProjections, &geometry.decodeProjections}) {
    std::sort(shapes->begin(), shapes->end());
    shapes->erase(std::unique(shapes->begin(), shapes->end()),
                  shapes->end());
  }
  return geometry;
}

// The qwen4exp layer encoder. The residual it encodes is hyperConnections
// fp32 streams per row, not the hidden rows every other target's steps
// hold, so the step buffers the family needs — the streams, a mix's
// normalized/low/gate/weights rows, the mixer's branch target, the PLE rows
// and the QSA score/mask/list scratch (upstream Qwen4Scratch) — the encoder
// allocates itself (Qwen4ExpTarget::addPrefillLayer/addVerify in
// runtime/qwen4/model is the call-order reference).
class Qwen4LayerEncoder final : public TargetLayerEncoder {
public:
  explicit Qwen4LayerEncoder(const TargetModel &model)
      : model_(model),
        weights_(static_cast<const Qwen4ExpWeights &>(*model.weights_)),
        auxiliaryOffset_(
            weights_.layout.gdnStateLayout().convolutionBytes() +
            weights_.layout.gdnStateLayout().recurrentBytes()) {
    const TargetLayout &layout = weights_.layout;
    prefill_ = stepScratch(model_.backend_, layout,
                           ExecutionLimits::prefillTokenBudget, false);
    verify_ = stepScratch(model_.backend_, layout,
                          uint64_t{ExecutionLimits::maximumBatchWidth} *
                              ExecutionLimits::targetVerifyRows,
                          true);
    prefillPle_ =
        pleScratch(model_.backend_, layout, ExecutionLimits::prefillTokenBudget);
    verifyPle_ = pleScratch(model_.backend_, layout,
                            uint64_t{ExecutionLimits::maximumBatchWidth} *
                                ExecutionLimits::targetVerifyRows);
    // The RoPE inverse frequencies the rope tables were made of
    // (PrefillArena fills its own the same way): the indexer's pool rotates
    // the pooled block keys with them.
    ropeInverseFrequencies_ = model_.backend_.allocateBuffer(
        uint64_t{layout.rotaryPairs} * sizeof(float),
        metal::BufferStorage::Shared, "qwen4-rope-frequencies");
    auto *frequencies =
        static_cast<float *>(ropeInverseFrequencies_.contents());
    if (!frequencies)
      throw std::logic_error("qwen4exp RoPE frequencies are not CPU-visible");
    for (uint32_t dim = 0; dim < layout.rotaryPairs; ++dim)
      frequencies[dim] = std::pow(
          layout.rotaryTheta, -static_cast<float>(dim) / layout.rotaryPairs);
  }

  // The chunk's PLE row ids, computed on the CPU by the packed prefill's
  // input preparation (qwen4ExpPleRows into each sequence's pleRowIds),
  // into the scratch the gather reads. Upstream also opens a pass with
  // HyperConnection::addInit — the embedding rows become stream 0 of each
  // row's streams and the rest zero — and writes the QSA block records;
  // those land with their ops.
  void beginPrefill(TargetModel::PrefillStep &step) const override {
    const uint32_t heads = weights_.layout.pleHeads();
    auto *indices =
        static_cast<uint32_t *>(prefillPle_.pleIndices.contents());
    if (!indices)
      throw std::logic_error("qwen4exp PLE row ids are not CPU-visible");
    for (const TargetModelPrefillSequence &sequence : step.sequences) {
      if (sequence.pleRowIds.size() != uint64_t{sequence.rows} * heads ||
          sequence.rowBegin + sequence.rows > step.rows)
        throw std::invalid_argument("invalid qwen4exp PLE row ids");
      std::memcpy(indices + uint64_t{sequence.rowBegin} * heads,
                  sequence.pleRowIds.data(),
                  sequence.pleRowIds.size() * sizeof(uint32_t));
    }
    // The streams start as the embedding's rows (upstream addPrefillBegin),
    // and the QSA block records the host computed into each sequence's
    // qsaBlocks land in the scratch (upstream preparePrefill's writeBlocks).
    ops::HyperConnection::addInit(step.graph, step.buffers.hidden[0],
                                  prefill_.streams, hyperShape(), step.rows);
    auto *records =
        static_cast<BlockIndexerBlock *>(prefill_.records.contents());
    if (!records)
      throw std::logic_error("qwen4exp QSA records are not CPU-visible");
    Qwen4PrefillExtension extension;
    const uint64_t capacity =
        prefill_.records.sizeBytes() / sizeof(BlockIndexerBlock);
    uint32_t recordOffset = 0;
    for (const TargetModelPrefillSequence &sequence : step.sequences) {
      const uint32_t count =
          qsaBlockCount(sequence.chunk.committed_tokens, sequence.rows);
      if (sequence.qsaBlocks.size() != count ||
          recordOffset + count > capacity)
        throw std::invalid_argument("invalid qwen4exp QSA block records");
      if (!sequence.indexTable)
        throw std::invalid_argument("qwen4exp prefill sequence lacks its index table");
      extension.recordOffsets.push_back(recordOffset);
      std::memcpy(records + recordOffset, sequence.qsaBlocks.data(),
                  count * sizeof(BlockIndexerBlock));
      recordOffset += count;
      extension.sparseAttention.push_back(ops::PagedAttention::prefillPlan(
          sequence.rows, model_.geometry_.attentionQueryHeads,
          model_.geometry_.layerKvLayout(false), /*sparse*/ true));
    }
    step.extension = std::move(extension);
  }

  void beginVerify(TargetModel::VerifyStep &step) const override {
    ops::HyperConnection::addInit(step.graph, step.buffers.hidden[0],
                                  verify_.streams, hyperShape(), step.rows);
    // The lanes' block records the host bound (their RoPE positions are its
    // to compute) move into the scratch; their first positions double as the
    // commit's (upstream prepareVerifyLane/verifyBlocks/verifyPositions).
    const uint32_t lanes = step.lanes;
    auto *records =
        static_cast<BlockIndexerBlock *>(verify_.records.contents());
    if (!records)
      throw std::logic_error("qwen4exp QSA records are not CPU-visible");
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      verifyFirstPositions_[lane] = step.chunks[lane].committed_tokens;
      const std::span<const BlockIndexerBlock> laneRecords =
          step.buffers.qsaBlocks.subspan(lane * kQsaLaneRecords,
                                         kQsaLaneRecords);
      const uint32_t count = qsaBlockCount(
          step.chunks[lane].committed_tokens, step.rowCapacity);
      if (laneRecords.size() < count)
        throw std::invalid_argument("invalid qwen4exp verify QSA records");
      std::memcpy(records + lane * kQsaLaneRecords, laneRecords.data(),
                  count * sizeof(BlockIndexerBlock));
      verifyBlockCounts_[lane] = count;
    }
    verifyLanes_ = lanes;
    verifyStride_ = step.rowCapacity;
    std::array<uint32_t, ExecutionLimits::maximumBatchWidth> histories{};
    for (uint32_t lane = 0; lane < lanes; ++lane)
      histories[lane] = step.chunks[lane].committed_tokens;
    Qwen4VerifyExtension extension;
    extension.sparseAttention.emplace(ops::PagedAttention::verifyPlan(
        lanes, model_.geometry_.attentionQueryHeads,
        model_.geometry_.layerKvLayout(false),
        std::span(histories).first(lanes), step.tree, step.liveNodes,
        step.rowCapacity, /*sparse*/ true));
    step.extension = std::move(extension);
  }

  // The encoder writes the capture layers' rows itself (the stream mean).
  [[nodiscard]] bool capturesHidden() const override { return true; }

  // The GPU's hash of the lanes' verify rows into PLE table ids
  // (ngram_embedding_hash): each row's token and the pleNgram - 1 tokens
  // before its lane's first row, which plePriorTokens carries (upstream
  // Qwen4Steps::prepareVerifyLane writes them on the CPU). The gather reads
  // pleIndices when the layer loop reaches pleLayer.
  void onVerifyInput(metal::CommandGraph &graph,
                     metal::MetalBuffer verifyInput,
                     metal::MetalBuffer plePriorTokens, uint32_t lanes,
                     uint32_t verifyStride) const override {
    const TargetLayout &layout = weights_.layout;
    const Qwen4PleHash &hash = weights_.pleHash;
    const uint32_t heads = layout.pleHeads();
    if (heads > NGRAM_EMBEDDING_MAXIMUM_HEADS ||
        hash.headVocabularies.size() != heads ||
        hash.headOffsets.size() != heads)
      throw std::logic_error("PLE hash geometry mismatch");
    if (!plePriorTokens || !lanes ||
        lanes > ExecutionLimits::maximumBatchWidth || !verifyStride ||
        verifyStride > ExecutionLimits::targetVerifyRows)
      throw std::invalid_argument("invalid qwen4exp PLE verify input");
    NgramHashParams params{};
    params.rows = lanes * verifyStride;
    params.ngram = layout.pleNgram;
    params.heads_per_ngram = layout.pleHeadsPerNgram;
    params.eos = weights_.pleEosToken;
    params.row_stride = verifyStride;
    std::copy(hash.multipliers.begin(), hash.multipliers.end(),
              params.multipliers);
    std::copy(hash.headVocabularies.begin(), hash.headVocabularies.end(),
              params.vocabularies);
    std::copy(hash.headOffsets.begin(), hash.headOffsets.end(),
              params.offsets);
    ops::NgramEmbedding::addHash(graph, std::move(verifyInput),
                                 std::move(plePriorTokens),
                                 verifyPle_.pleIndices, params, heads);
  }

  // Each lane's PLE history after its retained rows, from the convolution
  // inputs its verify's PLE block wrote (upstream
  // Qwen4ExpTarget::addStateCommit).
  void commitState(metal::CommandGraph &graph,
                   const TargetModelCommitBuffers &buffers, uint32_t lanes,
                   uint32_t verifyStride) const override {
    const ops::NgramShape shape = ngramShape();
    const uint64_t laneBytes =
        uint64_t{verifyStride} * shape.width() * sizeof(uint16_t);
    const uint64_t history = weights_.layout.pleHistoryBytes();
    for (uint32_t lane = 0; lane < lanes; ++lane)
      ops::NgramEmbedding::addCommit(
          graph,
          model_.backend_.view(verifyPle_.pleConvolution, lane * laneBytes,
                               laneBytes),
          model_.backend_.view(buffers.currentStates[lane], auxiliaryOffset_,
                               history),
          model_.backend_.view(buffers.nextStates[lane], auxiliaryOffset_,
                               history),
          model_.backend_.view(buffers.retainedCounts,
                               lane * sizeof(uint32_t), sizeof(uint32_t)),
          shape);
    // The QSA pending keys: each lane's last block's raw keys move to the
    // next cell's pending slots (upstream addStateCommit's
    // block_indexer_commit), from every attention layer's keys of the step
    // this command encoded.
    const uint32_t qsaLayers = weights_.layout.attentionLayerCount();
    if (qsaLayers) {
      const uint64_t auxiliary =
          weights_.layout.gdnStateLayout().auxiliaryBytes;
      std::array<metal::MetalBuffer, ExecutionLimits::maximumBatchWidth>
          pendingIn{}, pendingOut{};
      for (uint32_t lane = 0; lane < lanes; ++lane) {
        pendingIn[lane] = qsaPending(
            model_.backend_.view(buffers.currentStates[lane],
                                 auxiliaryOffset_, auxiliary),
            0, qsaLayers);
        pendingOut[lane] = qsaPending(
            model_.backend_.view(buffers.nextStates[lane],
                                 auxiliaryOffset_, auxiliary),
            0, qsaLayers);
      }
      ops::BlockIndexer::addCommit(
          graph, verify_.qsaKeys, qsaLayers,
          uint64_t{ExecutionLimits::maximumBatchWidth} *
              ExecutionLimits::targetVerifyRows,
          verifyStride,
          std::span(verifyFirstPositions_).first(lanes),
          buffers.retainedCounts, pendingIn, pendingOut);
    }
  }

  // One layer of a prefill chunk, in upstream order: the PLE block on
  // pleLayer, the mixer mix, the mixer (a GDN or a QSA attention), the
  // combine, the FFN mix, the MoE block's routing and routed work, the
  // combine and the stream-mean capture the draft reads.
  metal::MetalBuffer addPrefillLayer(TargetModel::PrefillStep &step,
                                     uint32_t index,
                                     metal::MetalBuffer /*input*/,
                                     metal::MetalBuffer output) const override {
    const Qwen4ExpLayerExtras &extra = weights_.extras[index];
    const TargetLayerWeights &layer = weights_.layers[index];
    if (extra.ple) addPrefillPle(step, *extra.ple, prefill_.streams);
    addPrefillMix(step, extra.mixerHyper);
    if (std::holds_alternative<GdnMixerWeights>(layer.mixer))
      addGdnMixer(step, std::get<GdnMixerWeights>(layer.mixer));
    else
      addPrefillQsa(step, *extra.indexer,
                    std::get<AttentionMixerWeights>(layer.mixer));
    ops::HyperConnection::addCombine(step.graph, prefill_.streams,
                                     prefill_.branch, prefill_.weights,
                                     hyperShape(), step.rows);
    addPrefillMix(step, extra.ffnHyper);
    addMoeBlock(step, layer);
    ops::HyperConnection::addCombine(step.graph, prefill_.streams,
                                     prefill_.branch, prefill_.weights,
                                     hyperShape(), step.rows);
    addPrefillCapture(step, index);
    return output;
  }

  // The same layer in a verify step: upstream folds the mixes' low-rank
  // projections into the decode tables, routes the verify rows alone and
  // commits the PLE history and QSA pending keys with the rest of the
  // state (Qwen4ExpTarget::addVerify/addStateCommit).
  void addVerifyLayer(TargetModel::VerifyStep &step, uint32_t index,
                      metal::MetalBuffer /*input*/,
                      metal::MetalBuffer /*output*/) const override {
    const Qwen4ExpLayerExtras &extra = weights_.extras[index];
    const TargetLayerWeights &layer = weights_.layers[index];
    if (extra.ple) addVerifyPle(step, *extra.ple, verify_.streams,
                                step.rowCapacity);
    const ops::Projection *inputProjection = nullptr;
    std::visit([&](const auto &mixer) { inputProjection = &mixer.inputProjection; },
               layer.mixer);
    const ops::PreparedInput mixInput =
        addVerifyMix(step, extra.mixerHyper, inputProjection);
    if (std::holds_alternative<GdnMixerWeights>(layer.mixer))
      addGdnMixer(step, std::get<GdnMixerWeights>(layer.mixer), mixInput);
    else
      addVerifyQsa(step, *extra.indexer,
                   std::get<AttentionMixerWeights>(layer.mixer), mixInput);
    ops::HyperConnection::addCombine(step.graph, verify_.streams,
                                     verify_.branch, verify_.weights,
                                     hyperShape(), step.rows);
    addVerifyMix(step, extra.ffnHyper, nullptr);
    addMoeBlock(step, layer);
    ops::HyperConnection::addCombine(step.graph, verify_.streams,
                                     verify_.branch, verify_.weights,
                                     hyperShape(), step.rows);
    addVerifyCapture(step, index);
  }

private:
  // The shape of the PLE gate and convolution: hyperConnections streams of
  // hiddenSize, pleConvolutionTaps taps pleNgram rows apart.
  [[nodiscard]] ops::NgramShape ngramShape() const {
    const TargetLayout &layout = weights_.layout;
    return {layout.hiddenSize, layout.hyperConnections,
            layout.pleConvolutionTaps, layout.pleNgram};
  }

  // The PLE block before layer pleLayer's mixer mix: gather each row's
  // n-gram table rows (ops::Embedding over the ids beginPrefill copied),
  // project key and value, gate the value per stream (ngram_embedding_gate)
  // and add it through the dilated convolution over each sequence's stream
  // history (ngram_embedding_conv, the cell's auxiliaryIn/auxiliaryOut
  // region). `streams` is the fp32 hyper-connected residual scratch; the
  // layer call sites bind it once the hyper-connection mix lands. Upstream
  // also captures the history mid-chunk for a state capture inside the
  // command (ngram_embedding_history); richengine's captures carry no
  // auxiliary state yet.
  void addPrefillPle(TargetModel::PrefillStep &step,
                     const Qwen4PleWeights &ple,
                     const metal::MetalBuffer &streams) const {
    const Qwen4PleScratch &h = prefillPle_;
    const ops::NgramShape ps = ngramShape();
    const ops::Linear &linear = model_.operators_.linear();
    const TargetModelPrefillBuffers &b = step.buffers;
    const uint32_t rows = step.rows;
    ops::Embedding::add(step.graph, h.pleIndices, weights_.pleTable,
                        h.pleRows, rows * weights_.layout.pleHeads());
    linear.addPrefill(step.graph, h.pleRows, ple.key, h.pleGate,
                      b.projectionSums, rows, b.linearScratch);
    linear.addPrefill(step.graph, h.pleRows, ple.value, h.pleValue,
                      b.projectionSums, rows, b.linearScratch);
    ops::NgramEmbedding::addGate(step.graph, h.pleGate, h.pleValue, streams,
                                 ple.keyNorm, ple.queryNorm,
                                 ple.convolutionNorm, h.pleGated,
                                 h.pleConvolution, ps, rows);
    const uint32_t width = ps.width();
    for (const TargetModelPrefillSequence &sequence : step.sequences) {
      if (!sequence.auxiliaryIn)
        throw std::logic_error(
            "qwen4exp prefill sequence lacks its PLE history state");
      ops::NgramEmbedding::addConvolution(
          step.graph,
          rowsOf<uint16_t>(model_.backend_, h.pleConvolution,
                           sequence.rowBegin, sequence.rows, width),
          sequence.auxiliaryIn, sequence.auxiliaryOut, ple.convolution,
          rowsOf<float>(model_.backend_, h.pleGated, sequence.rowBegin,
                        sequence.rows, width),
          rowsOf<float>(model_.backend_, streams, sequence.rowBegin,
                        sequence.rows, width),
          ps, sequence.rows);
    }
  }

  // The same block in a verify step: the ids are the GPU hash's
  // (onVerifyInput), each lane's rows follow its cell's PLE history and
  // commitState writes the history after its retained rows.
  void addVerifyPle(TargetModel::VerifyStep &step,
                    const Qwen4PleWeights &ple,
                    const metal::MetalBuffer &streams,
                    uint32_t verifyStride) const {
    const Qwen4PleScratch &h = verifyPle_;
    const ops::NgramShape ps = ngramShape();
    const ops::Linear &linear = model_.operators_.linear();
    const TargetModelVerifyBuffers &b = step.buffers;
    const uint32_t rows = step.rows;
    ops::Embedding::add(step.graph, h.pleIndices, weights_.pleTable,
                        h.pleRows, rows * weights_.layout.pleHeads());
    const ops::LinearPlan keyPlan =
        linear.decodePlan(ple.key, step.planLanes, ops::LinearEpilogue::None,
                          nullptr, step.rowCapacity);
    linear.add(step.graph,
               {.input = h.pleRows, .output = h.pleGate,
                .scratch = b.linearScratch},
               ple.key, keyPlan);
    const ops::LinearPlan valuePlan =
        linear.decodePlan(ple.value, step.planLanes, ops::LinearEpilogue::None,
                          nullptr, step.rowCapacity);
    linear.add(step.graph,
               {.input = h.pleRows, .output = h.pleValue,
                .scratch = b.linearScratch},
               ple.value, valuePlan);
    ops::NgramEmbedding::addGate(step.graph, h.pleGate, h.pleValue, streams,
                                 ple.keyNorm, ple.queryNorm,
                                 ple.convolutionNorm, h.pleGated,
                                 h.pleConvolution, ps, rows);
    const uint32_t width = ps.width();
    const uint64_t history = weights_.layout.pleHistoryBytes();
    for (uint32_t lane = 0; lane < step.lanes; ++lane)
      ops::NgramEmbedding::addConvolution(
          step.graph,
          rowsOf<uint16_t>(model_.backend_, h.pleConvolution,
                           lane * verifyStride, verifyStride, width),
          model_.backend_.view(b.currentGdnStates[lane], auxiliaryOffset_,
                               history),
          {}, ple.convolution,
          rowsOf<float>(model_.backend_, h.pleGated, lane * verifyStride,
                        verifyStride, width),
          rowsOf<float>(model_.backend_, streams, lane * verifyStride,
                        verifyStride, width),
          ps, verifyStride);
  }

  [[nodiscard]] ops::HyperShape hyperShape() const {
    const TargetLayout &layout = weights_.layout;
    return {layout.hiddenSize, layout.hyperConnections, layout.hyperRank};
  }

  // The streams' mix of a hyper block (upstream addPrefillMix): the norm
  // (which also emits the injection weights the combine reads), the
  // low-rank gate's down/silu/up and the gated stream mean into the
  // block's input, `output` — the shared normalized buffer.
  void addPrefillMix(TargetModel::PrefillStep &step,
                     const Qwen4HyperWeights &hyper) const {
    const TargetModelPrefillBuffers &b = step.buffers;
    const ops::Linear &linear = model_.operators_.linear();
    const ops::HyperShape hs = hyperShape();
    ops::HyperConnection::addNorm(step.graph, prefill_.streams, hyper.norm,
                                  hyper.inject, prefill_.normalized,
                                  prefill_.weights, hs, step.rows);
    linear.addPrefill(step.graph, prefill_.normalized, hyper.down,
                      prefill_.low, b.projectionSums, step.rows,
                      b.linearScratch);
    ops::HyperConnection::addLow(step.graph, prefill_.low, hs, step.rows);
    linear.addPrefill(step.graph, prefill_.low, hyper.up, prefill_.gate,
                      b.projectionSums, step.rows, b.linearScratch);
    ops::HyperConnection::addMix(step.graph, prefill_.normalized,
                                 prefill_.gate, b.normalized, hs, step.rows);
  }

  // The same in a verify step (upstream addVerifyMix): the returned
  // PreparedInput is the consumer projection's input table when it takes
  // one — `next`, the mix's block's input projection.
  ops::PreparedInput addVerifyMix(TargetModel::VerifyStep &step,
                                  const Qwen4HyperWeights &hyper,
                                  const ops::Projection *next) const {
    const TargetModelVerifyBuffers &b = step.buffers;
    const ops::Linear &linear = model_.operators_.linear();
    const ops::LinearScratch scratch = b.linearScratch;
    const ops::HyperShape hs = hyperShape();
    ops::HyperConnection::addNorm(step.graph, verify_.streams, hyper.norm,
                                  hyper.inject, verify_.normalized,
                                  verify_.weights, hs, step.rows);
    const ops::LinearPlan downPlan =
        linear.decodePlan(hyper.down, step.planLanes,
                          ops::LinearEpilogue::None, nullptr, step.rowCapacity);
    linear.add(step.graph,
               {.input = verify_.normalized, .output = verify_.low,
                .scratch = scratch},
               hyper.down, downPlan);
    const ops::LinearPlan upPlan =
        linear.decodePlan(hyper.up, step.planLanes,
                          ops::LinearEpilogue::None, nullptr, step.rowCapacity);
    const ops::PreparedInput low = ops::HyperConnection::addLow(
        step.graph, verify_.low, hs, step.rows, scratch, upPlan.input());
    linear.add(step.graph,
               {.input = verify_.low, .output = verify_.gate,
                .scratch = scratch, .prepared = low},
               hyper.up, upPlan);
    return ops::HyperConnection::addMix(
        step.graph, verify_.normalized, verify_.gate, b.normalized, hs,
        step.rows, scratch,
        next ? linear.decodePlan(*next, step.planLanes,
                                 ops::LinearEpilogue::None, nullptr,
                                 step.rowCapacity)
                  .input()
             : ops::LinearInput::Plain);
  }

  // A state cell's auxiliary region, of which the PLE history takes the
  // first pleHistoryBytes and each attention layer's pending keys
  // perLayer bytes after it (upstream Qwen4Steps::qsaPending).
  [[nodiscard]] metal::MetalBuffer
  qsaPending(const metal::MetalBuffer &auxiliary, uint32_t layer,
             uint32_t layers = 1) const {
    const uint64_t perLayer =
        uint64_t{BLOCK_INDEXER_PENDING} * BLOCK_INDEXER_DIMENSION *
        sizeof(uint16_t);
    return model_.backend_.view(
        auxiliary, weights_.layout.pleHistoryBytes() + uint64_t{layer} * perLayer,
        perLayer * layers);
  }

  // The indexer's fp32 projections of the mix's rows (upstream
  // addIndexerProjections): one float segment each.
  void addIndexerProjections(metal::CommandGraph &graph,
                             const Qwen4IndexerWeights &indexer,
                             metal::MetalBuffer input, metal::MetalBuffer query,
                             metal::MetalBuffer key, uint32_t rows) const {
    const ops::Linear &linear = model_.operators_.linear();
    const auto one = [&](const ops::Projection &weights,
                         metal::MetalBuffer output) {
      const auto &segments = weights.blocks().segments;
      if (segments.size() != 1 || !segments.front().isFloat())
        throw std::invalid_argument(
            "the QSA indexer projections are single float segments");
      ops::addGgufFloat(graph, input, segments.front(), std::move(output),
                        rows, weights.outputSize, 0,
                        ops::FloatOutput::Float32,
                        linear.ggufFloatTile(rows, weights.outputSize));
    };
    one(indexer.query, std::move(query));
    one(indexer.key, std::move(key));
  }

  // A full-attention layer's QSA in a prefill chunk (upstream
  // addPrefillQsa plus the attention body of its addPrefillMixer): each
  // sequence is a dispatch of one lane — project, pool, pending and, past
  // indexerTopBlocks complete blocks, select and list its blocks — then
  // the gated QKV attention writes the branch rows.
  void addPrefillQsa(TargetModel::PrefillStep &step,
                     const Qwen4IndexerWeights &indexer,
                     const AttentionMixerWeights &mixer) const {
    const TargetModelPrefillBuffers &b = step.buffers;
    const Qwen4StepScratch &h = prefill_;
    const auto &ext =
        std::any_cast<const Qwen4PrefillExtension &>(step.extension);
    const ops::Linear &linear = model_.operators_.linear();
    const TargetLayout &layout = weights_.layout;
    const TargetModelGeometry &geometry = model_.geometry_;
    const uint32_t layer = step.attentionLayer++;
    const uint32_t topBlocks = layout.indexerTopBlocks;
    const uint32_t keyWidth = layout.indexerHeadDimension;
    const uint32_t queryWidth = layout.indexerHeads * keyWidth;
    const uint32_t maskWords = qsaMaskWords(layout);
    const uint32_t scoreStride = qsaScoreStride(layout);
    const uint32_t listStride = qsaListStride(layout);
    // The layer's index offset in every index entry's plane: layer *
    // extentPages * a page's index bytes (kv::PageStorage::indexOffset).
    const uint64_t indexPageBytes = uint64_t{kv::kPageTokens} /
                                    BLOCK_INDEXER_BLOCK * keyWidth * 2;
    const uint32_t indexOffset = static_cast<uint32_t>(
        uint64_t{layer} * step.kvLayers[layer].extent_pages *
        indexPageBytes);
    const auto u16 = [&](const metal::MetalBuffer &buffer, uint32_t begin,
                         uint32_t count, uint32_t width) {
      return rowsOf<uint16_t>(model_.backend_, buffer, begin, count, width);
    };
    const auto f32 = [&](const metal::MetalBuffer &buffer, uint32_t begin,
                         uint32_t count, uint32_t width) {
      return rowsOf<float>(model_.backend_, buffer, begin, count, width);
    };
    const auto u32s = [&](const metal::MetalBuffer &buffer, uint32_t begin,
                          uint32_t count, uint32_t width) {
      return rowsOf<uint32_t>(model_.backend_, buffer, begin, count, width);
    };
    std::vector<uint32_t> firstPositions;
    for (const TargetModelPrefillSequence &sequence : step.sequences)
      firstPositions.push_back(
          static_cast<uint32_t>(sequence.chunk.committed_tokens));
    const auto qsaRows = [&](size_t at) {
      const TargetModelPrefillSequence &sequence = step.sequences[at];
      return ops::IndexerRows{sequence.rows,
                              std::span(firstPositions).subspan(at, 1),
                              {{&sequence.indexTable, 1}, indexOffset},
                              topBlocks, maskWords, scoreStride};
    };
    addIndexerProjections(step.graph, indexer, b.normalized, h.indexerQuery,
                          h.indexerKey, step.rows);
    std::vector<ops::AttentionBlocks> blocks(step.sequences.size());
    std::vector<ops::BlockListRows> sparseRows;
    std::vector<uint32_t> counts(step.sequences.size());
    uint32_t lists = 0;
    for (size_t at = 0; at < step.sequences.size(); ++at) {
      const TargetModelPrefillSequence &sequence = step.sequences[at];
      const metal::MetalBuffer keys =
          u16(h.qsaKeys, sequence.rowBegin, sequence.rows, keyWidth);
      const metal::MetalBuffer pending = qsaPending(sequence.auxiliaryIn, layer);
      counts[at] =
          qsaBlockCount(sequence.chunk.committed_tokens, sequence.rows);
      const metal::MetalBuffer records = model_.backend_.view(
          h.records, uint64_t{ext.recordOffsets[at]} * sizeof(BlockIndexerBlock),
          uint64_t{counts[at]} * sizeof(BlockIndexerBlock));
      ops::BlockIndexer::addProject(
          step.graph, f32(h.indexerQuery, sequence.rowBegin, sequence.rows,
                          queryWidth),
          f32(h.indexerKey, sequence.rowBegin, sequence.rows, keyWidth),
          indexer.queryNorm, f32(b.ropeCos, sequence.rowBegin, sequence.rows,
                                 geometry.rotaryPairs),
          f32(b.ropeSin, sequence.rowBegin, sequence.rows,
              geometry.rotaryPairs),
          u16(h.qsaQueries, sequence.rowBegin, sequence.rows, queryWidth),
          keys, qsaRows(at));
      ops::BlockIndexer::addPool(
          step.graph,
          {records, std::span(counts).subspan(at, 1), 0, keys, {&pending, 1}},
          indexer.keyNorm, ropeInverseFrequencies_, 1.0F, qsaRows(at));
      ops::BlockIndexer::addPending(step.graph, keys, pending,
                                    qsaPending(sequence.auxiliaryOut, layer),
                                    firstPositions[at], sequence.rows);
      if (!ops::BlockIndexer::sparse(firstPositions[at], sequence.rows,
                                     topBlocks))
        continue;
      ops::BlockIndexer::addSelect(
          step.graph,
          u16(h.qsaQueries, sequence.rowBegin, sequence.rows, queryWidth),
          h.qsaScores, kPrefillScoreRows,
          u32s(h.qsaMask, sequence.rowBegin, sequence.rows, maskWords),
          qsaRows(at));
      blocks[at] = {h.qsaLists, h.qsaListCounts, listStride, lists};
      sparseRows.push_back(
          {sequence.rowBegin, sequence.rows, firstPositions[at], lists});
      lists += kv::prefillAttentionTiles(sequence.rows);
    }
    if (!sparseRows.empty())
      ops::PagedAttention::addBlockLists(
          step.graph, h.qsaMask, maskWords, sparseRows,
          {h.qsaLists, h.qsaListCounts, listStride, 0});
    // The attention itself: QKV projection, store, sparse or full sweep
    // per sequence, the query gate with the out-projection's sums, then
    // the output projection into the branch (the combine adds it).
    linear.addPrefill(step.graph, b.normalized, mixer.inputProjection,
                      b.fullPacked, b.projectionSums, step.rows,
                      b.linearScratch);
    for (size_t at = 0; at < step.sequences.size(); ++at) {
      const TargetModelPrefillSequence &sequence = step.sequences[at];
      const uint64_t headBytes = uint64_t{sequence.attentionStride} *
                                 geometry.attentionHeadDimension *
                                 sizeof(uint16_t);
      const metal::MetalBuffer queries = model_.backend_.view(
          b.fullQueries, sequence.queryOffset,
          uint64_t{geometry.attentionQueryHeads} * headBytes);
      const metal::MetalBuffer attentionRows = model_.backend_.view(
          b.fullAttention, sequence.queryOffset,
          uint64_t{geometry.attentionQueryHeads} * headBytes);
      const metal::MetalBuffer keys = model_.backend_.view(
          b.chunkKeys, sequence.kvOffset,
          uint64_t{geometry.attentionKvHeads} * headBytes);
      const metal::MetalBuffer values = model_.backend_.view(
          b.chunkValues, sequence.kvOffset,
          uint64_t{geometry.attentionKvHeads} * headBytes);
      ops::PagedAttention::addPrefillProjection(
          step.graph, u16(b.fullPacked, sequence.rowBegin, sequence.rows,
                          geometry.packedFullWidth),
          mixer.queryNorm, mixer.keyNorm,
          f32(b.ropeCos, sequence.rowBegin, sequence.rows,
              geometry.rotaryPairs),
          f32(b.ropeSin, sequence.rowBegin, sequence.rows,
              geometry.rotaryPairs),
          queries, keys, values, sequence.rows, sequence.attentionStride,
          geometry.attentionQueryHeads, geometry.kvLayout);
      ops::PagedAttention::addPrefillStore(
          step.graph, step.kvLayers[layer], keys, values, sequence.pageTable,
          sequence.chunk, geometry.kvLayout);
      ops::PagedAttention::addPrefill(
          step.graph, step.kvLayers[layer], queries, attentionRows,
          b.attentionPartials, b.attentionStatistics, sequence.pageTable,
          sequence.chunk, ext.sparseAttention[at], blocks[at]);
      ops::PagedAttention::addPrefillGateSums(
          step.graph, u16(b.fullPacked, sequence.rowBegin, sequence.rows,
                          geometry.packedFullWidth),
          attentionRows,
          u16(b.attentionHidden, sequence.rowBegin, sequence.rows,
              geometry.attentionWidth),
          prefillSums(model_.backend_, b.projectionSums, sequence.rowBegin,
                      sequence.rows, geometry.attentionWidth),
          sequence.rows, sequence.attentionStride,
          geometry.attentionQueryHeads, geometry.kvLayout);
    }
    linear.addPrefill(step.graph, b.attentionHidden, mixer.outputProjection,
                      h.branch, b.projectionSums, step.rows, b.linearScratch);
  }

  // Every lane's QSA in one dispatch of each kernel (upstream addVerifyQsa
  // plus its verify mixer's attention body). Once a lane's rows attend
  // sparsely every lane's attends through its list.
  void addVerifyQsa(TargetModel::VerifyStep &step,
                    const Qwen4IndexerWeights &indexer,
                    const AttentionMixerWeights &mixer,
                    const ops::PreparedInput &mixInput) const {
    const TargetModelVerifyBuffers &b = step.buffers;
    const Qwen4StepScratch &h = verify_;
    const auto &ext =
        std::any_cast<const Qwen4VerifyExtension &>(step.extension);
    const ops::Linear &linear = model_.operators_.linear();
    const TargetLayout &layout = weights_.layout;
    const TargetModelGeometry &geometry = model_.geometry_;
    const uint32_t layer = step.attentionLayer++;
    const uint32_t topBlocks = layout.indexerTopBlocks;
    const uint32_t keyWidth = layout.indexerHeadDimension;
    const uint32_t queryWidth = layout.indexerHeads * keyWidth;
    const uint32_t maskWords = qsaMaskWords(layout);
    const uint32_t scoreStride = qsaScoreStride(layout);
    const uint32_t listStride = qsaListStride(layout);
    const uint64_t indexPageBytes = uint64_t{kv::kPageTokens} /
                                    BLOCK_INDEXER_BLOCK * keyWidth * 2;
    const uint32_t indexOffset = static_cast<uint32_t>(
        uint64_t{layer} * step.kvLayers[layer].extent_pages *
        indexPageBytes);
    const uint32_t stride = step.rowCapacity;
    const uint64_t layerRows = uint64_t{ExecutionLimits::maximumBatchWidth} *
                               ExecutionLimits::targetVerifyRows;
    addIndexerProjections(step.graph, indexer, b.normalized, h.indexerQuery,
                          h.indexerKey, step.rows);
    std::array<uint32_t, ExecutionLimits::maximumBatchWidth> firstPositions{};
    bool sparse = false;
    for (uint32_t lane = 0; lane < step.lanes; ++lane) {
      firstPositions[lane] = step.chunks[lane].committed_tokens;
      sparse = sparse || ops::BlockIndexer::sparse(firstPositions[lane],
                                                 stride, topBlocks);
    }
    const ops::IndexerRows qsa{stride,
                               std::span(firstPositions).first(step.lanes),
                               {std::span(b.indexTables).first(step.lanes),
                                indexOffset},
                               topBlocks, maskWords, scoreStride};
    const metal::MetalBuffer keys = rowsOf<uint16_t>(
        model_.backend_, h.qsaKeys, layer * layerRows, step.rows, keyWidth);
    std::array<metal::MetalBuffer, ExecutionLimits::maximumBatchWidth> pending;
    for (uint32_t lane = 0; lane < step.lanes; ++lane)
      pending[lane] = qsaPending(
          model_.backend_.view(b.currentGdnStates[lane], auxiliaryOffset_,
                               weights_.layout.gdnStateLayout().auxiliaryBytes),
          layer);
    ops::BlockIndexer::addProject(
        step.graph,
        rowsOf<float>(model_.backend_, h.indexerQuery, 0, step.rows,
                      queryWidth),
        rowsOf<float>(model_.backend_, h.indexerKey, 0, step.rows, keyWidth),
        indexer.queryNorm,
        rowsOf<float>(model_.backend_, b.ropeCos, 0, step.rows,
                      geometry.rotaryPairs),
        rowsOf<float>(model_.backend_, b.ropeSin, 0, step.rows,
                      geometry.rotaryPairs),
        rowsOf<uint16_t>(model_.backend_, h.qsaQueries, 0, step.rows,
                         queryWidth),
        keys, qsa);
    ops::BlockIndexer::addPool(
        step.graph,
        {h.records, std::span(verifyBlockCounts_).first(step.lanes),
         kQsaLaneRecords, keys, std::span(pending).first(step.lanes)},
        indexer.keyNorm, ropeInverseFrequencies_, 1.0F, qsa);
    ops::AttentionBlocks blocks{h.qsaLists, h.qsaListCounts, 0, 0};
    if (sparse) {
      blocks.stride = listStride;
      ops::BlockIndexer::addSelect(
          step.graph,
          rowsOf<uint16_t>(model_.backend_, h.qsaQueries, 0, step.rows,
                           queryWidth),
          h.qsaScores, step.rows,
          rowsOf<uint32_t>(model_.backend_, h.qsaMask, 0, step.rows,
                           maskWords),
          qsa);
      std::array<ops::BlockListRows, ExecutionLimits::maximumBatchWidth>
          laneLists{};
      for (uint32_t lane = 0; lane < step.lanes; ++lane)
        laneLists[lane] = {lane * stride, stride, firstPositions[lane], lane};
      ops::PagedAttention::addBlockLists(
          step.graph, h.qsaMask, maskWords,
          std::span(laneLists).first(step.lanes), blocks);
    }
    // The attention body (upstream's verify mixer): the input projection,
    // the lanes' chunk stores, the sparse or dense reduce into the
    // attention rows, the gate, and the out-projection into the branch.
    step.graph.beginBakedSpan();
    const ops::LinearPlan inputPlan =
        linear.decodePlan(mixer.inputProjection, step.planLanes,
                          ops::LinearEpilogue::None, nullptr, step.rowCapacity);
    linear.add(step.graph,
               {.input = b.normalized, .output = b.fullPacked,
                .scratch = b.linearScratch, .prepared = mixInput},
               mixer.inputProjection, inputPlan);
    ops::PagedAttention::addVerifyProjection(
        step.graph, b.fullPacked, mixer.queryNorm, mixer.keyNorm, b.ropeCos,
        b.ropeSin, b.fullQueries, b.chunkKeys[layer], b.chunkValues[layer],
        geometry.attentionQueryHeads, geometry.kvLayout, step.lanes,
        step.rowCapacity);
    const ops::LinearPlan outputPlan =
        linear.decodePlan(mixer.outputProjection, step.planLanes,
                          ops::LinearEpilogue::None, nullptr, step.rowCapacity);
    ops::PagedAttention::addVerify(
        step.graph, step.kvLayers[layer],
        {b.chunkKeys[layer], b.chunkValues[layer], b.fullQueries,
         b.attentionPartials, b.attentionStatistics, b.fullAttention,
         b.pageTables, b.treeMasks},
        step.chunks, *ext.sparseAttention, {}, {}, blocks);
    const ops::PreparedInput hidden = ops::PagedAttention::addVerifyGate(
        step.graph, b.fullPacked, b.fullAttention, b.attentionHidden,
        geometry.attentionQueryHeads, geometry.kvLayout, step.lanes,
        b.linearScratch, outputPlan.input(), true, step.rowCapacity);
    linear.add(step.graph,
               {.input = b.attentionHidden, .output = h.branch,
                .scratch = b.linearScratch, .prepared = hidden},
               mixer.outputProjection, outputPlan);
    step.graph.endBakedSpan();
  }

  // The GDN mixer over the mix's rows (upstream addPrefillMixer's
  // QwenGdnWeights branch; the standard hybrid path is
  // TargetModel::addPrefillMixer): the input projection, each sequence's
  // recurrent pass — the gate phase's GDN_SIGMOID_GATE variant (upstream
  // GDN.h flag 2), which this family's packed rows need — and the
  // out-projection into the branch the combine injects into the streams.
  void addGdnMixer(TargetModel::PrefillStep &step,
                   const GdnMixerWeights &mixer) const {
    const TargetModelPrefillBuffers &b = step.buffers;
    const TargetModelGeometry &geometry = model_.geometry_;
    const ops::Linear &linear = model_.operators_.linear();
    const uint32_t layer = step.gdnLayer++;
    linear.addPrefill(step.graph, b.normalized, mixer.inputProjection,
                      b.gdnPacked, b.projectionSums, step.rows,
                      b.linearScratch);
    for (const TargetModelPrefillSequence &sequence : step.sequences) {
      const auto u16 = [&](const metal::MetalBuffer &buffer, uint32_t width) {
        return rowsOf<uint16_t>(model_.backend_, buffer, sequence.rowBegin,
                                sequence.rows, width);
      };
      const auto f32 = [&](const metal::MetalBuffer &buffer, uint32_t width) {
        return rowsOf<float>(model_.backend_, buffer, sequence.rowBegin,
                             sequence.rows, width);
      };
      ops::GDN::addPrefill(
          step.graph,
          {u16(b.gdnPacked, geometry.packedGdnWidth),
           mixer.convolutionWeights, sequence.convolutionIn[layer],
           sequence.convolutionOut[layer],
           u16(b.gdnQueries, geometry.gdnKeyWidth()),
           u16(b.gdnKeys, geometry.gdnKeyWidth()),
           u16(b.gdnValues, geometry.attentionWidth), mixer.decay,
           mixer.timeBias, f32(b.gdnDecay, geometry.gdnValueHeads),
           u16(b.gdnBeta, geometry.gdnValueHeads),
           sequence.recurrentIn[layer], sequence.recurrentOut[layer],
           u16(b.recurrent, geometry.attentionWidth), mixer.mixerNorm,
           u16(b.gdnHidden, geometry.attentionWidth), b.gdnChunkScratch},
          geometry.gdnShape(), sequence.rows, mixer.outputHeadOrder,
          prefillSums(model_.backend_, b.projectionSums, sequence.rowBegin,
                      sequence.rows, geometry.attentionWidth),
          GDN_SIGMOID_GATE);
    }
    linear.addPrefill(step.graph, b.gdnHidden, mixer.outputProjection,
                      prefill_.branch, b.projectionSums, step.rows,
                      b.linearScratch);
  }

  // The same mixer in a verify step (upstream addVerifyMixer's
  // QwenGdnWeights branch; TargetModel::addVerifyMixer is the standard
  // path): the input projection reads the mix's prepared table, the fused
  // decode carries the same sigmoid flag, and the out-projection's branch
  // rows take the combine's injection instead of a residual epilogue.
  void addGdnMixer(TargetModel::VerifyStep &step,
                   const GdnMixerWeights &mixer,
                   const ops::PreparedInput &mixInput) const {
    const TargetModelVerifyBuffers &b = step.buffers;
    const TargetModelGeometry &geometry = model_.geometry_;
    const ops::Linear &linear = model_.operators_.linear();
    const uint32_t layer = step.gdnLayer++;
    const ops::GdnStateStrides state{
        geometry.stateLayout.convolutionLayerBytes(),
        geometry.stateLayout.recurrentLayerBytes(),
        geometry.stateLayout.convolutionBytes()};
    step.graph.beginBakedSpan();
    const ops::LinearPlan inputPlan =
        linear.decodePlan(mixer.inputProjection, step.planLanes,
                          ops::LinearEpilogue::None, nullptr,
                          step.rowCapacity);
    linear.add(step.graph,
               {.input = b.normalized, .output = b.gdnPacked[layer],
                .scratch = b.linearScratch, .prepared = mixInput},
               mixer.inputProjection, inputPlan);
    const ops::LinearPlan outputPlan =
        linear.decodePlan(mixer.outputProjection, step.planLanes,
                          ops::LinearEpilogue::None, nullptr,
                          step.rowCapacity);
    const ops::PreparedInput hidden =
        step.tree
            ? ops::GDN::addDecodeTree(
                  step.graph,
                  {b.gdnPacked[layer], mixer.convolutionWeights,
                   b.currentGdnStates, b.nextGdnStates, b.gdnMixed[layer],
                   mixer.decay, mixer.timeBias, b.gdnDecay[layer],
                   b.gdnBeta[layer], mixer.mixerNorm, b.gdnHidden,
                   b.linearScratch, b.gdnTape},
                  b.treeNodes, b.treeCounts, geometry.gdnShape(), step.lanes,
                  layer, state, mixer.outputHeadOrder, outputPlan.input(),
                  GDN_SIGMOID_GATE)
            : ops::GDN::addDecode(
                  step.graph,
                  {b.gdnPacked[layer], mixer.convolutionWeights,
                   b.currentGdnStates, b.nextGdnStates, b.gdnMixed[layer],
                   mixer.decay, mixer.timeBias, b.gdnDecay[layer],
                   b.gdnBeta[layer], mixer.mixerNorm, b.gdnHidden,
                   b.linearScratch, b.gdnTape},
                  geometry.gdnShape(), step.lanes, layer, state,
                  mixer.outputHeadOrder, outputPlan.input(), step.liveRows,
                  step.rowCapacity, step.tapeLanes, GDN_SIGMOID_GATE);
    linear.add(step.graph,
               {.input = b.gdnHidden, .output = verify_.branch,
                .scratch = b.linearScratch, .prepared = hidden},
               mixer.outputProjection, outputPlan);
    step.graph.endBakedSpan();
  }
  // The MoE block: upstream splits the routing (addPrefillLayer's
  // addRouting) from the grouped passes (addPrefillExperts' addRouted)
  // around its streamed expert load; richengine's experts are resident
  // (BlockMoeWeights), so one ops::MoE::add encodes the whole block — the
  // F32 router's 512-slot score rows, the _e512 select and grouping and
  // the routed tiles — adding the zero-residual route sum into the branch
  // the combine then injects. The verify step routes its lanes' verify
  // rows alone (MoeBuffers::verifyRows), as upstream's addVerify does.
  void addMoeBlock(TargetModel::PrefillStep &step,
                   const TargetLayerWeights &layer) const {
    const TargetModelPrefillBuffers &b = step.buffers;
    ops::MoE::add(step.graph,
                  {b.normalized, prefill_.zeros, prefill_.branch, b.moe,
                   {}, {}},
                  std::get<ops::MoeWeights>(layer.ffn), *step.moe);
  }
  void addMoeBlock(TargetModel::VerifyStep &step,
                   const TargetLayerWeights &layer) const {
    const TargetModelVerifyBuffers &b = step.buffers;
    ops::MoE::add(step.graph,
                  {b.normalized, verify_.zeros, verify_.branch, b.moe,
                   {}, b.verifyRows},
                  std::get<ops::MoeWeights>(layer.ffn), *step.moe);
  }

  // The draft's read of a capture layer's output: the mean of the row's
  // streams into the capture slot (upstream addStreamCapture).
  void addPrefillCapture(TargetModel::PrefillStep &step,
                         uint32_t index) const {
    const auto slot = model_.geometry_.captureSlot(index);
    if (!slot) return;
    const uint32_t width = model_.geometry_.hiddenSize;
    const uint32_t captured = model_.geometry_.capturedHiddenSize();
    for (const TargetModelPrefillSequence &sequence : step.sequences)
      for (uint32_t c = 0; c < sequence.captureCount; ++c) {
        const TargetModelPrefillCapture &capture = sequence.captures[c];
        ops::HyperConnection::addStreamMean(
            step.graph,
            rowsOf<float>(model_.backend_, prefill_.streams,
                          capture.sourceStart, capture.rows,
                          weights_.layout.streamWidth()),
            step.buffers.captured,
            {capture.destinationStart, captured, *slot * width},
            hyperShape(), capture.rows);
      }
  }
  void addVerifyCapture(TargetModel::VerifyStep &step, uint32_t index) const {
    const auto slot = model_.geometry_.captureSlot(index);
    if (!slot) return;
    const uint32_t width = model_.geometry_.hiddenSize;
    ops::HyperConnection::addStreamMean(
        step.graph, verify_.streams,
        step.tree ? step.buffers.capturedPath
                  : step.buffers.capturedTargetHidden,
        {0, model_.geometry_.capturedHiddenSize(), *slot * width},
        hyperShape(), step.rows);
  }

  const TargetModel &model_;
  const Qwen4ExpWeights &weights_;
  // The auxiliary region's offset inside a state cell: after the GDN
  // layers' convolution and recurrent bytes (the cell's PLE history comes
  // first in it).
  const uint64_t auxiliaryOffset_ = 0;
  // The step scratch and the PLE tensors, at the prefill chunk's and the
  // verify step's most rows (upstream Qwen4Steps' prefill/decode tensors).
  Qwen4StepScratch prefill_;
  Qwen4StepScratch verify_;
  Qwen4PleScratch prefillPle_;
  Qwen4PleScratch verifyPle_;
  // rotaryPairs fp32 inverse frequencies for the indexer's pooled-key
  // rotation.
  metal::MetalBuffer ropeInverseFrequencies_;
  // The lanes' first verify positions and their block counts, which
  // commitState's QSA pending commit reads (upstream verifyPositions).
  mutable std::array<uint32_t, ExecutionLimits::maximumBatchWidth>
      verifyFirstPositions_{};
  mutable std::array<uint32_t, ExecutionLimits::maximumBatchWidth>
      verifyBlockCounts_{};
  mutable uint32_t verifyLanes_ = 0;
  mutable uint32_t verifyStride_ = 0;
};

std::unique_ptr<TargetLayerEncoder>
makeQwen4LayerEncoder(const TargetModel &model) {
  return std::make_unique<Qwen4LayerEncoder>(model);
}

// The products stay below 2^63 (the multipliers are bounded by the
// vocabulary), so the unsigned hash equals the reference's signed one.
void qwen4ExpPleRows(const Qwen4ExpWeights &weights, uint64_t position,
                     std::span<const uint32_t> tokens,
                     std::span<const uint32_t> history, uint32_t *out) {
  const TargetLayout &layout = weights.layout;
  const Qwen4PleHash &hash = weights.pleHash;
  const uint32_t eos = weights.pleEosToken;
  if (history.size() < position)
    throw std::logic_error(
        "qwen4exp PLE history is shorter than its position");
  const auto token = [&](uint64_t at) {
    return at >= position ? tokens[at - position] : history[at];
  };
  for (uint32_t row = 0; row < tokens.size(); ++row) {
    std::array<uint64_t, kQwen4PleMaxNgram> context{tokens[row]};
    bool cut = false;
    // The sequence's start cuts the window as an EOS does.
    for (uint32_t back = 1; back < layout.pleNgram; ++back) {
      cut = cut || position + row < back || token(position + row - back) == eos;
      context[back] = cut ? eos : token(position + row - back);
    }
    uint32_t *rowHeads = out + uint64_t{row} * layout.pleHeads();
    for (uint32_t gram = 2; gram <= layout.pleNgram; ++gram) {
      uint64_t mixed = context[0] * hash.multipliers[0];
      for (uint32_t j = 1; j < gram; ++j)
        mixed ^= context[j] * hash.multipliers[j];
      for (uint32_t g = 0; g < layout.pleHeadsPerNgram; ++g) {
        const uint32_t head =
            (gram - 2) * layout.pleHeadsPerNgram + g;
        rowHeads[head] = static_cast<uint32_t>(
            mixed % hash.headVocabularies[head] + hash.headOffsets[head]);
      }
    }
  }
}

} // namespace richengine::model
