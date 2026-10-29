#include "LfmAudioEnc.hpp"

#include "Checked.hpp"
#include "metal/abi/Audio.h"
#include "metal/abi/Vision.h"
#include "ops/KernelNames.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace richengine::ops {
namespace {

using metal::CommandGraph;
using metal::MetalBuffer;

constexpr uint32_t kGemmRowTile = 64;
constexpr uint32_t kGemmColumnTile = 128;
constexpr uint32_t kMergerRowTile = 32;
constexpr uint32_t kMergerColumnTile = 256;
constexpr uint32_t kKeyTile = 128;
constexpr uint64_t kBf16Bytes = 2;
constexpr uint64_t kFloatBytes = 4;

// The lfm3a encoder the kernels are specialized for (LfmAudioWeights'
// declared layout must match it except outputHiddenSize, the text model's
// hidden width).
constexpr LfmAudioLayout kTower{.depth = 17,
                                .hiddenSize = 768,
                                .intermediateSize = 1792,
                                .outputHiddenSize = 2048,
                                .heads = 8,
                                .kvHeads = 4,
                                .headDimension = 96,
                                .melBins = 128,
                                .convChannels = 256,
                                .window = 32,
                                .tokenStride = 8};
static_assert(kTower.hiddenSize % kGemmColumnTile == 0 &&
                  kTower.intermediateSize % kGemmColumnTile == 0 &&
                  3 * kTower.hiddenSize % kGemmColumnTile == 0 &&
                  kTower.hiddenSize * kTower.heads % kGemmColumnTile == 0 &&
                  kTower.outputHiddenSize % kMergerColumnTile == 0,
              "lfm audio kernels' tile assumptions");

uint32_t roundUp(uint32_t value, uint32_t multiple) noexcept {
  return (value + multiple - 1) / multiple * multiple;
}

// Elementwise kernels run one thread per element: count THREADS in
// kDefaultThreads-sized groups — the groups argument is a threadgroup
// count, so a plain count would overdispatch by 256x and run the grid
// past the scratch rows the kernels bound their writes to.
metal::DispatchSize elementwise(uint64_t count) noexcept {
  return {(count + CommandGraph::kDefaultThreads - 1) /
              CommandGraph::kDefaultThreads,
          1, 1};
}

// Encoder positions a clip's mel frames produce (stride 8 in time).
uint32_t positions(uint32_t melFrames) noexcept {
  return (melFrames + kTower.tokenStride - 1) / kTower.tokenStride;
}

// Scratch tensor byte sizes for an encoder sized to maximumMelFrames, in
// Scratch order. Conv-stage buffers pad their (freq * time) position count
// to GEMM rows; trunk rows pad to 64, head planes to 128 tokens.
std::array<uint64_t, 20> scratchLayout(const LfmAudioLayout &layout,
                                       uint32_t maximumMelFrames) {
  const uint64_t t1 = maximumMelFrames;
  const uint64_t t2 = (t1 + 1) / 2;
  const uint64_t t3 = (t2 + 1) / 2;
  const uint64_t t4 = (t3 + 1) / 2;
  // The 1x1 convs are GEMMs, so each stage's (freq * time) position count
  // pads to a whole 64-row tile.
  const uint64_t stage1 = roundUp(64 * t2, kGemmRowTile);
  const uint64_t stage2 = roundUp(32 * t3, kGemmRowTile);
  const uint64_t stage3 = roundUp(16 * t4, kGemmRowTile);
  const uint64_t conv = layout.convChannels;
  const uint64_t rows = roundUp(positions(maximumMelFrames), kGemmRowTile);
  const uint64_t padded = roundUp(positions(maximumMelFrames), kKeyTile);
  const uint64_t hidden = layout.hiddenSize;
  const uint64_t kv = uint64_t{layout.kvHeads} * layout.headDimension;
  const uint64_t flat = uint64_t{layout.melBins} / kTower.tokenStride * conv;
  const uint64_t headRows = uint64_t{layout.heads} * padded;
  const uint64_t kvHeadRows = uint64_t{layout.kvHeads} * padded;
  return {
      // The mel no longer packs to bf16 (stage 1 reads it as f32 through
      // the im2col gather): the slot carries the three tap-major
      // transposed conv kernels (3 x conv x 9 elements) then the K=16
      // im2col rows (stage1 x 16) stage 1's matmul multiplies.
      std::max(uint64_t{layout.melBins} * t1,
               uint64_t{3} * conv * 9 + stage1 * 16) * kBf16Bytes, // MelBfloat
      stage1 * conv * kBf16Bytes,                           // Stage1
      stage2 * conv * kBf16Bytes,                           // Stage2
      stage2 * conv * kBf16Bytes,                           // Stage2Out
      stage3 * conv * kBf16Bytes,                           // Stage3
      stage3 * conv * kBf16Bytes,                           // Stage3Out
      rows * flat * kBf16Bytes,                             // Flat
      rows * hidden * kFloatBytes,                          // Hidden
      rows * hidden * kBf16Bytes,                           // Normalized
      rows * 3 * hidden * kBf16Bytes,                       // Wide
      rows * layout.intermediateSize * kBf16Bytes,          // Gate
      rows * layout.intermediateSize * kBf16Bytes,          // Up
      headRows * layout.headDimension * kBf16Bytes,         // Queries
      kvHeadRows * layout.headDimension * kBf16Bytes,       // Keys
      kvHeadRows * layout.headDimension * kBf16Bytes,       // Values
      headRows * layout.headDimension * kBf16Bytes,         // Attention
      rows * hidden * kBf16Bytes,                           // Query
      rows * kv * kBf16Bytes,                               // Key
      rows * kv * kBf16Bytes,                               // Value
      rows * hidden * kBf16Bytes,                           // Context
  };
}

void requireLayout(const LfmAudioLayout &layout) {
  LfmAudioLayout tower = layout;
  tower.outputHiddenSize = kTower.outputHiddenSize;
  if (tower != kTower || !layout.outputHiddenSize ||
      layout.outputHiddenSize % kMergerColumnTile) {
    throw std::invalid_argument(
        "audio kernels are specialized for the lfm3a 17-block encoder");
  }
}

} // namespace

uint64_t LfmAudioEnc::scratchBytes(const LfmAudioLayout &layout,
                                   uint32_t maximumMelFrames) {
  requireLayout(layout);
  if (!maximumMelFrames) {
    throw std::invalid_argument("audio scratch must cover mel frames");
  }
  uint64_t total = 0;
  for (uint64_t bytes : scratchLayout(layout, maximumMelFrames)) {
    const uint64_t aligned = alignUp(bytes);
    if (aligned > std::numeric_limits<uint64_t>::max() - total) {
      throw std::overflow_error("audio scratch byte count overflows");
    }
    total += aligned;
  }
  return total;
}

LfmAudioEnc::LfmAudioEnc(metal::MetalBackend &backend,
                         const LfmAudioWeights &model,
                         uint32_t maximumMelFrames)
    : model_(model), maximumMelFrames_(maximumMelFrames) {
  const uint64_t total = scratchBytes(model.layout, maximumMelFrames);
  arena_ = backend.allocateBuffer(total, metal::BufferStorage::Shared,
                                  "lfm-audio-scratch");
  uint64_t cursor = 0;
  const auto layout = scratchLayout(model.layout, maximumMelFrames);
  for (uint32_t index = 0; index < layout.size(); ++index) {
    scratch_[index] = backend.view(arena_, cursor, layout[index]);
    cursor += alignUp(layout[index]);
  }
  if (cursor != total)
    throw std::logic_error("lfm audio scratch arena mismatch");
}

void LfmAudioEnc::addGemm(CommandGraph &graph, const char *pipeline,
                          const MetalBuffer &input, const MetalBuffer &weight,
                          const MetalBuffer &bias, const MetalBuffer &output,
                          const MetalBuffer &residual, uint32_t outputSize,
                          uint32_t inputSize, uint32_t rows, uint32_t tileRows,
                          uint32_t tileColumns) const {
  graph.add(pipeline, {input, weight, bias, output, residual},
            VisionGemmParams{outputSize, inputSize},
            {(rows + tileRows - 1) / tileRows, outputSize / tileColumns, 1});
}

void LfmAudioEnc::addNorm(CommandGraph &graph, const MetalBuffer &input,
                          const MetalBuffer &weight, const MetalBuffer &output,
                          uint32_t rows) const {
  graph.add(std::string(kLfmVisionRmsnorm), {input, weight, output},
            VisionNormParams{model_.layout.hiddenSize}, {rows, 1, 1});
}

void LfmAudioEnc::encode(CommandGraph &graph, uint32_t melFrames,
                         const MetalBuffer &mel,
                         const MetalBuffer &embeddings) const {
  const LfmAudioLayout &layout = model_.layout;
  if (!melFrames || melFrames > maximumMelFrames_) {
    throw std::invalid_argument("audio mel frame count exceeds the arena");
  }
  const uint32_t tokens = positions(melFrames);
  const uint64_t embeddingBytes =
      uint64_t{tokens} * layout.outputHiddenSize * kBf16Bytes;
  if (!embeddings || embeddings.sizeBytes() < embeddingBytes) {
    throw std::invalid_argument("embedding buffer is too small for the clip");
  }
  const uint64_t melBytes = uint64_t{layout.melBins} * melFrames * kFloatBytes;
  if (!mel || mel.sizeBytes() < melBytes) {
    throw std::invalid_argument("mel buffer is too small for the clip");
  }

  const uint32_t t2 = (melFrames + 1) / 2;
  const uint32_t t3 = (t2 + 1) / 2;
  const uint32_t t4 = tokens;
  const uint32_t padded = roundUp(tokens, kKeyTile);
  const uint32_t conv = layout.convChannels;
  const uint32_t hidden = layout.hiddenSize;
  const uint32_t kvWidth = layout.kvHeads * layout.headDimension;
  const uint32_t flatWidth = layout.melBins / kTower.tokenStride * conv;
  const MetalBuffer &hiddenRows = scratch(Scratch::Hidden);
  const MetalBuffer &normalized = scratch(Scratch::Normalized);
  const MetalBuffer &wide = scratch(Scratch::Wide);
  const MetalBuffer &context = scratch(Scratch::Context);

  // The conv frontend: three stride-2 stages (dense, depthwise+pointwise,
  // depthwise+pointwise) then the channel-major flatten. The mel bf16 pack
  // and the three SiLU passes are folded into the consumers' loads — the
  // first conv reads the float32 mel directly (input_mode 2), the later
  // convs and the flatten apply SiLU on load (input_mode/input_silu 1). The
  // oc-major kernels are transposed tap-major into the repurposed MelBfloat
  // slot first so every tap's weights load as one bf16x4.
  const uint64_t stageWeights = uint64_t{conv} * 9;
  const uint32_t columnsBase = uint32_t(stageWeights * 3);
  graph.add(std::string(kLfmAudioConvWt),
            {model_.convInWeight, model_.convMidWeight, model_.convEndWeight,
             scratch(Scratch::MelBfloat)},
            LfmAudioCountParams{conv * 9}, elementwise(stageWeights * 3));
  // Stage 1 is a matmul: the gather writes one K=16 im2col row per output
  // position (nine taps plus seven zeros) and the m64n128 tensor matmul
  // applies the tap-major weights with bias.
  graph.add(std::string(kLfmAudioStage1Im2col),
            {mel, scratch(Scratch::MelBfloat)},
            LfmAudioConvParams{layout.melBins, melFrames, 1,
                               layout.melBins / 2, t2, conv, 0,
                               /*weight_offset=*/columnsBase},
            elementwise(64ull * t2 * 16));
  graph.add(std::string(kLfmAudioStage1Mm),
            {scratch(Scratch::MelBfloat), scratch(Scratch::MelBfloat),
             model_.convInBias, scratch(Scratch::Stage1)},
            LfmAudioConvParams{layout.melBins, melFrames, 1,
                               layout.melBins / 2, t2, conv, 0,
                               /*weight_offset=*/columnsBase},
            {uint32_t(t2), conv / 128, 1});
  graph.add(std::string(kLfmAudioConvDown),
            {scratch(Scratch::Stage1), scratch(Scratch::MelBfloat),
             model_.convMidBias, scratch(Scratch::Stage2)},
            LfmAudioConvParams{64, t2, conv, 32, t3, conv, /*input_mode=*/1,
                               uint32_t(stageWeights)},
            elementwise(32ull * t3 * conv / 4));
  addGemm(graph, kVisionGemmM64n128.data(), scratch(Scratch::Stage2),
          model_.linMidWeight, model_.linMidBias, scratch(Scratch::Stage2Out),
          context, conv, conv, 32 * t3, kGemmRowTile, kGemmColumnTile);
  graph.add(std::string(kLfmAudioConvDown),
            {scratch(Scratch::Stage2Out), scratch(Scratch::MelBfloat),
             model_.convEndBias, scratch(Scratch::Stage3)},
            LfmAudioConvParams{32, t3, conv, 16, t4, conv, /*input_mode=*/1,
                               uint32_t(stageWeights * 2)},
            elementwise(16ull * t4 * conv / 4));
  addGemm(graph, kVisionGemmM64n128.data(), scratch(Scratch::Stage3),
          model_.linEndWeight, model_.linEndBias, scratch(Scratch::Stage3Out),
          context, conv, conv, 16 * t4, kGemmRowTile, kGemmColumnTile);
  const uint32_t rowsPad = roundUp(tokens, kGemmRowTile);
  graph.add(std::string(kLfmAudioFlatten),
            {scratch(Scratch::Stage3Out), scratch(Scratch::Flat)},
            LfmAudioFlattenParams{tokens, t4, layout.melBins / kTower.tokenStride,
                                  conv, /*input_silu=*/1},
            elementwise(uint64_t{rowsPad} * flatWidth));
  // The residual stream starts at the lin_out projection: fp32 output,
  // bias fused, no zero-fill pass.
  graph.add(std::string(kLfmAudioGemmLinOut),
            {scratch(Scratch::Flat), model_.preEncodeWeight,
             model_.preEncodeBias, hiddenRows},
            VisionGemmParams{hidden, flatWidth},
            {(tokens + kGemmRowTile - 1) / kGemmRowTile,
             hidden / kGemmColumnTile, 1});

  const LfmAudioQkvParams qkvParams{tokens, padded, /*fused=*/1};
  const LfmAudioAttentionParams attentionParams{
      tokens, padded, 1.0F / std::sqrt(static_cast<float>(layout.headDimension))};
  for (const LfmAudioBlock &block : model_.blocks) {
    addNorm(graph, hiddenRows, block.norm1.weight, normalized, tokens);
    if (block.conv) {
      // conv_pw1 -> [B|C|X] in the wide buffer; the conv kernel writes
      // C * conv1d(B * X) rows, conv_pw2 folds them into the residual.
      addGemm(graph, kVisionGemmM64n128.data(), normalized, block.convPw1,
              model_.zeroBias, wide, context, 3 * hidden, hidden, tokens,
              kGemmRowTile, kGemmColumnTile);
      graph.add(std::string(kLfmAudioConvBcx),
                {wide, block.convDw, normalized}, qkvParams,
                elementwise(uint64_t{rowsPad} * hidden));
      addGemm(graph, kVisionGemmM64n128Residual.data(), normalized,
              block.convPw2, model_.zeroBias, hiddenRows, hiddenRows, hidden,
              hidden, tokens, kGemmRowTile, kGemmColumnTile);
    } else {
      // Same fusion as the vision tower: one 1536-wide GEMM writes the
      // [Q | K | V] rows across the contiguous Query/Key/Value slots.
      const MetalBuffer &fusedQkv = scratch(Scratch::Query);
      graph.add(std::string(kLfmAudioGemmQkv),
                {normalized, block.query, block.key, block.value, fusedQkv},
                VisionGemmParams{hidden + 2 * kvWidth, hidden},
                {(tokens + kGemmRowTile - 1) / kGemmRowTile,
                 (hidden + 2 * kvWidth) / kGemmColumnTile, 1});
      // Per-head q/k RMSNorm, NEOX rope over all 96 dims and the GQA
      // scatter in one 96-thread group per (token, head); padded rows
      // write zeros.
      graph.add(std::string(kLfmAudioQkvPrepare),
                {fusedQkv, fusedQkv, fusedQkv, block.queryNorm,
                 block.keyNorm, scratch(Scratch::Queries),
                 scratch(Scratch::Keys), scratch(Scratch::Values)},
                qkvParams, {padded * layout.heads, 1, 1}, {96, 1, 1});
      graph.add(std::string(kLfmAudioAttention),
                {scratch(Scratch::Queries), scratch(Scratch::Keys),
                 scratch(Scratch::Values), scratch(Scratch::Attention)},
                attentionParams, {tokens * layout.heads, 1, 1}, {32, 1, 1});
      graph.add(std::string(kLfmAudioAttentionPack),
                {scratch(Scratch::Attention), context}, qkvParams,
                elementwise(uint64_t{rowsPad} * hidden));
      addGemm(graph, kVisionGemmM64n128Residual.data(), context,
              block.outProjection, model_.zeroBias, hiddenRows, hiddenRows,
              hidden, hidden, tokens, kGemmRowTile, kGemmColumnTile);
    }

    addNorm(graph, hiddenRows, block.norm2.weight, normalized, tokens);
    addGemm(graph, kVisionGemmM64n128.data(), normalized, block.ffnGate,
            model_.zeroBias, scratch(Scratch::Gate), context,
            layout.intermediateSize, hidden, tokens, kGemmRowTile,
            kGemmColumnTile);
    // Same fusion as the vision tower: the up GEMM's epilogue applies the
    // silu gate multiply in place over the gate rows.
    addGemm(graph, kLfmVisionGemmM64n128SiluUp.data(), normalized,
            block.ffnUp, model_.zeroBias, scratch(Scratch::Gate),
            scratch(Scratch::Gate), layout.intermediateSize, hidden, tokens,
            kGemmRowTile, kGemmColumnTile);
    addGemm(graph, kVisionGemmM64n128Residual.data(), scratch(Scratch::Gate),
            block.ffnDown, model_.zeroBias, hiddenRows, hiddenRows, hidden,
            layout.intermediateSize, tokens, kGemmRowTile, kGemmColumnTile);
  }

  // a.post_ln (RMSNorm) then the mm.a.mlp projector: LayerNorm(eps 1e-5,
  // bias) -> hidden -> outputHidden with GELU(erf) -> outputHidden.
  addNorm(graph, hiddenRows, model_.postNorm, normalized, tokens);
  graph.add(std::string(kLfmAudioLayerNorm),
            {normalized, model_.projectorNorm.weight, model_.projectorNorm.bias,
             normalized},
            LfmAudioCountParams{hidden}, {tokens, 1, 1});
  addGemm(graph, kVisionGemmM32n256GeluErf.data(), normalized,
          model_.projectorUp.weight, model_.projectorUp.bias,
          scratch(Scratch::Flat), context, layout.outputHiddenSize, hidden,
          tokens, kMergerRowTile, kMergerColumnTile);
  addGemm(graph, kVisionGemmM32n256.data(), scratch(Scratch::Flat),
          model_.projectorDown.weight, model_.projectorDown.bias, embeddings,
          context, layout.outputHiddenSize, layout.outputHiddenSize, tokens,
          kMergerRowTile, kMergerColumnTile);
}

} // namespace richengine::ops
