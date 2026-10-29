#pragma once

#include "metal/CommandGraph.hpp"
#include "ops/Vision.hpp"

#include <cstdint>
#include <vector>

namespace richengine::ops {

// The lfm3a audio encoder of an LFM3-Omni mmproj: a conv frontend that
// downsamples the float32 [128 mel x T] log-mel by eight in time, then a
// seventeen-block hybrid trunk at hidden 768 — depthwise-causal conv blocks
// at 0,1,3,4,6,7,9,10,12,13 and causal sliding-window-32 GQA attention
// blocks at 2,5,8,11,14,15,16 — an SiLU-gated FFN of 1792 (the GGUF's
// clip.audio.feed_forward_length; the HF config's 2560 states a different
// module), a final RMSNorm, and the LayerNorm + GELU(erf) projector into
// the text model's hidden width. A block kind follows its tensors: a conv
// block carries conv_pw1/conv_pw2 pointwise projections and a taps-3
// depthwise kernel, an attention block the bias-free q/k/v/out set with
// per-head q/k RMSNorms, NEOX rope over all 96 head dims and no gate.
struct LfmAudioLayout final {
  uint32_t depth = 0;          // 17
  uint32_t hiddenSize = 0;     // 768
  uint32_t intermediateSize = 0; // 1792
  uint32_t outputHiddenSize = 0; // 2048 (the text model's hidden)
  uint32_t heads = 0;          // 8
  uint32_t kvHeads = 0;        // 4
  uint32_t headDimension = 0;  // 96
  uint32_t melBins = 0;        // 128
  uint32_t convChannels = 0;   // 256
  uint32_t window = 0;         // 32 (causal sliding attention window)
  uint32_t tokenStride = 0;    // 8 mel frames per language token

  bool operator==(const LfmAudioLayout &) const = default;
};

struct LfmAudioBlock final {
  VisionNorm norm1; // RMSNorm: the weight is used, the bias stays zero.
  // Conv block: conv_pw1 hidden->3*hidden ([B|C|X]), the taps-3 depthwise
  // kernel (hidden x 3 rows), conv_pw2. Attention block: bias-free
  // q/k/v/out with per-head q/k RMSNorms.
  bool conv = false;
  metal::MetalBuffer convPw1;
  metal::MetalBuffer convDw;
  metal::MetalBuffer convPw2;
  metal::MetalBuffer query;
  metal::MetalBuffer key;
  metal::MetalBuffer value;
  metal::MetalBuffer queryNorm;
  metal::MetalBuffer keyNorm;
  metal::MetalBuffer outProjection;
  VisionNorm norm2;
  metal::MetalBuffer ffnGate;
  metal::MetalBuffer ffnUp;
  metal::MetalBuffer ffnDown;
};

struct LfmAudioWeights final {
  LfmAudioLayout layout;
  // The conv frontend: stage 1 dense 1->convChannels k3 stride-2 conv,
  // stages 2 and 3 a depthwise conv plus a 1x1 mixing conv each. Kernels
  // are packed oc-major ([out][ky*3+kx]); the 1x1 convs [out][in].
  metal::MetalBuffer convInWeight, convInBias;
  metal::MetalBuffer convMidWeight, convMidBias;
  metal::MetalBuffer linMidWeight, linMidBias;
  metal::MetalBuffer convEndWeight, convEndBias;
  metal::MetalBuffer linEndWeight, linEndBias;
  // a.pre_encode.out: the flattened (convChannels*melBins/8) -> hidden map.
  metal::MetalBuffer preEncodeWeight, preEncodeBias;
  std::vector<LfmAudioBlock> blocks;
  metal::MetalBuffer postNorm; // a.post_ln: RMSNorm, weight only.
  // The mm.a.mlp projector: LayerNorm(eps 1e-5, bias) then hidden ->
  // outputHidden with GELU(erf), then outputHidden -> outputHidden.
  VisionAffine projectorNorm;
  VisionAffine projectorUp;
  VisionAffine projectorDown;
  // The tower's block projections are bias-free; the shared GEMM kernels
  // read a bias row regardless, so they bind this zero row.
  metal::MetalBuffer zeroBias;
};

// The lfm3a encoder: one full clip's mel in one graph pass (no streaming
// state; the window the conv frontend looks back over is zero, matching a
// whole-clip encode). Output is one bf16 row of outputHiddenSize per
// language placeholder token (ceil(melFrames / 8)).
class LfmAudioEnc final {
public:
  // The scratch arena a clip of at most `maximumMelFrames` needs.
  [[nodiscard]] static uint64_t scratchBytes(const LfmAudioLayout &layout,
                                               uint32_t maximumMelFrames);

  LfmAudioEnc(metal::MetalBackend &backend, const LfmAudioWeights &model,
              uint32_t maximumMelFrames);
  LfmAudioEnc(const LfmAudioEnc &) = delete;
  LfmAudioEnc &operator=(const LfmAudioEnc &) = delete;

  [[nodiscard]] uint64_t arenaBytes() const noexcept {
    return arena_.sizeBytes();
  }
  [[nodiscard]] uint32_t maximumMelFrames() const noexcept {
    return maximumMelFrames_;
  }

  // Encodes the float32 [melBins x melFrames] rows of `mel` into one
  // outputHiddenSize-wide bf16 row per placeholder token of `embeddings`.
  void encode(metal::CommandGraph &graph, uint32_t melFrames,
              const metal::MetalBuffer &mel,
              const metal::MetalBuffer &embeddings) const;

private:
  enum class Scratch : uint32_t {
    MelBfloat,
    Stage1,
    Stage2,
    Stage2Out,
    Stage3,
    Stage3Out,
    Flat,
    Hidden,
    Normalized,
    Wide,
    Gate,
    Up,
    Queries,
    Keys,
    Values,
    Attention,
    Query,
    Key,
    Value,
    Context,
    Count,
  };

  [[nodiscard]] const metal::MetalBuffer &scratch(Scratch tensor) const noexcept {
    return scratch_[static_cast<uint32_t>(tensor)];
  }
  void addGemm(metal::CommandGraph &graph, const char *pipeline,
               const metal::MetalBuffer &input, const metal::MetalBuffer &weight,
               const metal::MetalBuffer &bias, const metal::MetalBuffer &output,
               const metal::MetalBuffer &residual, uint32_t outputSize,
               uint32_t inputSize, uint32_t rows, uint32_t tileRows,
               uint32_t tileColumns) const;
  void addNorm(metal::CommandGraph &graph, const metal::MetalBuffer &input,
               const metal::MetalBuffer &weight, const metal::MetalBuffer &output,
               uint32_t rows) const;

  const LfmAudioWeights &model_;
  uint32_t maximumMelFrames_ = 0;
  metal::MetalBuffer arena_;
  metal::MetalBuffer scratch_[static_cast<uint32_t>(Scratch::Count)];
};

} // namespace richengine::ops
