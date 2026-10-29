#pragma once

#include <array>
#include <cstdint>
#include <deque>
#include <random>
#include <span>
#include <vector>

// LFM3-Omni speech output, host-side (fp32, one request at a time per
// instance): the MaskGIT depthformer (a.gen.code.*), the LiquidCodec
// decoder (a.gen.wav.*) with its streaming state and the ISTFT producing
// 24 kHz PCM. Every generated audio frame is one LM position: the engine
// feeds the LM's final hidden state at that position in, and the frame's
// feedback embedding goes back as the next position's input embedding
// (the <|audio_out|> placeholder slot).
namespace richengine::ops {

struct LfmSpeechWeights final {
  // Depthformer (a.gen.code.*): hidden 1024, 12 codebook slots, 3
  // bidirectional layers, 32 query / 8 key-value heads of 32.
  struct DepthLayer final {
    std::vector<float> ln1;
    std::vector<float> attnQ;     // [1024][1024]
    std::vector<float> attnK;     // [256][1024]
    std::vector<float> attnV;     // [256][1024]
    std::vector<float> attnQNorm; // [32]
    std::vector<float> attnKNorm; // [32]
    std::vector<float> attnGate;  // [32][1024]
    std::vector<float> attnOut;   // [1024][1024]
    std::vector<float> ln2;
    std::vector<float> ffnGate;   // [2816][1024]
    std::vector<float> ffnUp;
    std::vector<float> ffnDown;   // [1024][2816]
  };
  std::vector<float> projIn;     // [12288][2048]
  std::vector<float> projInBias; // [12288]
  std::vector<float> embd;       // [12][2017][1024], tied output head
  std::vector<float> maskEmbd;   // [1024]
  std::vector<float> outEmbd;    // [12*2017][2048]
  std::vector<float> outputNorm; // [1024]
  std::vector<float> headNorm;   // [12][1024]
  std::array<DepthLayer, 3> depth{};

  // LiquidCodec decoder (a.gen.wav.*): hidden 512, intermediate 2304, 8
  // layers (conv at 0,1,3,5,7; sliding-30 attention at 2,4,6), 8 query /
  // 2 key-value heads of 64, sandwich norms.
  struct WavLayer final {
    bool conv = false;
    std::vector<float> ln1;
    std::vector<float> post1; // attn_post_norm
    std::vector<float> ln2;
    std::vector<float> post2; // ffn_post_norm
    std::vector<float> ffnGate; // [2304][512]
    std::vector<float> ffnUp;
    std::vector<float> ffnDown; // [512][2304]
    // conv layers only
    std::vector<float> convPw1; // [1536][512]
    std::vector<float> convDw;  // [7][512]
    std::vector<float> convPw2; // [512][512]
    // attention layers only
    std::vector<float> attnQ;    // [512][512]
    std::vector<float> attnK;    // [128][512]
    std::vector<float> attnV;    // [128][512]
    std::vector<float> attnQNorm; // [64]
    std::vector<float> attnKNorm;
    std::vector<float> attnGate; // [8][512]
    std::vector<float> attnOut;  // [512][512]
  };
  std::vector<float> fsqTable;   // [2016][4]
  std::vector<float> quantOut;   // [256][48]
  std::vector<float> quantOutB;  // [256]
  std::vector<float> tfmIn;      // [512][256]
  std::vector<float> tfmInB;     // [512]
  std::vector<float> tfmOut;     // [1282][512]
  std::vector<float> tfmOutB;    // [1282]
  std::vector<float> tfmOutNorm; // [512]
  std::array<WavLayer, 8> wav{};
};

// The codec decoder's cross-call state: rope offset, short-conv taps and
// the last-29 k/v positions of each attention layer, plus the ISTFT
// overlap buffers. One per in-flight audio request.
struct LfmSpeechCodecState final {
  int64_t position = 0;
  // per conv layer: last 6 b*x rows [6][512]
  std::array<std::vector<float>, 8> convTaps{};
  // per attention layer: last 29 positions [29][128] (2 kv heads x 64)
  std::array<std::vector<float>, 8> keyWindow{};
  std::array<std::vector<float>, 8> valueWindow{};

  // streaming ISTFT (n_fft 1280, hop 320)
  std::vector<float> overlap = std::vector<float>(1280, 0.0F);
  std::vector<float> windowSum = std::vector<float>(1280, 0.0F);
  int32_t paddingToRemove = 480;
};

// One speech-generation context = one request. Holds the audio
// repetition history (64 frames) and the codec streaming state.
class LfmSpeechGen final {
public:
  static constexpr int kCodebooks = 12;
  static constexpr int kVocab = 2017;   // codes 0..2016; 2016 = end of audio
  static constexpr int kEosCode = 2016;
  static constexpr int kDepthHidden = 1024;
  static constexpr int kLmHidden = 2048;
  static constexpr int kWavHidden = 512;
  static constexpr int kFsqDim = 4;
  static constexpr int kFsqCodes = 2016;
  static constexpr int kQuantDim = 256;
  static constexpr int kFrameStack = 6;     // STFT columns per codec frame
  static constexpr int kStftBins = 641;     // n_fft/2 + 1
  static constexpr int kNfft = 1280;
  static constexpr int kHop = 320;
  static constexpr int kSamplesPerFrame = 1920; // 6*320 @24kHz
  static constexpr int kAttnPrefix = 29;    // sliding window 30 - 1
  static constexpr int kConvTaps = 7;
  static constexpr int kRepWindow = 64;
  static constexpr float kAudioTemp = 0.4F;
  static constexpr float kRepPenalty = 1.2F;

  explicit LfmSpeechGen(const LfmSpeechWeights &weights, uint64_t seed = 0)
      : weights_(weights), rng_(static_cast<std::mt19937::result_type>(seed)) {}

  // h_state: the LM's final hidden state (fp32, kLmHidden) at the position
  // preceding the frame. Returns false when the weights are absent.
  // On return: codes holds the frame's 12 codes (all 2016 on end-of-audio),
  // eos flags end-of-audio, fb the feedback embedding for the next LM
  // input (still produced for the end-of-audio frame: the position after
  // it continues in text).
  [[nodiscard]] bool sampleFrame(std::span<const float> hState,
                                 std::array<int32_t, kCodebooks> &codes,
                                 bool &eos, std::vector<float> &fb);

  // Runs a non-EOS frame's codes through the codec + ISTFT. Returns the
  // frame's appended PCM (1920 float samples).
  [[nodiscard]] std::vector<float>
  decodeFrame(const std::array<int32_t, kCodebooks> &codes);

  // Drains the ISTFT tail at the end of an audio segment: remaining
  // overlap samples minus the last 480 of the segment.
  [[nodiscard]] std::vector<float> flush();

  [[nodiscard]] const LfmSpeechCodecState &codecState() const noexcept { return codec_; }
  [[nodiscard]] LfmSpeechCodecState &codecState() noexcept { return codec_; }

private:
  // One GEN_CODE pass: proj is projIn(hState)+bias cached by the caller;
  // need/nNeed selects the codebook slots whose logits are required (0
  // skips the tied-embd head entirely); fb is written when fbOut is set.
  void depthForward(const float *proj, const int32_t *codes,
                    const float *mask, float *logits, const int32_t *need,
                    int nNeed, float *fb) const;
  void repPenalty(float *logits, int cb) const;
  int32_t sampleCode(const float *logits);

  const LfmSpeechWeights &weights_;
  std::mt19937 rng_;
  std::deque<std::array<int32_t, kCodebooks>> history_; // last 64 frames
  LfmSpeechCodecState codec_;
};

// The periodic Hann window the ISTFT applies (n=1280).
[[nodiscard]] std::vector<float> lfmHannWindow(int n);

// De-interleaves the bf16 image rows the loader writes into the fp32
// weight struct; throws std::invalid_argument on shape mismatches.
[[nodiscard]] LfmSpeechWeights
decodeLfmSpeechWeights(std::span<const uint16_t> bf16,
                       std::span<const uint64_t> offsets);

// The per-section offsets of the packed image, in the order the loader
// writes them (mirrors LfmGenLoader::sectionCount):
//   7 head sections + 3x12 depth-layer sections + 8 wav-head sections
//   + 5 conv layers x 10 + 3 attention layers x 13.
inline constexpr uint32_t kLfmGenSectionCount = 7 + 3 * 12 + 8 + 5 * 10 + 3 * 13;

} // namespace richengine::ops
