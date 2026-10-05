#pragma once

// Plans the images (model/GgufImageLayout.hpp) of a Qwen3.8 (qwen35)
// or Qwen3.6 MoE (qwen35moe) target read straight from a llama.cpp GGUF,
// or a dense (llama), LFM2 (lfm2) or LFM2-MoE (lfm2moe) one, from
// its metadata alone: section offsets, the header and descriptor bytes, and
// the source rows each tensor section is written from
// (model/GgufPreparation.hpp). A 3-D expert tensor is one quantized tensor of
// experts * N rows.

#include <cstdint>
#include <string>
#include <vector>

#include "model/GgufFile.hpp"

namespace richengine::model::gguf {

struct TargetGeometry {
  uint32_t layers = 0;
  uint32_t hiddenSize = 0;
  uint32_t vocabularySize = 0;
  uint32_t intermediateSize = 0; // dense FFN
  uint32_t gdnKeyHeads = 0;
  uint32_t gdnValueHeads = 0;
  uint32_t gdnHeadDimension = 0;
  uint32_t convolutionDimension = 0;
  uint32_t attentionWidth = 0;
  uint32_t attentionKvHeads = 0;
  uint32_t attentionHeadDimension = 0;
  // The rotated dimension pairs of each attention head and their RoPE base.
  uint32_t rotaryPairs = 0;
  float rotaryTheta = 0.0F;
  uint32_t fullAttentionPeriod = 0;
  // A sparse MoE FFN (qwen35moe, lfm2moe) when experts is set; the shared
  // expert has the routed experts' intermediate width when present.
  uint32_t experts = 0;
  uint32_t expertsPerToken = 0;
  uint32_t expertIntermediateSize = 0;
  // A MoE target's leading dense-FFN layers (lfm2moe's
  // leading_dense_block_count; 0 means every layer is MoE).
  uint32_t leadingDenseLayers = 0;
  // Whether the MoE blocks carry a shared expert (the qwen35moe shexp
  // tensors); lfm2moe has none — it routes sigmoid probabilities with a
  // per-expert selection bias (exp_probs_b) instead.
  bool sharedExpert = true;
  // A mask-typed attention schedule (LFM2) when set: bit `layer` is a
  // full-attention layer. Zero keeps the fullAttentionPeriod schedule.
  uint64_t attentionMask = 0;
  // The conv kernel's taps (the GDN's 4 when zero; LFM2's conv_L_cache 3).
  uint32_t convolutionTaps = 0;
  // The RMS norms' epsilon (1e-6, or LFM2's norm_eps 1e-5).
  float rmsEpsilon = 1e-6F;
  // The q*k softmax scale (Granite's attention.scale); zero selects the
  // head dimension's default.
  float attentionScale = 0.0F;
  // Whether the packed query rows interleave a gate row each (Qwen) and
  // whether the q/k heads carry RMS norms (absent in the dense target).
  bool attentionQueryGate = true;
  bool attentionQkNorm = true;
  // Whether the LM head shares the token embedding (LFM2 ties them).
  bool tiedOutput = false;
  // The general.architecture a GGUF of this target declares; empty keeps
  // the Qwen families' names.
  std::string arch;
  [[nodiscard]] bool isFullAttentionLayer(uint32_t layer) const noexcept {
    if (attentionMask) return layer < 64 && (attentionMask >> layer) & 1;
    return (layer + 1) % fullAttentionPeriod == 0;
  }
  [[nodiscard]] bool sparseMoe() const noexcept { return experts != 0; }
  // The general.architecture of a GGUF of this target.
  [[nodiscard]] std::string architecture() const {
    return arch.empty() ? (sparseMoe() ? "qwen35moe" : "qwen35") : arch;
  }
};

// The order of a tensor's rows in the image. Rows below `from` keep their
// order; from there on, blocks of headRows rows are value heads, which
// llama.cpp stores tiled (value head of its key head * keyHeads + key head)
// and richengine groups by key head (key head * valueHeadsPerKey + value head).
// A rotaryInterleaved row order instead deinterleaves each headRows block:
// a "llama" GGUF stores a rotated head's rows as rope pairs (HF dimension j
// in stored row 2j, j + headRows/2 in 2j + 1); the image keeps HF order, so
// image row headRows*h + j reads stored row headRows*h + 2*(j % half) +
// (j >= half).
struct RowOrder {
  uint64_t from = UINT64_MAX; // UINT64_MAX: rows as stored
  uint32_t headRows = 0;
  uint32_t keyHeads = 0;
  uint32_t valueHeadsPerKey = 0;
  bool rotaryInterleaved = false;
};

// Rows [0, rows) of one source tensor in image order.
struct TensorRows {
  std::string name;
  uint32_t type = 0;   // ggml type
  uint64_t offset = 0; // in the file's tensor data
  uint64_t rows = 0;
  uint64_t rowBytes = 0;
  RowOrder order{};
};

// Header and descriptor bytes.
struct Fill {
  uint64_t offset = 0;
  std::vector<uint8_t> bytes;
};
// How a copy writes each value: as stored, narrowed from F32 to the bf16
// value it equals exactly (rows the kernels read as bf16), or widened from
// BF16 to the F32 value it equals (rows the kernels read as F32).
enum class Conversion : uint8_t { None, NarrowToBfloat16, WidenToFloat32 };
// Rows written back to back, each value converted as `conversion` says.
struct Copy {
  uint64_t destination = 0;
  TensorRows source;
  Conversion conversion = Conversion::None;
};
// Quantized rows repacked into the planes of their format; the rows of the
// sources in order, then zero rows up to `rows`.
struct Repack {
  uint32_t format = 0; // GGUF_FMT_*
  uint64_t rows = 0;
  uint64_t columns = 0;
  uint64_t plane0 = 0, plane1 = 0, meta = 0; // image offsets; plane1 when the format has one
  std::vector<TensorRows> sources;
};
struct Image {
  std::string name; // layer-N.bin, head.bin, embedding.bin
  std::string magic; // kGgufImageMagic
  uint32_t layer = 0;
  uint32_t type = 0;
  uint64_t bytes = 0;
  std::vector<Fill> fills;
  std::vector<Copy> copies;
  std::vector<Repack> repacks;
};

// The layers' images, then the head's and the embedding's. Checks the
// architecture, the geometry the metadata declares, its rotary embedding and
// norms included, and each tensor's shape; throws GgufError naming every
// missing tensor and every tensor of a type this build cannot load.
[[nodiscard]] std::vector<Image> planImages(const GgufFile &file, const TargetGeometry &geometry);

} // namespace richengine::model::gguf
