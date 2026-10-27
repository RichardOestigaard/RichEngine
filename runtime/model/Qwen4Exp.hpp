#pragma once

// The qwen4exp target (Qwen3.8-Flash-Next; config model_type
// qwen4_exp_text, GGUF architecture qwen4exp): the qwen3_5 hybrid of GDN
// and gated full attention over a sparse MoE, plus three additions the
// canonical TargetLayout now carries (upstream's runtime/qwen4/model/
// Qwen4Exp.hpp is the reference):
//  - Hyper-connections replace every norm: the residual is
//    hyperConnections parallel streams of hiddenSize values. A mix before
//    each mixer and each FFN (Qwen4HyperWeights) normalizes every stream,
//    gates it by a low-rank projection of all of them and averages the
//    streams into the block's input; the block's output is added back to
//    each stream scaled by its own injection weight. A last mix without
//    injection feeds the head; the model has no output norm.
//  - QSA sparse attention: each full-attention layer's rows attend, past
//    the first indexerTopBlocks blocks of indexerBlockTokens tokens, to
//    the indexerTopBlocks blocks an indexer scores highest plus the row's
//    own block.
//  - A PLE n-gram embedding feeds one GDN layer (pleLayer): the 2- and
//    3-gram hashes of every token gather rows of a table of hundreds of
//    millions of rows, which gate a value added to every stream through a
//    dilated convolution.
// Its weights load from packed files only: no richengine GGUF image plan
// or safetensors loader maps the qwen4exp tensor set yet.

#include "model/TargetLayout.hpp"
#include "model/TargetModel.hpp"
#include "model/TargetFiles.hpp"
#include "ops/Linear.hpp"
#include "ops/MoE.hpp"
#include "ops/Normalization.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace richengine::model {

// Qwen3.8-Flash-Next's canonical layout. Its packed files carry the
// family's own magics (layer sections in the order Qwen4Exp.cpp reads,
// mirroring the vendored GGUF image's: each mix's norm, down, up and
// inject; the mixer; the QSA indexer on full-attention layers; the MoE
// block; the PLE layer's tensors on layer pleLayer).
[[nodiscard]] constexpr TargetLayout qwen4ExpTargetLayout() noexcept {
  TargetLayout layout;
  layout.arch = "qwen4exp";
  layout.ggufArch = "qwen4exp";
  layout.layerMagic = "MDFQ0001";
  layout.headMagic = "MDFQ0002";
  layout.ffnKind = FfnKind::SparseMoe;
  // The packed QKV rows interleave a gate row per query head ([q|gate]).
  layout.attentionQueryStride = 2;
  // The per-head RMS norms on the queries and keys.
  layout.attentionQkNorm = true;
  // No MLX, oQ or GGUF image plan covers the qwen4exp tensor set; packages
  // are packed at installation.
  layout.packedOnly = true;
  layout.convolutionTaps = kGdnConvolutionTaps;
  layout.maximumContextTokens = kv::kMaximumLogicalTokens;
  layout.layers = 48;
  layout.hiddenSize = 2560;
  layout.vocabularySize = 248320;
  // qkv 10240 | z 6144 | beta, alpha 96, to whole 64-column tiles.
  layout.packedGdnWidth = 16512;
  layout.packedFullWidth = 13312;
  layout.convolutionDimension = 10240;
  layout.gdnKeyHeads = 16;
  layout.gdnValueHeads = 48;
  layout.gdnHeadDimension = 128;
  layout.attentionWidth = 6144;
  layout.attentionQueryHeads = 24;
  layout.attentionKvHeads = 2;
  layout.attentionHeadDimension = 256;
  layout.rotaryPairs = 32;
  layout.rotaryTheta = 10'000'000.0F;
  layout.fullAttentionPeriod = 4;
  layout.maskToken = 248077;
  layout.stopTokens = {248044, 248046};
  // The DFlash2 draft's target_layer_ids.
  layout.hiddenCaptureLayers = {3, 17, 31, 47};
  layout.captureLayerCount = 4;
  // The QSA attention core: 24 query heads on 2 KV heads of 256, dense and
  // block-list sparse kernels alike (kv::Core::Kv2Group12H256). The kv
  // layout kvLayout() derives also carries the block index plane and
  // token-major values the sparse kernels read.
  layout.attentionCore = kv::Core::Kv2Group12H256;
  layout.experts = 512;
  layout.expertsPerToken = 10;
  layout.expertIntermediateSize = 640;
  // The Value48 kernel set matches the recurrent block's widths; the
  // layers' encodes pass its gate phase GDN_SIGMOID_GATE (the upstream
  // GDN kernels' flag), sigmoid(z) where the default is silu(z).
  layout.gdnCore = ops::GdnCore::Value48;
  layout.moeSharedExpert = true;
  // The hyper-connected residual: four streams of hiddenSize and the
  // mixes' 320-wide low rank.
  layout.hyperConnections = 4;
  layout.hyperRank = 320;
  // The QSA indexer: 4 heads of 128, top-512 blocks of 4 tokens.
  layout.indexerHeads = 4;
  layout.indexerHeadDimension = 128;
  layout.indexerTopBlocks = 512;
  layout.indexerBlockTokens = 4;
  // The PLE layer and its n-gram embedding.
  layout.pleLayer = 1;
  layout.pleNgram = 3;
  layout.pleHeadsPerNgram = 8;
  layout.pleHeadDimension = 160;
  layout.pleConvolutionTaps = 4;
  layout.pleEosToken = 248044;
  layout.pleImageToken = 248056;
  return layout;
}

// The longest n-gram the PLE hash mixes (upstream's
// NGRAM_EMBEDDING_MAXIMUM_NGRAM, metal/abi/NgramEmbedding.h).
inline constexpr uint32_t kQwen4PleMaxNgram = 3;
// The hash's "no token" marker (upstream's NGRAM_EMBEDDING_NONE).
inline constexpr uint32_t kQwen4PleNoToken = 0xffffffffU;

// The PLE hash the packed table file's header states: the multiplier of
// each n-gram position (0 past pleNgram), each head's table rows and the
// table's rows; and the tokens the hash treats as end-of-sequence and
// reads at image positions.
struct Qwen4PleHash final {
  std::array<uint64_t, kQwen4PleMaxNgram> multipliers{};
  std::vector<uint32_t> headOffsets;
  std::vector<uint32_t> headVocabularies;
  uint64_t tableRows = 0;
};

// The PLE hash header the packed ple-table.bin opens with, little-endian
// (the .cpp reads it as its file's first section).
struct Qwen4PleHeader final {
  uint32_t eosToken = 0;
  uint32_t imageToken = 0;
  uint32_t headCount = 0;
  uint32_t reserved = 0;
  uint64_t tableRows = 0;
  uint64_t multipliers[kQwen4PleMaxNgram]{};
  // headOffsets then headVocabularies, layout.pleHeads() entries each.
  uint32_t heads[2 * ((kQwen4PleMaxNgram - 1) * 8)]{};
};
static_assert(sizeof(Qwen4PleHeader) ==
                  24 + 8 * kQwen4PleMaxNgram +
                      4 * 2 * ((kQwen4PleMaxNgram - 1) * 8),
              "the packed PLE hash header is POD");

// One hyper-connection mix of streamWidth() stream values: their norm
// (streamWidth() multipliers), the low-rank gate's down [hyperRank,
// streamWidth()] and up [streamWidth(), hyperRank] projections, and, for
// a mix that injects its block's output back, the injection weights
// [hyperConnections][streamWidth()] F32.
struct Qwen4HyperWeights final {
  ops::NormWeights norm;
  ops::Projection down;
  ops::Projection up;
  metal::MetalBuffer inject;
};

// The QSA indexer of a full-attention layer: its query [heads * dim,
// hidden] and key [dim, hidden] projections (float tensors as the GGUF
// stores them) and their norms.
struct Qwen4IndexerWeights final {
  ops::Projection query;
  ops::Projection key;
  ops::NormWeights queryNorm;
  ops::NormWeights keyNorm;
};

// The PLE layer's projections of the gathered n-gram rows, its norms (of
// streamWidth() each) and its dilated convolution's taps
// ([streamWidth()][taps] F32: a channel's taps are adjacent).
struct Qwen4PleWeights final {
  ops::Projection key;
  ops::Projection value;
  ops::NormWeights keyNorm;
  ops::NormWeights queryNorm;
  ops::NormWeights convolutionNorm;
  metal::MetalBuffer convolution;
};

// What a qwen4exp layer carries past the shared TargetLayerWeights record
// (which holds its mixer and its MoE block): the hyper-connection mixes
// that replace its norms, the full-attention layers' QSA indexer and the
// PLE layer's tensors.
struct Qwen4ExpLayerExtras final {
  Qwen4HyperWeights mixerHyper;
  std::optional<Qwen4IndexerWeights> indexer;
  Qwen4HyperWeights ffnHyper;
  std::optional<Qwen4PleWeights> ple;
};

// The qwen4exp package's weights: the shared record's layers (each its
// mixer and MoE block; a hyper-connected layer has no norms of its own, so
// inputNorm/postAttentionNorm stay empty) plus, in extras, the same
// layers' hyper mixes, indexers and PLE tensors in the same order.
struct Qwen4ExpWeights final : TargetModelWeights {
  std::vector<Qwen4ExpLayerExtras> extras;
  // The last mix, which feeds the head in place of a final norm.
  Qwen4HyperWeights finalHyper;
  // The PLE table's native rows of pleHeadDimension values.
  ops::EmbeddingWeights pleTable;
  Qwen4PleHash pleHash;
  // The table's native row format (the packed file's row type).
  uint32_t pleFormat = 0;
  uint32_t pleEosToken = 0;
  uint32_t pleImageToken = 0;
};

[[nodiscard]] Qwen4ExpWeights
loadQwen4ExpWeights(metal::MetalBackend &backend, const TargetLayout &layout,
                    const TargetFiles &files);

// The geometry the layers' and head's projections dispatch, the base's
// plus the extras' (the hyper mixes', the indexers' and the PLE layer's).
[[nodiscard]] TargetModelGeometry
targetModelGeometry(const Qwen4ExpWeights &weights);

// The PLE rows of `tokens` from `position` on into `out`, each row's
// heads' in order: the 2- to n-gram hashes of a row's token and the ones
// before it, which come from `history` before `position`, an EOS or the
// sequence start cutting the window there, each n-gram's heads taking its
// hash modulo their row counts. Ported unchanged from the vendored
// runtime/qwen4/model/Qwen4Steps.cpp qwen4PleRows.
void qwen4ExpPleRows(const Qwen4ExpWeights &weights, uint64_t position,
                     std::span<const uint32_t> tokens,
                     std::span<const uint32_t> history, uint32_t *out);

} // namespace richengine::model
