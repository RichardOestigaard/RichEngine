#pragma once

// The maxwell-moe MTP draft's scratch views (model/MaxwellMtpDraft.hpp):
// the per-lane decode slice, the context-commit/prefill slice and the
// prefill chunk size, shared by the draft's three call sites.

#include "metal/MetalBackend.hpp"
#include "ops/MoE.hpp"

#include <cstdint>

namespace richengine::model {

// The maxwell-moe MTP draft's decode scratch, per dispatch lane. `streams`
// is the decoder's fp32 multistream state the chained passes carry;
// `eh`/`ehOut` the fused input projection's bf16 operand and output;
// `plane`/`mappings`/`headWeights` the mHC fp32 scratch; the rest the
// layer's ordinary rows and the pass token.
struct MtpDecodeScratch final {
  ops::MoeScratch moe{};
  metal::MetalBuffer streams;
  metal::MetalBuffer eh;
  metal::MetalBuffer ehOut;
  metal::MetalBuffer plane;
  metal::MetalBuffer mappings;
  metal::MetalBuffer headWeights;
  metal::MetalBuffer embed;
  metal::MetalBuffer embedNormed;
  metal::MetalBuffer aggregate;
  metal::MetalBuffer normed;
  metal::MetalBuffer queries;
  metal::MetalBuffer keys;
  metal::MetalBuffer values;
  metal::MetalBuffer attention;
  metal::MetalBuffer branch;
  metal::MetalBuffer shared;
  metal::MetalBuffer intermediate;
  metal::MetalBuffer head;
  metal::MetalBuffer logits;
  metal::MetalBuffer tokenIn;
  // The nemotron MTP draft's sigmoid-MoE scratch (empty on maxwell):
  // moeRoutes the nmtp_moe_route records (16 u32 per row — top-K ids and
  // fp32 weights) and moePartials the down pass's fp32 per-slot outputs
  // ([kTopK + 1][hidden] per row).
  metal::MetalBuffer moeRoutes;
  metal::MetalBuffer moePartials;
};

// The same scratch at context-commit and prefill rows (the retained verify
// rows or a prefill span chunk): the MTP path runs the eh projection and
// attention mHC module to write the rows' K/V, no MoE or head.
struct MtpContextScratch final {
  metal::MetalBuffer streams;
  metal::MetalBuffer eh;
  metal::MetalBuffer ehOut;
  metal::MetalBuffer plane;
  metal::MetalBuffer mappings;
  metal::MetalBuffer tokens;
  metal::MetalBuffer embed;
  metal::MetalBuffer embedNormed;
  metal::MetalBuffer keys;
  metal::MetalBuffer values;
};

// The rows of a prefill span each MTP context chunk processes.
inline constexpr uint32_t kMtpPrefillChunk = 256;

} // namespace richengine::model
