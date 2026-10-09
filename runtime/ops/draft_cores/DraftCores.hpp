#pragma once

// The per-head-geometry compiled draft attention cores DraftAttention.cpp
// dispatches. One draft_cores/<geometry>.cpp resolves each geometry's
// compiled kernel names for the base ops of ops/KernelNames.hpp (a base is
// the geometry-neutral constant, e.g. kDraftAttentionQkv); adding a
// geometry adds a file here plus a HeadKernel case in DraftAttention.cpp.
#include <string_view>

namespace richengine::ops {

// 32 query heads over 8 KV heads of dimension 128: the DFlash2 draft
// blocks and the plain drafts, whose kernels keep the unsuffixed names.
[[nodiscard]] const char *draftCore32x8x128(std::string_view base);
// 16 query heads over 2 KV heads of dimension 128: MiniCPM5's DSpark
// draft, whose kernels carry the "_q16k2" suffix.
[[nodiscard]] const char *draftCore16x2x128(std::string_view base);
// 32 query heads over 8 KV heads of dimension 64 with interleaved rotary:
// LFM2.5's draft, whose kernels carry the "_q32k8d64i" suffix.
[[nodiscard]] const char *draftCore32x8x64Interleaved(std::string_view base);

} // namespace richengine::ops
