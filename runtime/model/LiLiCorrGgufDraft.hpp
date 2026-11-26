#pragma once

// NVIDIA's LiLiCorr DFlash draft GGUF, adapted into the DFlash2 weight
// containers the draft kernels already run. The scorer tensors are parsed
// into the same image: the trunk rows first, then the lattice scorer's
// dense and NVFP4 matrices the lilicorr kernels read directly.

#include "model/DFlashTwoDraft.hpp"
#include "model/WeightImages.hpp"

#include <filesystem>

namespace richengine::model {

inline constexpr std::string_view kLiLiCorrGgufDraftMagic = "MDFL0007";

// Loads draft/lilicorr.gguf (or the single .gguf in draft/) as one block
// image. The image stores BF16 projections as bf16 float segments, NVFP4
// projections repacked with their tensor-global scales folded into meta,
// and F32 norms/conversions narrowed to the bf16 the draft kernels read.
[[nodiscard]] DFlashDraftWeights loadLiLiCorrGgufDraftWeights(
    metal::MetalBackend &backend, WeightImages &images,
    const std::filesystem::path &directory, DFlashDraftLayout layout);

// The adapted image's exact size, for memory planning before the package is
// opened. Throws the same metadata/shape errors the load does.
[[nodiscard]] uint64_t liLiCorrGgufDraftImageBytes(
    const std::filesystem::path &directory, const DFlashDraftLayout &layout);

} // namespace richengine::model
