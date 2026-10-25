#pragma once

// The per-family descriptor makers and draft layouts kModelFamilies
// dispatches and the manifest parsers swap in (ModelDescriptor.mm). One
// model/families/<family>.mm defines each family's makers; shared JSON
// validation stays in ModelDescriptor.mm.
#include "model/ModelDescriptor.hpp"

#include <string>

namespace richengine::model {

// Qwen 3.8 dense (Qwen3.8-27B) and Qwen 3.6 MoE
// (Qwen3.6-35B-A3B, Ornith-1.5-35B-A3B).
[[nodiscard]] ModelDescriptor qwen38Descriptor(std::string name);
[[nodiscard]] ModelDescriptor qwen36Descriptor(std::string name);
// The 27B family's DFlash2 default.
[[nodiscard]] DFlashDraftLayout qwen38DraftLayout();
// The 35B-A3B family's DFlash2 default and released plain DFlash draft.
[[nodiscard]] DFlashDraftLayout qwen36DraftLayout();
[[nodiscard]] DFlashDraftLayout qwen36DFlashV1DraftLayout();

// Ornith 1.5 9B.
[[nodiscard]] ModelDescriptor ornithDescriptor(std::string name);
// Ornith 9B's released plain DFlash draft (no DFlash2 layout exists — no
// compiled context core covers hidden 4096/dynamic 1024).
[[nodiscard]] DFlashDraftLayout ornith9DFlashV1DraftLayout();

// MiniCPM5-2B.
[[nodiscard]] ModelDescriptor denseDescriptor(std::string name);
[[nodiscard]] DFlashDraftLayout denseDraftLayout();
// The dense family's DFlash2 layout, swapped in when an assembly's draft
// declares DFlash2DraftModel instead of the DSpark default.
[[nodiscard]] DFlashDraftLayout denseDFlash2DraftLayout();

// MiniCPM-V-4.7-35B-A3B: the qwen3_5_moe backbone with the vision tower this
// runtime does not serve, and its DSpark draft.
[[nodiscard]] ModelDescriptor minicpmV47Descriptor(std::string name);
[[nodiscard]] DFlashDraftLayout minicpmV47DraftLayout();

// LFM2.5-2.6B and LFM2.5-8B-A1B.
[[nodiscard]] ModelDescriptor lfm2Descriptor(std::string name);
[[nodiscard]] ModelDescriptor d1Descriptor(std::string name);
[[nodiscard]] ModelDescriptor lfm2moeDescriptor(std::string name);
[[nodiscard]] DFlashDraftLayout lfm2DraftLayout();
[[nodiscard]] DFlashDraftLayout lfm2moeDraftLayout();

// LFM2.5-24B-A2B: the 40-layer lfm2_moe target, its Null draft layout and
// the released DSpark draft's.
[[nodiscard]] ModelDescriptor lfm2moe24BDescriptor(std::string name);
[[nodiscard]] DFlashDraftLayout lfm2moe24BDraftLayout();
[[nodiscard]] DFlashDraftLayout lfm2moe24BDSparkDraftLayout();

// Granite-4.2-3B and Granite-4.2-8B.
[[nodiscard]] ModelDescriptor granite3BDescriptor(std::string name);
[[nodiscard]] ModelDescriptor granite8BDescriptor(std::string name);

// Gemma4-26B-A4B and DiffusionGemma-26B-A4B.
[[nodiscard]] ModelDescriptor gemma4Descriptor(std::string name);
[[nodiscard]] ModelDescriptor diffusionGemmaDescriptor(std::string name);
// The plain DFlash layout a packed gemma4 manifest's draft declaration
// swaps in for the Null default.
[[nodiscard]] DFlashDraftLayout gemma4DFlashV1DraftDefaults();

} // namespace richengine::model
