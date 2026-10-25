#pragma once

#include "model/DFlashTwoDraft.hpp"

namespace richengine::model {

// The wide DFlash2 pathway (DFlashTwo16width): any DFlash2 draft whose
// trained block exceeds the block-8 tile keeps the compiled 16-row verify
// and draft geometry. Selected at model load for DFlashDraftWeights whose
// layout block size is above 8; the dedicated class pins the m16 fused-M
// kernel variant so a future block-16 DFlash2 has its own tuning surface
// without touching the block-8 pathway.
class DFlashTwo16width final : public DFlashDraft {
public:
  static constexpr uint32_t kTrainedVerifyWidth = 16;

  DFlashTwo16width(const DFlashDraftWeights &weights,
                   metal::MetalBackend &backend,
                   const ops::ExecutionPlans &operators);
};

} // namespace richengine::model
