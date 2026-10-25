#pragma once

#include "model/DFlashTwoDraft.hpp"

namespace richengine::model {

// The block-8 DFlash2 pathway (DFlashTwo8width): the draft forward and the
// verify tile both run the trained eight rows per lane — upstream splash's
// native 8-row geometry (SPLASH_TARGET_VERIFY_ROWS=8). Selected at model
// load for any DFlashDraftWeights whose layout block size is at most 8;
// wider DFlash2 drafts take DFlashTwo16width. The class pins the m8 fused-M
// kernel variants so the 27B/35B DFlash2 family runs upstream-shaped tiles.
class DFlashTwo8width final : public DFlashDraft {
public:
  static constexpr uint32_t kTrainedVerifyWidth = 8;

  DFlashTwo8width(const DFlashDraftWeights &weights,
                  metal::MetalBackend &backend,
                  const ops::ExecutionPlans &operators);
};

} // namespace richengine::model
