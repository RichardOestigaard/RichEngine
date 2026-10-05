#pragma once

#include "DFlashDraft.hpp"
#include "WeightStore.hpp"
#include "ops/DraftSelector.hpp"
#include "ops/Linear.hpp"
#include "ops/Normalization.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace richengine::model {

// A target without a GPU draft (Granite 4.2: no DFlash/DSpark/MTP checkpoint
// exists). The descriptor still carries a valid draft layout so geometry
// invariants hold, but every encode point is a no-op: the n-gram predraft
// always supplies the proposals the verify path consumes.
struct NullDraftWeights final {
  DFlashDraftLayout layout;
  std::vector<WeightFileRecord> files;
  uint64_t actualAllocatedBytes = 0;
};

class NullDraft final {
public:
  NullDraft() = default;

  void addContextPrefill(metal::CommandGraph &, DFlashPrefillBuffers,
                         uint32_t, std::span<const DFlashPrefillSpan>) const {}
  void addDecode(metal::CommandGraph &, DFlashDecodeBuffers,
                 const ops::Projection &, std::span<const uint32_t>) const {}
  void addSelection(metal::CommandGraph &, const ops::DraftSelectorBuffers &,
                    std::span<const uint32_t>,
                    std::span<const ops::SamplingPolicy>, uint32_t) const {}
  void addContextCommit(metal::CommandGraph &, DFlashContextBuffers,
                        std::span<const uint32_t>) const {}
};

} // namespace richengine::model
