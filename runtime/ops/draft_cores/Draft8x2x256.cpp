#include "ops/draft_cores/DraftCores.hpp"

#include "ops/KernelNames.hpp"

#include <stdexcept>

namespace richengine::ops {

// MiniCPM-V 4.7 35B-A3B's DSpark draft: 8 query heads over 2 KV heads of
// dimension 256, whose instantiations carry the "_q8k2d256" suffix.
const char *draftCore8x2x256(std::string_view base) {
  if (base == kDraftAttentionQkv)
    return kDraftAttentionQkvQ8k2d256.data();
  if (base == kDraftAttentionBf16Split)
    return kDraftAttentionBf16SplitQ8k2d256.data();
  if (base == kDraftAttentionBf16Reduce)
    return kDraftAttentionBf16ReduceQ8k2d256.data();
  if (base == kDraftAttentionReorder)
    return kDraftAttentionReorderQ8k2d256.data();
  if (base == kDraftContextKvCommit)
    return kDraftContextKvCommitQ8k2d256.data();
  if (base == kPrefillDraftContextKv)
    return kPrefillDraftContextKvQ8k2d256.data();
  throw std::invalid_argument("unknown draft attention kernel");
}

} // namespace richengine::ops
