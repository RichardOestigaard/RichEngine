#include "ops/draft_cores/DraftCores.hpp"

#include "ops/KernelNames.hpp"

#include <stdexcept>

namespace richengine::ops {

// MiniCPM5's DSpark draft: 16 query heads over 2 KV heads of dimension
// 128, whose instantiations carry the "_q16k2" suffix.
const char *draftCore16x2x128(std::string_view base) {
  if (base == kDraftAttentionQkv)
    return kDraftAttentionQkvQ16k2.data();
  if (base == kDraftAttentionBf16Split)
    return kDraftAttentionBf16SplitQ16k2.data();
  if (base == kDraftAttentionBf16Reduce)
    return kDraftAttentionBf16ReduceQ16k2.data();
  if (base == kDraftAttentionReorder)
    return kDraftAttentionReorderQ16k2.data();
  if (base == kDraftContextKvCommit)
    return kDraftContextKvCommitQ16k2.data();
  if (base == kPrefillDraftContextKv)
    return kPrefillDraftContextKvQ16k2.data();
  throw std::invalid_argument("unknown draft attention kernel");
}

} // namespace richengine::ops
