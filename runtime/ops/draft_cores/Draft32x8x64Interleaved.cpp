#include "ops/draft_cores/DraftCores.hpp"

#include "ops/KernelNames.hpp"

#include <stdexcept>

namespace richengine::ops {

// LFM2.5's draft: 32 query heads over 8 KV heads of dimension 64 with
// interleaved rotary, whose instantiations carry the "_q32k8d64i" suffix.
const char *draftCore32x8x64Interleaved(std::string_view base) {
  if (base == kDraftAttentionQkv)
    return kDraftAttentionQkvQ32k8d64i.data();
  if (base == kDraftAttentionBf16Split)
    return kDraftAttentionBf16SplitQ32k8d64i.data();
  if (base == kDraftAttentionBf16Reduce)
    return kDraftAttentionBf16ReduceQ32k8d64i.data();
  if (base == kDraftAttentionReorder)
    return kDraftAttentionReorderQ32k8d64i.data();
  if (base == kDraftContextKvCommit)
    return kDraftContextKvCommitQ32k8d64i.data();
  if (base == kPrefillDraftContextKv)
    return kPrefillDraftContextKvQ32k8d64i.data();
  throw std::invalid_argument("unknown draft attention kernel");
}

} // namespace richengine::ops
