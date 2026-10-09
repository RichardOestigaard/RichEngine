#include "ops/draft_cores/DraftCores.hpp"

#include "ops/KernelNames.hpp"

#include <stdexcept>

namespace richengine::ops {

// The original compiled draft cores: 32x8x128's instantiations keep the
// base names unsuffixed.
const char *draftCore32x8x128(std::string_view base) {
  if (base == kDraftAttentionQkv)
    return kDraftAttentionQkv.data();
  if (base == kDraftAttentionBf16Split)
    return kDraftAttentionBf16Split.data();
  if (base == kDraftAttentionBf16Reduce)
    return kDraftAttentionBf16Reduce.data();
  if (base == kDraftAttentionReorder)
    return kDraftAttentionReorder.data();
  if (base == kDraftContextKvCommit)
    return kDraftContextKvCommit.data();
  if (base == kPrefillDraftContextKv)
    return kPrefillDraftContextKv.data();
  throw std::invalid_argument("unknown draft attention kernel");
}

} // namespace richengine::ops
