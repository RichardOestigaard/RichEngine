#include "ops/draft_cores/DraftCores.hpp"

#include "ops/KernelNames.hpp"

#include <stdexcept>

namespace richengine::ops {

// Nemotron-3.5-Lightning's DSpark draft: 32 query heads over 2 KV heads of
// dimension 128. The QKV, split and reduce kernels carry the "_q32k2"
// suffix (the split tiles each sixteen-head KV group as two M=64 subtiles,
// the reduce binds a per-head attention-sink bias); the context writers
// see only the KV geometry, so they reuse the q16k2 instantiations.
const char *draftCore32x2x128(std::string_view base) {
  if (base == kDraftAttentionQkv)
    return kDraftAttentionQkvQ32k2.data();
  if (base == kDraftAttentionBf16Split)
    return kDraftAttentionBf16SplitQ32k2.data();
  if (base == kDraftAttentionBf16Reduce)
    return kDraftAttentionBf16ReduceQ32k2.data();
  if (base == kDraftContextKvCommit)
    return kDraftContextKvCommitQ16k2.data();
  if (base == kPrefillDraftContextKv)
    return kPrefillDraftContextKvQ16k2.data();
  throw std::invalid_argument("unknown draft attention kernel");
}

} // namespace richengine::ops
