#include "model/families/FamilyMakers.hpp"

namespace richengine::model {

DFlashDraftLayout qwen36DraftLayout() {
  DFlashDraftLayout layout;
  layout.layers = 6;
  layout.hiddenSize = 2048;
  layout.dynamicSize = 512;
  layout.intermediateSize = 6144;
  layout.targetHiddenSize = 16384;
  layout.blockSize = 8;
  // The pool selector stays off: on the decode benchmark its walk never
  // rescues a rejected position (0/60 target picks at the pool argmax) while
  // the 128-wide edge table costs ~7% of decode throughput re-reading
  // successor codebook rows. RICHENGINE_DFLASH_POOL=1 re-enables it.
  return layout;
}

// Ornith 1.5 35B-A3B's released DFlash draft (ornith-ai/Ornith-1.5-35B-A3B-DFlash):
// the same plain structure over hidden 2048 (attention stays 4096 wide).
DFlashDraftLayout qwen36DFlashV1DraftLayout() {
  DFlashDraftLayout layout;
  layout.kind = DraftKind::DFlashV1;
  layout.slidingWindow = 4096;
  layout.causalLayers = 0x1F;
  layout.layers = 6;
  layout.hiddenSize = 2048;
  layout.dynamicSize = 0;
  layout.qkvSize = 6144;
  layout.attentionSize = 4096;
  layout.intermediateSize = 6144;
  layout.targetHiddenSize = 16384;
  layout.selectorRank = 0;
  layout.kvHeads = 8;
  layout.blockSize = 16;
  return layout;
}

ModelDescriptor qwen38Descriptor(std::string name) {
  return makeModelDescriptor(std::move(name), Qwen3_8Layout{},
                             DFlashDraftLayout{}, ops::VisionLayout{});
}

ModelDescriptor qwen36Descriptor(std::string name) {
  constexpr Qwen3_6MoeLayout target;
  ops::VisionLayout vision;
  vision.outputHiddenSize = target.hiddenSize;
  return makeModelDescriptor(std::move(name), target, qwen36DraftLayout(),
                             vision);
}

} // namespace richengine::model
