#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// DuoAttention's per-layer bounded-attention policy (the _duo paged
// attention kernels, ops::PagedAttention): a KV head whose bit of head_mask
// is set is a streaming head — it attends only to the first sink_tokens
// tokens of its history plus the trailing window_tokens ending at each row's
// causal end, instead of the full causal history. The split kernels admit
// the pages covering those ranges; the per-token mask trims each row's own
// window exactly. Clear bits name full heads, which attend identically to
// the dense kernels — a layer whose heads are all clear remaps to the
// identity range. window_tokens 0 disables the policy entirely.
struct RichDuoAttentionPolicy {
  uint32_t sink_tokens;
  uint32_t window_tokens;
  uint64_t head_mask;
#ifndef __METAL_VERSION__
  bool operator==(const RichDuoAttentionPolicy &) const = default;
#endif
};

static_assert(sizeof(RichDuoAttentionPolicy) == 16,
              "DuoAttention policy is 16 bytes on both sides");
