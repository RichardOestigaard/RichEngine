#pragma once

// Parameter layouts shared by ops::LfmAudioEnc.mm and lfm_audio.metal.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

struct LfmAudioConvParams {
  uint32_t in_freq;
  uint32_t in_time;
  uint32_t in_channels;
  uint32_t out_freq;
  uint32_t out_time;
  uint32_t out_channels;
  // Input transform the consumer applies on load: 0 = bf16 rows, 1 = bf16
  // rows passed through SiLU (the silu pass folds into this kernel), 2 =
  // float32 rows (the mel, so no bf16 pack pass runs at all).
  uint32_t input_mode;
  // Element offset of this stage's weights inside the bound weight buffer:
  // the lfm_audio_conv_wt repack transposes all three stages (oc-major ->
  // tap-major) into one scratch so the conv loads them as bfloat4.
  uint32_t weight_offset;
};

static_assert(sizeof(LfmAudioConvParams) == 32,
              "audio conv parameters are 32 bytes on both sides");

struct LfmAudioCountParams {
  uint32_t count;
};

static_assert(sizeof(LfmAudioCountParams) == 4,
              "audio count parameters are 4 bytes on both sides");

struct LfmAudioFlattenParams {
  uint32_t tokens;   // encoder positions (mel frames / 8, ceiling)
  uint32_t frames;   // the last conv stage's out_time, == tokens
  uint32_t freq;     // 16
  uint32_t channels; // 256
  // Nonzero: the input rows pass through SiLU on load, folding the
  // standalone silu pass into the gather.
  uint32_t input_silu;
};

static_assert(sizeof(LfmAudioFlattenParams) == 20,
              "audio flatten parameters are 20 bytes on both sides");

struct LfmAudioQkvParams {
  uint32_t tokens;
  uint32_t padded_tokens;
  // Nonzero when the q/k/v inputs are one fused [Q | K | V]-per-token
  // buffer produced by lfm_audio_gemm_qkv instead of three separate
  // projection outputs.
  uint32_t fused = 0;
};

static_assert(sizeof(LfmAudioQkvParams) == 12,
              "audio qkv parameters are 12 bytes on both sides");

struct LfmAudioAttentionParams {
  uint32_t tokens;
  uint32_t padded_tokens;
  float scale;
};

static_assert(sizeof(LfmAudioAttentionParams) == 12,
              "audio attention parameters are 12 bytes on both sides");
