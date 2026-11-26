#pragma once

// Parameter layouts shared by the LiLiCorr scorer host code and kernels.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

#define LILICORR_SLOTS 7u
#define LILICORR_TOP_K 8u
#define LILICORR_NODES (LILICORR_SLOTS * LILICORR_TOP_K)
#define LILICORR_HIDDEN 1024u
#define LILICORR_TARGET_HIDDEN 4096u
#define LILICORR_HEADS 8u
#define LILICORR_HEAD_DIM (LILICORR_HIDDEN / LILICORR_HEADS)
#define LILICORR_MLP_HIDDEN 2048u
#define LILICORR_FEATURE_WIDTH 5u
#define LILICORR_EDGES (LILICORR_TOP_K * LILICORR_TOP_K)

// One lane's fp32 scratch image. The offsets are byte offsets into the lane's
// stride and every section is at least 256-byte aligned.
#define LILICORR_SCRATCH_TOKEN_INPUT 0u
#define LILICORR_SCRATCH_PASS_INPUT 0x000e0000u
#define LILICORR_SCRATCH_FEATURES 0x00100000u
#define LILICORR_SCRATCH_TOKENS 0x00100480u
#define LILICORR_SCRATCH_TOKEN_STATES 0x00100600u
#define LILICORR_SCRATCH_PASS_STATES 0x00138600u
#define LILICORR_SCRATCH_FEATURE_NORMED 0x0013f600u
#define LILICORR_SCRATCH_FEATURE_UP 0x00177600u
#define LILICORR_SCRATCH_FEATURE_STATES 0x001af600u
#define LILICORR_SCRATCH_X 0x001e7600u
#define LILICORR_SCRATCH_NORMED 0x0021f600u
#define LILICORR_SCRATCH_QKV 0x00257600u
#define LILICORR_SCRATCH_ATTENTION_OUT 0x002ff600u
#define LILICORR_SCRATCH_MLP 0x00337600u
#define LILICORR_SCRATCH_FACTOR_INPUT 0x003a7600u
#define LILICORR_SCRATCH_FACTORS 0x0044f600u
#define LILICORR_SCRATCH_OUT 0x00487600u
#define LILICORR_SCRATCH_IN 0x004bf600u
#define LILICORR_SCRATCH_ANCHOR_STATE 0x004f7600u
#define LILICORR_SCRATCH_ANCHOR_NORMED 0x004f8600u
#define LILICORR_SCRATCH_ANCHOR_OUT 0x004f9600u
#define LILICORR_SCRATCH_SCORES 0x004fa600u
#define LILICORR_SCRATCH_ANCHOR_INPUT 0x004fb000u
#define LILICORR_SCRATCH_STRIDE 0x00500000u

struct LiLiCorrFeatureParams {
  uint32_t lanes;
  uint32_t vocabulary;
  uint32_t logits_stride;
  uint32_t query_rows;
};

static_assert(sizeof(LiLiCorrFeatureParams) == 16,
              "LiLiCorr feature parameters are 16 bytes on both sides");

struct LiLiCorrGemmParams {
  uint32_t rows;
  uint32_t input_size;
  uint32_t output_size;
  uint32_t rows_per_lane;
  uint32_t lane_stride;
  uint32_t input_stride;
  uint32_t output_stride;
  uint32_t epilogue;
  float scale;
};

static_assert(sizeof(LiLiCorrGemmParams) == 36,
              "LiLiCorr GEMM parameters are 36 bytes on both sides");

struct LiLiCorrNormParams {
  uint32_t rows;
  uint32_t width;
  uint32_t rows_per_lane;
  uint32_t lane_stride;
  uint32_t input_stride;
  uint32_t output_stride;
  float epsilon;
};

static_assert(sizeof(LiLiCorrNormParams) == 28,
              "LiLiCorr norm parameters are 28 bytes on both sides");

struct LiLiCorrAttentionParams {
  uint32_t lanes;
  uint32_t lane_stride;
  float scale;
};

static_assert(sizeof(LiLiCorrAttentionParams) == 12,
              "LiLiCorr attention parameters are 12 bytes on both sides");

struct LiLiCorrWalkParams {
  uint32_t lanes;
  uint32_t lane_stride;
  float vector_eps;
  float logit_scale;
};

static_assert(sizeof(LiLiCorrWalkParams) == 16,
              "LiLiCorr walk parameters are 16 bytes on both sides");

struct LiLiCorrCopyParams {
  uint32_t width;
  uint32_t source_row;
};

static_assert(sizeof(LiLiCorrCopyParams) == 8,
              "LiLiCorr copy parameters are 8 bytes on both sides");
