#pragma once

// NariQ trellis kernel parameters shared by host dispatch and Metal. All
// row-major activations are contiguous unless out_stride names a wider
// destination row.

struct TrellisParams {
  uint32_t input_size;   // K
  uint32_t output_size;  // N of this segment
  uint32_t rows;         // activation rows
  uint32_t out_stride;   // destination columns per row
  uint32_t out_offset;   // first destination column of this segment
  uint32_t input_block;  // normalized Sylvester block, 1 for identity
  uint32_t output_block; // normalized Sylvester block, 1 for identity
  uint32_t flags;        // bit 0 pin, bit 1 pout, bit 2 vocabulary scatter
};
static_assert(sizeof(TrellisParams) == 32, "trellis parameters are 32 bytes on both sides");

struct TrellisTransformParams {
  uint32_t length;
  uint32_t block;
  float scale;
  uint32_t reserved;
};
static_assert(sizeof(TrellisTransformParams) == 16, "trellis transform parameters are 16 bytes on both sides");

struct TrellisEmbeddingParams {
  uint32_t rows;
  uint32_t input_size;
  uint32_t input_block;
  float input_scale;
};
static_assert(sizeof(TrellisEmbeddingParams) == 16, "trellis embedding parameters are 16 bytes on both sides");
