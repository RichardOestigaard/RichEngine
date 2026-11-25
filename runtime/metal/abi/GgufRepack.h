#pragma once

// Weight preparation's repack of native GGUF rows into the MDGG0001 planes
// (kernels/shared/gguf_repack.metal), shared by the host executor
// (model/GgufPreparation.cpp) and the kernel. The host stages a chunk of
// rows in image order; one thread per (row, 32-wide K group) writes its
// planes into the output buffer: plane0 at 0, then plane1 and meta.
#include "metal/abi/QuantFormat.h"

struct GgufRepackParams {
  uint32_t rows;          // rows of the chunk (a multiple of QUANT_TILE_ROWS)
  uint32_t input_size;    // K of the chunk (a multiple of 256)
  uint32_t fmt;           // GGUF_FMT_*
  uint32_t src_row_bytes; // bytes per staged row
  uint32_t dst_plane1;    // byte offsets in the output buffer
  uint32_t dst_meta;
  // NVFP4's tensor-global fp32 scale (llama.cpp's separate `.scale` tensor),
  // folded into every meta unit the kernel writes; 1 for every other format.
  float scale = 1.0f;
  uint32_t reserved = 0;
};
static_assert(sizeof(GgufRepackParams) == 32, "GGUF repack parameters are 32 bytes on both sides");

// The same repack from a safetensors affine tensor's stored planes
// (gguf_repack_stored, model/AffineToBlock.cpp): the host stages the chunk's
// codes, scales and biases as the checkpoint stores them — a contiguous run
// per plane — and one thread per (row, 32-wide K group) writes the same
// plane0 and meta tiles gguf_repack writes of the native block the planes
// hold as parts.
struct GgufRepackStoredParams {
  uint32_t rows;            // rows of the chunk
  uint32_t input_size;      // K of the chunk (a multiple of the format's groups)
  uint32_t first_row;       // image row of the chunk's row 0 (its plane tile row)
  uint32_t fmt;             // GGUF_FMT_AF4
  uint32_t codes_row_bytes; // staged bytes a row of the codes plane
  uint32_t meta_row_bytes;  // of the scale plane and of the bias plane alike
  uint32_t src_scales;      // the scale and bias planes' byte offsets in src
  uint32_t src_biases;
  uint32_t dst_meta;        // the meta plane's byte offset in dst (plane0 at 0)
};
static_assert(sizeof(GgufRepackStoredParams) == 36,
              "GGUF stored-planes repack parameters are 36 bytes on both sides");
