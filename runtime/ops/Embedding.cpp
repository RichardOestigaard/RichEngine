#include "ops/Embedding.hpp"

#include "metal/abi/Embedding.h"
#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/Gguf.h"
#include "metal/abi/Sampling.h"
#include "ops/KernelNames.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace richengine::ops {

NativeRows::NativeRows(metal::MetalBuffer rows, uint32_t formatId) : rows(std::move(rows)), formatId(formatId) {
  if (!gguf_embedding_format(formatId)) throw std::invalid_argument("unsupported native embedding format");
}
const char *NativeRows::name() const noexcept { return kQuantFormats[formatId].name; }

void Embedding::add(metal::CommandGraph &graph, metal::MetalBuffer tokens,
                    const EmbeddingWeights &table, metal::MetalBuffer output,
                    uint32_t rows, float scale) {
  if (!rows || !table.outputSize || !table.inputSize)
    throw std::invalid_argument("invalid Q4 embedding shape");
  // Both gathers read `rows` token ids and write `rows` bf16 rows of the table's width.
  if (tokens.sizeBytes() < uint64_t{rows} * sizeof(uint32_t) ||
      output.sizeBytes() < uint64_t{rows} * table.inputSize * sizeof(uint16_t))
    throw std::invalid_argument("embedding buffers are smaller than the gathered rows");
  if (table.layout() == WeightLayout::Block32) {
    const NativeRows &native = table.blocks();
    const GgufEmbedParams params{rows, table.outputSize, table.inputSize};
    if (table.rotation) {
      // One threadgroup per rotation block of a row, which gathers the block
      // and inverts its rotation in fp32 (kernels/shared/gguf_rotation.metal).
      if (native.formatId != GGUF_FMT_PQ20 || table.inputSize % GGUF_ROTATION_BLOCK ||
          table.rotation.signs.sizeBytes() < table.inputSize)
        throw std::invalid_argument("a rotated token table takes PQ2_0 rows of whole rotation blocks and their signs");
      // The GGUF decode kernels produce no output when they replay from an
      // indirect command buffer on this driver (the gguf expert kernels fail
      // the same way): never mark them.
      graph.add(std::string(kGgufEmbedRotatedPq20), {std::move(tokens), native.rows, table.rotation.signs, std::move(output)},
                params, {table.inputSize / GGUF_ROTATION_BLOCK, rows, 1}, {GGUF_ROTATION_THREADS, 1, 1});
      return;
    }
    graph.add(std::string(kGgufEmbed) + native.name(),
              {std::move(tokens), native.rows, std::move(output)}, params,
              {(rows * table.inputSize + 255) / 256, 1, 1}, {256, 1, 1});
    return;
  }
  const uint32_t hiddenGroups = (table.inputSize + 127) / 128;
  const Q4EmbeddingParams params{rows, table.outputSize};
  const AffineWeights &affine = table.affine();
  if (scale != 0.0F) {
    // Gemma's scaled gather (embedding_q4_scaled_h2816 only; other widths
    // have no scaled kernel).
    if (table.inputSize != 2816)
      throw std::invalid_argument("no scaled embedding kernel for this width");
    graph.addTail(std::string(kEmbeddingQ4ScaledH) + std::to_string(table.inputSize),
                  {std::move(tokens), affine.weights, affine.scales, affine.biases,
                   std::move(output)},
                  params, scale, {hiddenGroups, 1, 1});
    return;
  }
  // One kernel per compiled hidden size (kernels/shared/embedding.metal);
  // the tiled variants read a packed install's 256-row planes.
  // Unmarked: the prefill path shares this op, and a one-dispatch span's
  // cached ICB and params arena cost more than its encode saves.
  graph.add(std::string(table.tiled ? kEmbeddingQ4tH : kEmbeddingQ4H) +
                std::to_string(table.inputSize),
            {std::move(tokens), affine.weights, affine.scales, affine.biases, std::move(output)},
            params, {hiddenGroups, 1, 1});
}

void Embedding::addVerifyInput(metal::CommandGraph &graph,
                               metal::MetalBuffer draftInputTokens,
                               metal::MetalBuffer proposedTokens,
                               metal::MetalBuffer verifyInputTokens,
                               uint32_t vocabulary, uint32_t lanes) {
  if (!vocabulary || !lanes || lanes > RICHENGINE_MAXIMUM_BATCH_WIDTH)
    throw std::invalid_argument("invalid verify input batch");
  const VerifyInputBatchParams params{vocabulary};
  // One static-parameter dispatch over stable arena buffers: replayable.
  graph.beginBakedSpan();
  graph.add(std::string(kVerifyInputTokens),
            {std::move(draftInputTokens), std::move(proposedTokens),
             std::move(verifyInputTokens)},
            params, {uint64_t{lanes} * RICHENGINE_TARGET_VERIFY_ROWS, 1, 1},
            {1, 1, 1});
  graph.endBakedSpan();
}

void Embedding::addVerifyTreeInput(
    metal::CommandGraph &graph, metal::MetalBuffer treeTokens,
    metal::MetalBuffer treeNodes, metal::MetalBuffer treeCounts,
    metal::MetalBuffer verifyInputTokens, metal::MetalBuffer positions,
    metal::MetalBuffer masks, const uint32_t base[][3], uint32_t vocabulary,
    uint32_t maskToken, uint32_t lanes) {
  if (!vocabulary || !lanes || lanes > RICHENGINE_MAXIMUM_BATCH_WIDTH || !base)
    throw std::invalid_argument("invalid verify tree input batch");
  VerifyTreeInputParams params{};
  params.vocabulary = vocabulary;
  params.mask_token = maskToken;
  for (uint32_t lane = 0; lane < lanes; ++lane)
    for (uint32_t axis = 0; axis < 3; ++axis)
      params.base[lane][axis] = base[lane][axis];
  graph.add(std::string(kVerifyInputTreeTokens),
            {std::move(treeTokens), std::move(treeNodes),
             std::move(treeCounts), std::move(verifyInputTokens),
             std::move(positions), std::move(masks)},
            params,
            {uint64_t{lanes} * RICHENGINE_TREE_VERIFY_NODES, 1, 1}, {1, 1, 1});
}

void Embedding::addTreeCaptureGather(
    metal::CommandGraph &graph, metal::MetalBuffer source,
    metal::MetalBuffer retainedPath, metal::MetalBuffer retained,
    metal::MetalBuffer destination, uint32_t width, uint32_t lanes) {
  if (!width || !lanes || lanes > RICHENGINE_MAXIMUM_BATCH_WIDTH)
    throw std::invalid_argument("invalid tree capture gather");
  graph.add(std::string(kTreeCaptureGather),
            {std::move(source), std::move(retainedPath), std::move(retained),
             std::move(destination)},
            width,
            {uint64_t{lanes} * RICHENGINE_TARGET_VERIFY_ROWS * width, 1, 1},
            {1, 1, 1});
}

} // namespace richengine::ops
