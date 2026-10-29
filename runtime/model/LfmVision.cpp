#include "model/LfmVision.hpp"
#include "Checked.hpp"
#include "model/LfmVisionLoader.hpp"
#include "model/WeightLayout.hpp"

#include <cstring>
#include <string>
#include <utility>

namespace richengine::model {
namespace {

metal::MetalBuffer readSection(WeightFile &file, uint64_t elements,
                               std::string_view label) {
  return file.section(
      checkedMultiply<WeightStoreError>(elements, kBFloat16Bytes,
                                        "vision weight bytes"),
      label);
}

} // namespace

LfmVisionWeights loadLfmVisionWeights(metal::MetalBackend &backend,
                                      WeightImages &images,
                                      const LfmVisionLoader &source) {
  const ops::LfmVisionLayout &layout = source.layout();
  const uint64_t allocationBaseline = backend.memoryStats().allocatedBytes;
  LfmVisionWeights result;
  result.tensors.layout = layout;
  result.tensors.blocks.reserve(layout.depth);

  WeightFile file = images.load(source.image());
  const auto kv = uint64_t{layout.kvHeads} * layout.headDimension;
  result.tensors.patchEmbedding =
      readSection(file, uint64_t{layout.hiddenSize} * layout.patchDimension,
                  "patch-embed");
  result.tensors.embeddingNorm = readSection(file, layout.hiddenSize, "embed-norm");
  const std::vector<bool> &conv = source.convBlocks();
  for (uint32_t blockIndex = 0; blockIndex < layout.depth; ++blockIndex) {
    ops::LfmVisionBlock block;
    block.norm1.weight = readSection(file, layout.hiddenSize, "ln1");
    if ((block.conv = conv[blockIndex])) {
      block.inProjection = readSection(
          file, uint64_t{3} * layout.hiddenSize * layout.hiddenSize, "conv-in");
      block.convKernel = readSection(file, uint64_t{9} * layout.hiddenSize, "conv-dw");
      block.outProjection = readSection(
          file, uint64_t{layout.hiddenSize} * layout.hiddenSize, "conv-out");
    } else {
      block.query = readSection(
          file, uint64_t{layout.hiddenSize} * layout.hiddenSize, "attn-q");
      block.key = readSection(file, kv * layout.hiddenSize, "attn-k");
      block.value = readSection(file, kv * layout.hiddenSize, "attn-v");
      block.queryNorm = readSection(file, layout.headDimension, "attn-q-norm");
      block.keyNorm = readSection(file, layout.headDimension, "attn-k-norm");
      block.outProjection = readSection(
          file, uint64_t{layout.hiddenSize} * layout.hiddenSize, "attn-out");
    }
    block.norm2.weight = readSection(file, layout.hiddenSize, "ln2");
    block.ffnGate = readSection(
        file, uint64_t{layout.intermediateSize} * layout.hiddenSize, "ffn-gate");
    block.ffnUp = readSection(
        file, uint64_t{layout.intermediateSize} * layout.hiddenSize, "ffn-up");
    block.ffnDown = readSection(
        file, uint64_t{layout.intermediateSize} * layout.hiddenSize, "ffn-down");
    result.tensors.blocks.push_back(std::move(block));
  }
  result.tensors.mergerNorm.weight = readSection(file, layout.hiddenSize, "proj-norm-w");
  result.tensors.mergerNorm.bias = readSection(file, layout.hiddenSize, "proj-norm-b");
  result.tensors.mergerUpProjection = {
      readSection(file,
                  uint64_t{layout.mergedHiddenSize} * layout.mergedHiddenSize,
                  "proj-up-w"),
      readSection(file, layout.mergedHiddenSize, "proj-up-b")};
  result.tensors.mergerDownProjection = {
      readSection(file,
                  uint64_t{layout.outputHiddenSize} * layout.mergedHiddenSize,
                  "proj-down-w"),
      readSection(file, layout.outputHiddenSize, "proj-down-b")};
  file.finish();
  result.files.push_back(file.record());

  // The tower's projections are bias-free; the shared GEMM kernels still
  // read a bias row, so they bind one zeroed row of the widest output.
  const uint64_t zeroBytes =
      checkedMultiply<WeightStoreError>(
          uint64_t{std::max(layout.mergedHiddenSize, 3 * layout.hiddenSize)},
          kBFloat16Bytes, "vision zero-bias bytes");
  result.tensors.zeroBias =
      backend.allocateBuffer(zeroBytes, metal::BufferStorage::Shared, "vision-zero-bias");
  std::memset(result.tensors.zeroBias.contents(), 0, zeroBytes);

  result.actualAllocatedBytes = metal::allocationDelta(
      allocationBaseline, backend.memoryStats().allocatedBytes);
  return result;
}

} // namespace richengine::model
