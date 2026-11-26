#include "model/NemotronVision.hpp"
#include "Checked.hpp"
#include "model/NemotronVisionLoader.hpp"
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

NemotronVisionWeights loadNemotronVisionWeights(metal::MetalBackend &backend,
                                                WeightImages &images,
                                                const NemotronVisionLoader &source) {
  const ops::NemotronVisionLayout &layout = source.layout();
  const uint64_t allocationBaseline = backend.memoryStats().allocatedBytes;
  NemotronVisionWeights result;
  result.tensors.layout = layout;
  result.tensors.blocks.reserve(layout.depth);

  WeightFile file = images.load(source.image());
  const uint32_t positions = layout.positionGridSide * layout.positionGridSide;
  result.tensors.patchEmbedding =
      readSection(file, uint64_t{layout.hiddenSize} * layout.patchDimension,
                  "patch-embed");
  result.tensors.positionTable =
      readSection(file, uint64_t{positions} * layout.hiddenSize, "position-table");
  result.tensors.classEmbedding =
      readSection(file, uint64_t{layout.registers} * layout.hiddenSize, "registers");
  for (uint32_t blockIndex = 0; blockIndex < layout.depth; ++blockIndex) {
    ops::NemotronVisionBlock block;
    block.norm1.weight = readSection(file, layout.hiddenSize, "ln1-w");
    block.norm1.bias = readSection(file, layout.hiddenSize, "ln1-b");
    const auto affine = [&](ops::VisionAffine &affine, uint64_t rows, uint64_t columns,
                            std::string_view label) {
      affine.weight = readSection(file, rows * columns, std::string(label) + "-w");
      affine.bias = readSection(file, rows, std::string(label) + "-b");
    };
    affine(block.query, layout.hiddenSize, layout.hiddenSize, "attn-q");
    affine(block.key, layout.hiddenSize, layout.hiddenSize, "attn-k");
    affine(block.value, layout.hiddenSize, layout.hiddenSize, "attn-v");
    affine(block.projection, layout.hiddenSize, layout.hiddenSize, "attn-out");
    block.norm2.weight = readSection(file, layout.hiddenSize, "ln2-w");
    block.norm2.bias = readSection(file, layout.hiddenSize, "ln2-b");
    affine(block.upProjection, layout.intermediateSize, layout.hiddenSize, "ffn-up");
    affine(block.downProjection, layout.hiddenSize, layout.intermediateSize, "ffn-down");
    result.tensors.blocks.push_back(std::move(block));
  }
  result.tensors.projectorNorm = readSection(file, layout.mergedHiddenSize, "proj-norm");
  result.tensors.projectorUpProjection =
      readSection(file, uint64_t{layout.projectorHiddenSize} * layout.mergedHiddenSize,
                  "proj-up");
  result.tensors.projectorDownProjection =
      readSection(file, uint64_t{layout.outputHiddenSize} * layout.projectorHiddenSize,
                  "proj-down");
  file.finish();
  result.files.push_back(file.record());

  // The patch embedding and the projector's two maps are bias-free; the
  // shared GEMM kernels still read a bias row, so they bind one zeroed row
  // of the widest output.
  const uint64_t zeroBytes =
      checkedMultiply<WeightStoreError>(
          uint64_t{std::max(layout.projectorHiddenSize, layout.hiddenSize)},
          kBFloat16Bytes, "vision zero-bias bytes");
  result.tensors.zeroBias =
      backend.allocateBuffer(zeroBytes, metal::BufferStorage::Shared, "vision-zero-bias");
  std::memset(result.tensors.zeroBias.contents(), 0, zeroBytes);

  result.actualAllocatedBytes = metal::allocationDelta(
      allocationBaseline, backend.memoryStats().allocatedBytes);
  return result;
}

} // namespace richengine::model
