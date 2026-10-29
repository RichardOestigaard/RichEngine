#include "model/LfmAudio.hpp"
#include "Checked.hpp"
#include "model/LfmAudioLoader.hpp"
#include "model/WeightLayout.hpp"

#include <cstring>
#include <string>
#include <utility>

namespace richengine::model {
namespace {

metal::MetalBuffer readAudioSection(WeightFile &file, uint64_t elements,
                                    std::string_view label) {
  return file.section(
      checkedMultiply<WeightStoreError>(elements, kBFloat16Bytes,
                                        "audio weight bytes"),
      label);
}

} // namespace

LfmAudioWeights loadLfmAudioWeights(metal::MetalBackend &backend,
                                    WeightImages &images,
                                    const LfmAudioLoader &source) {
  const ops::LfmAudioLayout &layout = source.layout();
  const uint64_t allocationBaseline = backend.memoryStats().allocatedBytes;
  LfmAudioWeights result;
  result.tensors.layout = layout;
  result.tensors.blocks.reserve(layout.depth);

  WeightFile file = images.load(source.image());
  const auto read = [&](uint64_t elements, std::string_view label) {
    return readAudioSection(file, elements, label);
  };
  const auto kv = uint64_t{layout.kvHeads} * layout.headDimension;
  const auto conv = uint64_t{layout.convChannels};
  const auto hidden = uint64_t{layout.hiddenSize};
  result.tensors.convInWeight = read(conv * 9, "conv-in-w");
  result.tensors.convInBias = read(conv, "conv-in-b");
  result.tensors.convMidWeight = read(conv * 9, "conv-mid-w");
  result.tensors.convMidBias = read(conv, "conv-mid-b");
  result.tensors.linMidWeight = read(conv * conv, "lin-mid-w");
  result.tensors.linMidBias = read(conv, "lin-mid-b");
  result.tensors.convEndWeight = read(conv * 9, "conv-end-w");
  result.tensors.convEndBias = read(conv, "conv-end-b");
  result.tensors.linEndWeight = read(conv * conv, "lin-end-w");
  result.tensors.linEndBias = read(conv, "lin-end-b");
  result.tensors.preEncodeWeight =
      read(hidden * (layout.melBins / layout.tokenStride * conv), "pre-encode-w");
  result.tensors.preEncodeBias = read(hidden, "pre-encode-b");
  const std::vector<bool> &convBlocks = source.convBlocks();
  for (uint32_t blockIndex = 0; blockIndex < layout.depth; ++blockIndex) {
    ops::LfmAudioBlock block;
    block.norm1.weight = read(hidden, "ln1");
    if ((block.conv = convBlocks[blockIndex])) {
      block.convPw1 = read(3 * hidden * hidden, "conv-pw1");
      block.convDw = read(3 * hidden, "conv-dw");
      block.convPw2 = read(hidden * hidden, "conv-pw2");
    } else {
      block.query = read(hidden * hidden, "attn-q");
      block.key = read(kv * hidden, "attn-k");
      block.value = read(kv * hidden, "attn-v");
      block.queryNorm = read(layout.headDimension, "attn-q-norm");
      block.keyNorm = read(layout.headDimension, "attn-k-norm");
      block.outProjection = read(hidden * hidden, "attn-out");
    }
    block.norm2.weight = read(hidden, "ln2");
    block.ffnGate = read(uint64_t{layout.intermediateSize} * hidden, "ffn-gate");
    block.ffnUp = read(uint64_t{layout.intermediateSize} * hidden, "ffn-up");
    block.ffnDown = read(hidden * layout.intermediateSize, "ffn-down");
    result.tensors.blocks.push_back(std::move(block));
  }
  result.tensors.postNorm = read(hidden, "post-ln");
  result.tensors.projectorNorm = {read(hidden, "proj-norm-w"),
                                  read(hidden, "proj-norm-b")};
  result.tensors.projectorUp = {
      read(uint64_t{layout.outputHiddenSize} * hidden, "proj-up-w"),
      read(uint64_t{layout.outputHiddenSize}, "proj-up-b")};
  result.tensors.projectorDown = {
      read(uint64_t{layout.outputHiddenSize} * layout.outputHiddenSize,
           "proj-down-w"),
      read(uint64_t{layout.outputHiddenSize}, "proj-down-b")};
  file.finish();
  result.files.push_back(file.record());

  // The block projections are bias-free; the shared GEMM kernels still
  // read a bias row, so they bind one zeroed row of the widest output.
  const uint64_t zeroBytes = checkedMultiply<WeightStoreError>(
      uint64_t{std::max<uint64_t>(3 * hidden, layout.intermediateSize)},
      kBFloat16Bytes, "audio zero-bias bytes");
  result.tensors.zeroBias =
      backend.allocateBuffer(zeroBytes, metal::BufferStorage::Shared,
                             "audio-zero-bias");
  std::memset(result.tensors.zeroBias.contents(), 0, zeroBytes);

  result.actualAllocatedBytes = metal::allocationDelta(
      allocationBaseline, backend.memoryStats().allocatedBytes);
  return result;
}

} // namespace richengine::model
