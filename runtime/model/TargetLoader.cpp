#include "model/TargetLoader.hpp"
#include "Checked.hpp"

#include <utility>

namespace richengine::model {

ops::Projection BlockTargetFormat::fused(WeightFile &file, uint32_t outputSize, uint32_t inputSize,
                                         std::string_view,
                                         std::initializer_list<std::string_view> tensors) const {
  ops::BlockWeights weights;
  uint32_t offset = 0;
  for (std::string_view tensor : tensors) {
    ops::QuantizedSegment s = readQuantizedSegment(file, tensor);
    s.columnOffset = offset;
    offset += s.outputSize;
    weights.segments.push_back(std::move(s));
  }
  // Trellis beta and alpha are separate coded tensors but one logical tail
  // segment of the fused GDN projection. Ordinary GGUFs repack them into one
  // padded tensor instead, so only the trellis form consumes a second section.
  if (tensors.size() == 3 && *(tensors.end() - 1) == "gdn-ab" &&
      !weights.segments.empty() && weights.segments.back().isTrellis()) {
    ops::QuantizedSegment alpha = readQuantizedSegment(file, "gdn-alpha");
    if (!alpha.isTrellis() || alpha.outputSize > outputSize - offset)
      throw WeightStoreError("GGUF trellis alpha/beta segments do not fit the projection");
    alpha.columnOffset = offset;
    offset += alpha.outputSize;
    weights.segments.push_back(std::move(alpha));
  }
  return {outputSize, inputSize, std::move(weights)};
}

template <class Format>
MixerWeights readMixer(WeightFile &file, const Format &format,
                               const MixerGeometry &geometry, bool fullAttention) {
  constexpr uint64_t kFloat32Bytes = 4;
  if (fullAttention) {
    AttentionMixerWeights attention;
    attention.inputProjection =
        format.fused(file, geometry.packedFullWidth, geometry.hiddenSize, "attention-input",
                     {"attn-q", "attn-k", "attn-v"});
    attention.queryNorm = format.norm(file, geometry.attentionHeadDimension, "query-norm");
    attention.keyNorm = format.norm(file, geometry.attentionHeadDimension, "key-norm");
    attention.outputProjection =
        format.projection(file, geometry.hiddenSize, geometry.attentionWidth, "attention-output");
    return attention;
  }
  GdnMixerWeights gdn;
  gdn.inputProjection = format.fused(file, geometry.packedGdnWidth, geometry.hiddenSize,
                                     "gdn-input", {"gdn-qkv", "gdn-z", "gdn-ab"});
  gdn.convolutionWeights = file.section(
      checkedMultiply<WeightStoreError>(
          checkedMultiply<WeightStoreError>(geometry.convolutionDimension,
                                            kGdnConvolutionTaps,
                                            "convolution elements"),
          kBFloat16Bytes, "convolution bytes"),
      "gdn-convolution");
  gdn.decay = file.section(
      checkedMultiply<WeightStoreError>(geometry.gdnValueHeads, kFloat32Bytes,
                                        "GDN decay bytes"),
      "gdn-decay");
  gdn.timeBias = file.section(
      checkedMultiply<WeightStoreError>(geometry.gdnValueHeads, kBFloat16Bytes,
                                        "GDN time bias bytes"),
      "gdn-time-bias");
  gdn.mixerNorm = format.norm(file, geometry.gdnHeadDimension, "gdn-norm");
  gdn.outputProjection =
      format.projection(file, geometry.hiddenSize, geometry.attentionWidth, "gdn-output");
  gdn.outputHeadOrder = Format::gdnOutputOrder;
  return gdn;
}

template MixerWeights readMixer(WeightFile &, const AffineTargetFormat &,
                                        const MixerGeometry &, bool);
template MixerWeights readMixer(WeightFile &, const BlockTargetFormat &,
                                        const MixerGeometry &, bool);
template MixerWeights readMixer(WeightFile &, const OqTargetFormat &,
                                        const MixerGeometry &, bool);

} // namespace richengine::model
