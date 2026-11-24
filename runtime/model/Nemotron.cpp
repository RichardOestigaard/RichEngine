// The nemotron_h_moe target's GGUF loader and its layer encoder
// (model/Nemotron.hpp). The HF reference is transformers'
// modeling_nemotron_h.py (NemotronHMamba2Mixer, the NemotronHMoE block);
// llama.cpp's nemotron-h.cpp shows the same layout on the GGUF this
// loader plans.
#include "model/GgufTarget.hpp"
#include "model/Nemotron.hpp"
#include "model/TargetLayerEncoder.hpp"
#include "model/TargetLoader.hpp"
#include "model/TargetModelGemmaHelpers.hpp"
#include "model/TargetWeights.hpp"
#include "ops/Linear.hpp"
#include "ops/Mamba2.hpp"
#include "ops/MoE.hpp"
#include "ops/Normalization.hpp"
#include "ops/PagedAttention.hpp"
#include "ops/RowCopy.hpp"

#include <any>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace richengine::model {
namespace {

// The nemotron_h_moe additions to requireTargetLayout's checks: the two
// layer masks name every block exactly once, and the Mamba-2 geometry
// agrees with the packed in_proj and the conv input it splits into.
void requireNemotronLayout(const TargetLayout &layout) {
  requireTargetLayout(layout, /*affine=*/false);
  if (layout.mambaHeads != 64 || layout.mambaHeadDimension != 64 ||
      layout.mambaGroups != 8 || layout.mambaStateSize != 128 ||
      layout.mambaInnerSize != 4096 || layout.mambaTimeStepRank != 64 ||
      layout.convolutionDimension !=
          layout.mambaInnerSize +
              2 * layout.mambaGroups * layout.mambaStateSize ||
      layout.convolutionTaps != 4 ||
      layout.packedGdnWidth !=
              (layout.mambaInnerSize + layout.convolutionDimension +
               layout.mambaTimeStepRank + 255) /
                  256 * 256 ||
      layout.attentionLayerCount() + layout.moeLayerCount() +
              layout.gdnStateLayout().layers !=
          layout.layers)
    throw WeightStoreError("nemotron target layout is inconsistent");
}

// A Mamba-2 block's mixer sections: the padded fused in_proj, the
// taps-major conv kernel and its bias, the raw ssm_a (the kernels apply
// -exp), the D skip, the dt bias, the grouped gated norm and the out
// projection.
Mamba2MixerWeights readMambaMixer(WeightFile &file,
                                  const BlockTargetFormat &format,
                                  const TargetLayout &layout) {
  Mamba2MixerWeights mamba;
  mamba.inputProjection =
      format.projection(file, layout.packedGdnWidth, layout.hiddenSize,
                        "mamba-input");
  mamba.convolutionWeights =
      file.section(uint64_t{layout.convolutionDimension} *
                       layout.convolutionTaps * sizeof(uint16_t),
                   "mamba-convolution");
  mamba.convolutionBias = file.section(
      uint64_t{layout.convolutionDimension} * sizeof(uint16_t),
      "mamba-convolution-bias");
  mamba.decay = file.section(
      uint64_t{layout.mambaHeads} * sizeof(float), "mamba-decay");
  mamba.skip = file.section(
      uint64_t{layout.mambaHeads} * sizeof(float), "mamba-skip");
  mamba.timeBias = file.section(
      uint64_t{layout.mambaHeads} * sizeof(uint16_t), "mamba-time-bias");
  mamba.mixerNorm = format.norm(
      file, layout.mambaInnerSize, "mamba-norm");
  mamba.mixerNorm.rmsEpsilon = layout.rmsEpsilon;
  mamba.outputProjection =
      format.projection(file, layout.hiddenSize, layout.mambaInnerSize,
                        "mamba-output");
  return mamba;
}

// A plain GQA attention block: the fused QKV and the out projection; no
// per-head norms, no gate rows.
AttentionMixerWeights readNemotronAttention(
    WeightFile &file, const BlockTargetFormat &format,
    const MixerGeometry &geometry) {
  AttentionMixerWeights attention;
  attention.inputProjection =
      format.fused(file, geometry.packedFullWidth, geometry.hiddenSize,
                   "attention-input", {"attn-q", "attn-k", "attn-v"});
  attention.outputProjection = format.projection(
      file, geometry.hiddenSize, geometry.attentionWidth,
      "attention-output");
  return attention;
}

// The ungated relu-squared MoE block: the F32 router, the F32 per-expert
// selection bias, the expert up/down slabs and the ungated shared
// expert's — the gate slab slots stay empty.
ops::MoeWeights readNemotronMoe(WeightFile &file) {
  ops::BlockMoeWeights block;
  block.router = readQuantizedSegment(file, "router");
  block.expertBias = readQuantizedSegment(file, "expert-bias");
  block.up.routed = readQuantizedSegment(file, "experts-up");
  block.down.routed = readQuantizedSegment(file, "experts-down");
  block.up.shared = readQuantizedSegment(file, "shared-expert-up");
  block.down.shared = readQuantizedSegment(file, "shared-expert-down");
  return ops::MoeWeights{std::move(block)};
}

// nemotron's state strides: the mamba cell's conv FIFO and recurrent
// matrix share the GDN cells' lane layout.
ops::Mamba2StateStrides nemotronState(const TargetModelGeometry &geometry) {
  return {geometry.stateLayout.convolutionLayerBytes(),
          geometry.stateLayout.recurrentLayerBytes(),
          geometry.stateLayout.convolutionBytes()};
}

} // namespace

// The nemotron layer encoder, at namespace scope — not the anonymous one
// — because TargetModel's friend declaration names it.
class NemotronLayerEncoder final : public TargetLayerEncoder {
public:
  explicit NemotronLayerEncoder(const TargetModel &model) : model_(model) {}

  // One block of a prefill chunk: its input norm, then its one sub-layer
  // — the Mamba-2 mixer, the GQA attention block or the MoE — added to
  // the residual in `output`.
  metal::MetalBuffer addPrefillLayer(TargetModel::PrefillStep &step,
                                     uint32_t index,
                                     metal::MetalBuffer input,
                                     metal::MetalBuffer output) const override {
    const TargetLayerWeights &layer = model_.weights_->layers[index];
    if (const auto *mamba =
            std::get_if<Mamba2MixerWeights>(&layer.mixer)) {
      addPrefillMamba(step, *mamba, layer.inputNorm, input, output);
    } else if (const auto *attention =
                   std::get_if<AttentionMixerWeights>(&layer.mixer)) {
      const metal::MetalBuffer residual =
          model_.addPrefillMixer(step, *attention, layer.inputNorm, input);
      ops::RowCopy::add(step.graph, residual,
                        {0, model_.geometry_.hiddenSize, 0}, output,
                        {0, model_.geometry_.hiddenSize, 0}, step.rows,
                        model_.geometry_.hiddenSize);
    } else {
      addPrefillMoe(step, layer, input, output);
    }
    return output;
  }

  void addVerifyLayer(TargetModel::VerifyStep &step, uint32_t index,
                      metal::MetalBuffer input,
                      metal::MetalBuffer output) const override {
    const TargetLayerWeights &layer = model_.weights_->layers[index];
    if (const auto *mamba =
            std::get_if<Mamba2MixerWeights>(&layer.mixer)) {
      addVerifyMamba(step, *mamba, layer.inputNorm, input, output);
    } else if (const auto *attention =
                   std::get_if<AttentionMixerWeights>(&layer.mixer)) {
      const metal::MetalBuffer residual =
          model_.addVerifyMixer(step, *attention, layer.inputNorm, input);
      ops::RowCopy::add(step.graph, residual,
                        {0, model_.geometry_.hiddenSize, 0}, output,
                        {0, model_.geometry_.hiddenSize, 0}, step.rows,
                        model_.geometry_.hiddenSize);
    } else {
      addVerifyMoe(step, layer, input, output);
    }
  }

private:
  // A Mamba-2 block: input norm, the fused in_proj into the packed rows,
  // the conv/scan/gated-norm chain over every sequence's state cells, and
  // the out projection's residual write into `output`.
  void addPrefillMamba(TargetModel::PrefillStep &step,
                       const Mamba2MixerWeights &mixer,
                       const ops::NormWeights &norm,
                       metal::MetalBuffer input,
                       metal::MetalBuffer output) const {
    const TargetModelPrefillBuffers &b = step.buffers;
    const uint32_t layer = step.gdnLayer++;
    model_.addPrefillNorm(step, input, norm,
                          mixer.inputProjection.layout());
    model_.operators_.linear().addPrefill(
        step.graph, b.normalized, mixer.inputProjection, b.gdnPacked,
        b.projectionSums, step.rows, b.linearScratch);
    for (const TargetModelPrefillSequence &sequence : step.sequences) {
      const auto u16 = [&](const metal::MetalBuffer &buffer,
                           uint32_t width) {
        return rowsOf<uint16_t>(model_.backend_, buffer, sequence.rowBegin,
                                sequence.rows, width);
      };
      const auto f32 = [&](const metal::MetalBuffer &buffer,
                           uint32_t width) {
        return rowsOf<float>(model_.backend_, buffer, sequence.rowBegin,
                             sequence.rows, width);
      };
      ops::Mamba2::addPrefill(
          step.graph,
          {u16(b.gdnPacked, model_.geometry_.packedGdnWidth),
           mixer.convolutionWeights, mixer.convolutionBias,
           sequence.convolutionIn[layer],
           sequence.convolutionOut[layer],
           u16(b.gdnPostConv, model_.geometry_.convolutionDimension),
           f32(b.gdnDecay, model_.geometry_.mambaHeads), mixer.decay,
           mixer.skip, mixer.timeBias, sequence.recurrentIn[layer],
           sequence.recurrentOut[layer],
           u16(b.recurrent, model_.geometry_.mambaInnerSize),
           mixer.mixerNorm,
           u16(b.gdnHidden, model_.geometry_.mambaInnerSize)},
          model_.geometry_.mambaShape(), sequence.rows,
          prefillSums(model_.backend_, b.projectionSums, sequence.rowBegin,
                      sequence.rows, model_.geometry_.mambaInnerSize));
    }
    model_.addPrefillOutput(step, b.gdnHidden, mixer.outputProjection,
                            input, output);
  }

  // The MoE block: its input norm feeds the router and the ungated
  // experts; the block's output adds to the block input.
  void addPrefillMoe(TargetModel::PrefillStep &step,
                     const TargetLayerWeights &layer,
                     metal::MetalBuffer input,
                     metal::MetalBuffer output) const {
    const TargetModelPrefillBuffers &b = step.buffers;
    const ops::MoeWeights &weights = std::get<ops::MoeWeights>(layer.ffn);
    ops::Normalization::addRms(step.graph, input, layer.inputNorm,
                               b.normalized, model_.geometry_.hiddenSize,
                               step.rows);
    ops::MoE::add(step.graph, {b.normalized, input, output, b.moe, {}, {}},
                  weights, *step.moe);
  }

  // A verify step's Mamba-2 block: the norm, in_proj, conv/scan/norm and
  // residual out_proj replay from a baked indirect command buffer.
  void addVerifyMamba(TargetModel::VerifyStep &step,
                      const Mamba2MixerWeights &mixer,
                      const ops::NormWeights &norm,
                      metal::MetalBuffer input,
                      metal::MetalBuffer output) const {
    const TargetModelVerifyBuffers &b = step.buffers;
    const ops::Linear &linear = model_.operators_.linear();
    const uint32_t layer = step.gdnLayer++;
    const ops::LinearPlan inputPlan = linear.decodePlan(
        mixer.inputProjection, step.planLanes, ops::LinearEpilogue::None,
        nullptr, step.rowCapacity);
    const ops::LinearPlan outputPlan = linear.decodePlan(
        mixer.outputProjection, step.planLanes,
        ops::LinearEpilogue::Residual, nullptr, step.rowCapacity);
    step.graph.beginBakedSpan();
    const ops::PreparedInput normalized = ops::Normalization::addRms(
        step.graph, input, norm, b.normalized,
        model_.geometry_.hiddenSize, step.rows, b.linearScratch,
        inputPlan.input(), false);
    linear.add(step.graph,
               {.input = b.normalized, .output = b.gdnPacked[layer],
                .scratch = b.linearScratch, .prepared = normalized},
               mixer.inputProjection, inputPlan);
    if (step.tree)
      throw std::invalid_argument("mamba2 verify does not support trees");
    const ops::Mamba2DecodeBuffers mamba{
        b.gdnPacked[layer], mixer.convolutionWeights, mixer.convolutionBias,
        b.currentGdnStates,
        b.nextGdnStates, b.gdnMixed[layer], mixer.decay, mixer.timeBias,
        b.gdnDecay[layer], mixer.skip, mixer.mixerNorm, b.gdnHidden,
        b.linearScratch, b.gdnTape};
    // The tape lane record carries byte offsets only — the GDN record and
    // the mamba one are layout-identical, the layer stride aside, and the
    // stride the runtime computes is already the mamba shape's.
    ops::Mamba2TapeLanes mambaTape;
    const ops::Mamba2TapeLanes *tape = nullptr;
    if (step.tapeLanes) {
      mambaTape.pending = step.tapeLanes->pending;
      mambaTape.step = step.tapeLanes->step;
      mambaTape.rows = step.tapeLanes->rows;
      mambaTape.layerBytes = step.tapeLanes->layerBytes;
      tape = &mambaTape;
    }
    const ops::PreparedInput hidden = ops::Mamba2::addDecode(
        step.graph, mamba, model_.geometry_.mambaShape(), step.lanes, layer,
        nemotronState(model_.geometry_), outputPlan.input(), step.liveRows,
        step.rowCapacity, tape);
    linear.add(step.graph,
               {.input = b.gdnHidden, .output = output, .residual = input,
                .scratch = b.linearScratch, .prepared = hidden},
               mixer.outputProjection, outputPlan);
    step.graph.endBakedSpan();
  }

  void addVerifyMoe(TargetModel::VerifyStep &step,
                    const TargetLayerWeights &layer,
                    metal::MetalBuffer input,
                    metal::MetalBuffer output) const {
    const TargetModelVerifyBuffers &b = step.buffers;
    const ops::MoeWeights &weights = std::get<ops::MoeWeights>(layer.ffn);
    step.graph.beginBakedSpan();
    ops::Normalization::addRms(step.graph, input, layer.inputNorm,
                               b.normalized, model_.geometry_.hiddenSize,
                               step.rows);
    ops::MoE::add(step.graph,
                  {b.normalized, input, output, b.moe, {}, b.verifyRows},
                  weights, *step.moe);
    step.graph.endBakedSpan();
  }

  const TargetModel &model_;
};

TargetModelWeights loadNemotronWeights(metal::MetalBackend &backend,
                                       const TargetLayout &layout,
                                       const TargetFiles &files) {
  requireNemotronLayout(layout);
  // The arch's only source: a nemotron_h_moe GGUF's images. No packed
  // schema or safetensors map covers the tensor set.
  const auto *gguf =
      std::get_if<std::reference_wrapper<GgufTargetLoader>>(&files);
  if (!gguf)
    throw WeightStoreError("nemotron loads from a GGUF only");
  const BlockTargetFormat format;
  const uint64_t allocationBaseline = backend.memoryStats().allocatedBytes;
  TargetModelWeights result;
  result.layout = layout;
  result.layers.reserve(layout.layers);
  for (uint32_t index = 0; index < layout.layers; ++index) {
    WeightFile file = gguf->get().layer(index);
    auto &layer = result.layers.emplace_back();
    layer.inputNorm =
        format.norm(file, layout.hiddenSize, "input-norm");
    layer.inputNorm.rmsEpsilon = layout.rmsEpsilon;
    // The variants' first alternatives (GdnMixerWeights, DenseFfnWeights)
    // are what a default member holds — an unused slot must hold its Empty
    // record or zero projections register in the runtime geometry.
    if (layout.isMoeLayer(index)) {
      layer.mixer = EmptyMixerWeights{};
      layer.ffn = readNemotronMoe(file);
    } else {
      layer.ffn = EmptyFfnWeights{};
      if (layout.isFullAttentionLayer(index))
        layer.mixer =
            readNemotronAttention(file, format, layout.mixerGeometry());
      else
        layer.mixer = readMambaMixer(file, format, layout);
    }
    file.finish();
    result.files.push_back(file.record());
  }
  {
    // The batch commit replays every mamba layer in one dispatch, so its
    // ssm_a operand is the layers' decays stacked in state order.
    const uint32_t cells = layout.gdnStateLayout().layers;
    result.mambaDecays = backend.allocateBuffer(
        uint64_t{cells} * layout.mambaHeads * sizeof(float),
        metal::BufferStorage::Shared, "mamba-decays");
    float *decays = static_cast<float *>(result.mambaDecays.contents());
    uint32_t cell = 0;
    for (const TargetLayerWeights &layer : result.layers)
      if (const auto *mamba =
              std::get_if<Mamba2MixerWeights>(&layer.mixer)) {
        std::memcpy(decays + uint64_t{cell} * layout.mambaHeads,
                    mamba->decay.contents(),
                    layout.mambaHeads * sizeof(float));
        ++cell;
      }
    if (cell != cells)
      throw WeightStoreError("nemotron mamba layer count mismatch");
  }
  {
    WeightFile file = gguf->get().head();
    result.finalNorm =
        format.norm(file, layout.hiddenSize, "final-norm");
    result.finalNorm.rmsEpsilon = layout.rmsEpsilon;
    result.logitsProjection =
        format.projection(file,
                          paddedVocabularySize(layout.vocabularySize),
                          layout.hiddenSize, "logits");
    // bf16 logits would round near-ties together: their spacing is 0.125
    // at logits of 16 to 32.
    result.logitsProjection.destination = ops::FloatOutput::Float32;
    file.finish();
    result.files.push_back(file.record());
  }
  {
    WeightFile file = gguf->get().embedding();
    result.tokenEmbedding =
        format.embedding(file, layout.vocabularySize, layout.hiddenSize);
    file.finish();
    result.files.push_back(file.record());
  }
  result.manifestFingerprintSha256 =
      weightManifestFingerprint(result.files);
  result.actualAllocatedBytes = metal::allocationDelta(
      allocationBaseline, backend.memoryStats().allocatedBytes);
  return result;
}

std::unique_ptr<TargetLayerEncoder>
makeNemotronLayerEncoder(const TargetModel &model) {
  return std::make_unique<NemotronLayerEncoder>(model);
}

} // namespace richengine::model
