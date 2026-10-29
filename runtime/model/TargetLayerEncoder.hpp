#pragma once

// The layer-encode seam: TargetModel's per-layer dispatch is delegated to
// the target architecture's registered encoder (ModelRegistry.hpp's
// TargetArchRegistration::makeEncoder; a null factory selects the standard
// encoder). A new architecture implements the four hooks in its own
// translation unit instead of branching the shared layer loops. The step
// records the hooks walk live in the internal TargetModelGemmaHelpers.hpp.
#include "model/TargetModel.hpp"

#include <memory>
#include <span>

namespace richengine::model {

// The pass inputs addPrefill routes to the active layer encoder:
// PrefillStep::extension holds one until the encoder's beginPrefill
// replaces it with the encoder's own per-pass state record.
struct TargetPrefillInputs {
  // The caller's per-layer scalar overrides — the DiffusionGemma encoder
  // pass runs the trunk with the separately packed encoder scalars while
  // its canvas pass keeps the files' (decoder) values. Empty applies each
  // layer's packed layer_scalar.
  std::span<const float> layerScalars;
};

class TargetLayerEncoder {
public:
  virtual ~TargetLayerEncoder() = default;

  // One layer of the pass over rows [input -> output]; the prefill hook
  // returns the buffer the layer's rows land in (`output` today).
  virtual metal::MetalBuffer addPrefillLayer(TargetModel::PrefillStep &step,
                                           uint32_t index,
                                           metal::MetalBuffer input,
                                           metal::MetalBuffer output) const = 0;
  virtual void addVerifyLayer(TargetModel::VerifyStep &step, uint32_t index,
                              metal::MetalBuffer input,
                              metal::MetalBuffer output) const = 0;
  // The encoder's per-pass state, built once before the layer loop.
  // step.extension holds the routed pass inputs (prefill:
  // TargetPrefillInputs) on entry and the encoder's state record after.
  virtual void beginPrefill(TargetModel::PrefillStep &) const {}
  virtual void beginVerify(TargetModel::VerifyStep &) const {}
  // Whether the encoder writes the capture layers' rows itself (qwen4exp's
  // stream-mean capture): the shared loops' flat row copies of the layer
  // output are skipped then. False keeps the shared RowCopy capture.
  [[nodiscard]] virtual bool capturesHidden() const { return false; }
  // A chain-verify step's input hook: TargetModel::addVerifyInput calls it
  // after the token gather with the verify rows' tokens and the
  // pleNgram - 1 tokens before each lane's first row (`plePriorTokens`,
  // empty for targets without the qwen4exp PLE hash, which the GPU
  // computes from them).
  virtual void onVerifyInput(metal::CommandGraph &,
                             metal::MetalBuffer /*verifyInput*/,
                             metal::MetalBuffer /*plePriorTokens*/,
                             uint32_t /*lanes*/,
                             uint32_t /*verifyStride*/) const {}
  // The auxiliary-state commit at the end of TargetModel::addStateCommit:
  // the shared commit consumed the GDN cells; an arch whose state cells
  // reserve auxiliaryBytes (qwen4exp's PLE history and QSA pending keys)
  // commits them here from the same lane cells and retained counts.
  virtual void commitState(metal::CommandGraph &,
                           const TargetModelCommitBuffers & /*buffers*/,
                           uint32_t /*lanes*/,
                           uint32_t /*verifyStride*/) const {}
};

// The Gemma 4 target's layer encoder (TargetModelGemma.cpp), shared by the
// gemma4 and diffusion_gemma archs — the diffusion trunk's canvas and
// encoder-scalar state is its step extension.
[[nodiscard]] std::unique_ptr<TargetLayerEncoder>
makeGemmaLayerEncoder(const TargetModel &model);

// The qwen4exp target's layer encoder (Qwen4Exp.cpp): the hyper-connected
// residual streams, the PLE layer and the QSA sparse attention. The pieces
// whose kernels richengine does not have yet throw named errors at encode
// time rather than silently skipping.
[[nodiscard]] std::unique_ptr<TargetLayerEncoder>
makeQwen4LayerEncoder(const TargetModel &model);

// The lfm3 target's layer encoder (TargetModelLfm3.cpp): the tri-hybrid
// conv/sliding/global schedule, sandwich norms and the head-wise attention
// gate.
[[nodiscard]] std::unique_ptr<TargetLayerEncoder>
makeLfm3LayerEncoder(const TargetModel &model);

} // namespace richengine::model
