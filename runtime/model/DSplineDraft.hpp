#pragma once

#include "DFlashDraft.hpp"
#include "DFlashV1Draft.hpp"
#include "Model.hpp"
#include "WeightImages.hpp"
#include "WeightStore.hpp"
#include "ops/DraftAttention.hpp"
#include "ops/DraftSelector.hpp"
#include "ops/ExecutionPlans.hpp"
#include "ops/Linear.hpp"
#include "ops/Normalization.hpp"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <span>
#include <string_view>
#include <variant>
#include <vector>

namespace richengine::model {

// A DSpline draft (Tencent's DSpine, architectures: ["DFlashDraftModel"]):
// the plain transformer draft's layer stack — fused QKV, per-head Q/K
// norms, output projection, gated MLP, every layer sliding-attention and
// causal inside the block — plus a selected-prefix code bridge. After each
// layer the bridge writes each block position's predicted feature into its
// successor: the layer output's RMS-normalized row reads a code through a
// per-layer read_weight (the block's anchor row takes the anchor token's
// codebook row instead), the code projects through the packed
// [gate; predecessor_gate; prev] weights, a sigmoid gate mixes the current
// row's query with its predecessor's sender and value, and the result
// writes back through feature_write_weights scaled by the row's RMS — the
// causal conditioning chain the backbone layers then refine in depth.
//
// The bridge's elementwise phases (per-row RMS with the tiny-floor clamp,
// L2 code normalization with anchor substitution, the packed gate/prev
// projection, the predecessor-shifted sigmoid message and the RMS-scaled
// write-back) run as the dspline_bridge_* kernels
// (runtime/metal/kernels/decode/dspline_bridge.metal), dispatched from
// DSplineDraft::addDecode at every layer of layout.bridgeWriteLayers. The
// fused next-layer norm the reference's finish emits stays an ordinary
// addRms dispatch.
//
// The serial last-write refinement (code_bridge_second_pass, sglang_dspine
// RefinementTail.greedy): upstream never proposes the plain argmax chain;
// it rescores every mask row's top-16 head candidates under the ACTUAL
// predecessor token's code, fixing the last write's sender. The pieces,
// mapped onto this tree's buffers:
//   * stash: the last write's `packed` field (bf16 [rows][3*rank] at
//     `intermediate` offset tileRows*2*code elements) and `rms` (fp32
//     after the delta field) survive through the selection window, and
//     `raw * final_norm` is staged into the dead `normalized` rows by the
//     dspline_refine_stash dispatch between the last write's delta matmul
//     and its finish.
//   * cand: kDraftSelectTop16Sharded emits top-16 ids/values into the
//     selector's partialIds/partialValues.
//   * sid[i][k] = anchor for the first mask row, cand[i-1][k] after.
//   * score[i][k_prev][k_self] = base[i][k_self] + rms[i] * dot(
//       sigmoid(q[i] + tsv[sid][0:R] + gate.bias) * tsv[sid][R:2R],
//       tg[cand[i][k_self]])   with q = packed[1:, :rank]
//     idx[i][k] = argmax_k_self score; the proposal chain walks
//     j = idx[0][0], token[i] = cand[i][j], j = idx[i+1][j] serially and
//     lands in ProposedTokens/proposalProbabilities/confidence under the
//     draft_select_plain contract (dspline_refine_rows +
//     dspline_refine_walk in kernels/decode/dspline_bridge.metal).
//   * base[i][k] = sum_h raw[i][h]*norm.weight[h]*head[cand_k][h] — the
//     unnormalized logit: a second vocabularyProjection pass over the
//     staged raw*norm.weight rows, into the logits buffer the top-16 pass
//     has already consumed.
//   * tables, built once at load (upstream build_serial_tables):
//     TSV [vocab][2*rank] = (sqrt(code)*codebook) @ [pred_gate;prev].T —
//     chunked through dspline_bridge_codes + dspline_bridge_matmul; TG
//     [vocab][rank] = (head_rows*norm.weight) @ feature_write_weights[last]
//     — the quantized vocabulary head unpacked by dspline_table_dequant.
// Sampled lanes take the same rescored table but renormalized
// (upstream serial_rs_table): each predecessor lane's score divides by
// the RMS its refined row would take, and the walk draws under the
// lane's joint min-p/top-k/top-p filter at its temperature on the
// decode's proposal uniforms, chained on the sampled predecessor
// (dspline_refine_norm + the sampled branch of dspline_refine_walk).
// The two extra load-time tables that normalization needs are the
// write Gram and the norm-divided write projection below. A comb-tree
// batch, a refinement width other than the selector's candidate count,
// a bridge rank past the norm kernel's staged bound, or tables that
// could not build still take the plain walk.
struct DSplineDraftWeights final {
  // The plain-draft state the forward runs on: the parsed layout (kind
  // DSpline), the layers' projections and norms, the fc context fusion
  // and the final norm. Mirrored into `layout` for the package checks.
  DFlashV1DraftWeights plain;
  // The selected-prefix bridge tensors, bf16 as stored:
  // [vocabularySize][bridgeCodeSize] anchor-token codes.
  metal::MetalBuffer codebook;
  // [layers - 1][bridgeCodeSize][hiddenSize]: layers 0..L-2's read weights.
  metal::MetalBuffer earlyReadWeights;
  // [bridgeCodeSize][hiddenSize]: the last layer's read weight.
  metal::MetalBuffer readWeight;
  // [layers][hiddenSize][bridgeRank]: each layer's feature write.
  metal::MetalBuffer featureWriteWeights;
  // The code projections, [bridgeRank][bridgeCodeSize] each — the write
  // consumes them packed as [gate; predecessor_gate; prev]; a packed copy
  // can be staged when the bridge kernels land, or the rows sliced the way
  // contextKvRows slices the fused QKV.
  metal::MetalBuffer gateWeight;
  metal::MetalBuffer predecessorGateWeight;
  metal::MetalBuffer prevWeight;
  // [bridgeRank]: the gate projection's bias.
  metal::MetalBuffer gateBias;
  // The serial refinement's load-time tables (build_serial_tables), both
  // bf16 row-major: [vocabularySize][2*bridgeRank] sender gate/value pairs
  // TSV[v] = [predecessor_gate; prev] @ bf16(sqrt(code)*codebook[v]), and
  // [vocabularySize][bridgeRank] candidate targets TG[v] =
  // bf16(head[v] * finalNorm) @ featureWriteWeights[last]. Empty when the
  // vocabulary head's layout cannot feed the build or the checkpoint's
  // head_topk is not the selector's candidate count — selection then
  // stays on the plain walk.
  metal::MetalBuffer serialTsv;
  metal::MetalBuffer serialTg;
  // The sampled lanes' normalization tables (upstream serial_rs_table):
  // serialGram [bridgeRank][bridgeRank] = writeT @ writeT — the Gram
  // write_w^T write_w scoring a gated sender's delta magnitude
  // (m^T S m = ||m @ write_w^T||^2) — and serialRW
  // [bridgeRank][hiddenSize] = writeT / finalNorm, so the staged
  // raw*final_norm rows matmul into raw @ write_w, the raw x delta
  // cross term. Both bf16; built alongside the serial tables and empty
  // with them — a sampled lane then stays on the plain walk.
  metal::MetalBuffer serialGram;
  metal::MetalBuffer serialRW;
  // The layout the descriptor resolved — the same object plain.layout
  // holds; a copy so the variant's `weights.layout` reads uniformly.
  DFlashDraftLayout layout;
  std::vector<WeightFileRecord> files;
  uint64_t actualAllocatedBytes = 0;
};

inline constexpr std::string_view kDSplineDraftMagic = "MDFS0007";

// An RichEngine package's DSpline-draft files: layer-<N>.bin (the plain
// draft's section order) and model.bin (the plain model's sections, then
// the bridge tensors), as DSplineDraft.cpp reads them.
struct PackedDSplineDraftFiles final {
  WeightImages &images;
  std::filesystem::path directory;
  const DFlashDraftLayout &layout;
  [[nodiscard]] WeightFile layer(uint32_t index) const;
  [[nodiscard]] WeightFile model() const;
};

class DSplineCheckpointLoader;

using DSplineDraftFiles =
    std::variant<PackedDSplineDraftFiles,
                 std::reference_wrapper<DSplineCheckpointLoader>>;

// vocabularyHead, when non-null and in the affine layout, lets the loader
// build the serial refinement's TSV/TG tables (the target's packed
// vocabulary head, shared with the draft). Null leaves them empty and
// selection falls back to the plain walk.
[[nodiscard]] DSplineDraftWeights
loadDSplineDraftWeights(metal::MetalBackend &backend,
                        const DSplineDraftFiles &files,
                        DFlashDraftLayout layout,
                        const ops::Projection *vocabularyHead = nullptr);

// The DSpline draft as the runtime dispatches it: the four entry points
// mirror DFlashDraft's through the plain draft it composes. The forward's
// bridge seam is DSplineDraft::addDecode.
class DSplineDraft final {
public:
  DSplineDraft(const DSplineDraftWeights &weights,
               metal::MetalBackend &backend,
               const ops::ExecutionPlans &operators);

  void addContextPrefill(metal::CommandGraph &graph,
                         DFlashPrefillBuffers buffers, uint32_t rows,
                         std::span<const DFlashPrefillSpan> spans) const;
  // `devicePack` binds the committed step's DecodeStepPackLane block: the
  // attention splits take each lane's cache length from it, so the
  // GPU-chained predraft encodes before the host knows them. cacheLengths
  // still sizes the batch.
  void addDecode(metal::CommandGraph &graph, DFlashDecodeBuffers buffers,
                 const ops::Projection &vocabularyProjection,
                 std::span<const uint32_t> cacheLengths,
                 const metal::MetalBuffer &devicePack = {}) const;
  // The serial last-write refinement (code_bridge_second_pass) when its
  // tables built: the plain top-16 over the head logits, the gather or
  // second head pass over the stashed raw*final_norm rows for the
  // unnormalized base logits, dspline_refine_rows rescore and the
  // dspline_refine_walk chain under the draft_select_plain contract.
  // Sampled lanes take the same rescore renormalized —
  // dspline_bridge_matmul's raw-projection pass plus
  // dspline_refine_norm's refined-RMS denominators — and the walk's
  // sampled branch draws under each lane's policy on its proposal
  // uniforms, chained on the sampled predecessor (upstream
  // serial_rs_table + serial_sample_chain). A tree mask, missing
  // tables or a bridge rank past the norm kernel's bound takes the
  // plain walk.
  void addSelection(metal::CommandGraph &graph,
                    const ops::DraftSelectorBuffers &buffers,
                    std::span<const uint32_t> anchors,
                    std::span<const ops::SamplingPolicy> policies,
                    uint32_t treeMask) const;
  void addContextCommit(metal::CommandGraph &graph,
                        DFlashContextBuffers buffers,
                        std::span<const uint32_t> startPositions) const;
  const DSplineDraftWeights &weights() const { return weights_; }

private:
  // The trained write after `layer`'s output (buffers.hidden[layer & 1 ^ 1],
  // updated in place): the five phases of the seam comment in addDecode —
  // per-row RMS, the layer's read projection to sqrt(code)-normalized
  // codes with the anchor codebook substitution, the packed
  // [gate; predecessor_gate; prev] projection, the predecessor-shifted
  // sigmoid message and the RMS-scaled write-back. The code path's scratch
  // reuses the decode pass's idle intermediate and projected rows.
  void addBridgeWrite(metal::CommandGraph &graph,
                      const DFlashDecodeBuffers &buffers, uint32_t layer,
                      uint32_t rows) const;

  // The refinement's window into the decode pass's buffers, which
  // addSelection never sees: the intermediate scratch (the last write's
  // packed q columns and rms survive there), the staged raw*final_norm
  // rows the second head pass reads, the linear scratch it stages with,
  // and the vocabulary head itself. addDecode fills it once a decode
  // runs; selection falls back to the plain walk without it.
  struct RefineStash final {
    metal::MetalBuffer intermediate;
    metal::MetalBuffer rawnorm;
    ops::LinearScratch scratch;
    const ops::Projection *head = nullptr;
    uint32_t rows = 0;
    bool valid = false;
  };

  const DSplineDraftWeights &weights_;
  metal::MetalBackend &backend_;
  const ops::ExecutionPlans &operators_;
  DFlashV1Draft plain_;
  mutable RefineStash stash_;
};

} // namespace richengine::model
