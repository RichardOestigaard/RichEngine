#pragma once

// Manifold-constrained hyper-connections (kernels/shared/mhc.metal), the
// maxwell_moe residual: `streams` parallel fp32 streams of `hidden`
// values per row. Each decoder layer runs two modules — attention's and
// the MoE block's — and the model ends in a head that contracts the
// streams to one row.
//
// The projections these ops consume run on the CALLER through
// ops::Linear: a module's mapping projection (fp32 rows * width ->
// mixWidth logits) and the head's (rows * width -> streams), both from
// fp32 GGUF weight segments into fp32 rows. The ops below apply the
// gates those logits make — none of them runs a projection itself.

#include "metal/CommandGraph.hpp"

#include <cstdint>

namespace richengine::ops {

// `streams` streams of `hidden` values per row, and a module's Sinkhorn
// iteration count.
struct MhcShape final {
  uint32_t hidden = 0;
  uint32_t streams = 0;
  uint32_t sinkhornIterations = 0;
  [[nodiscard]] constexpr uint32_t width() const noexcept {
    return hidden * streams;
  }
  // A row's mapping width: h_pre (streams) | h_post (streams) | h_res
  // (streams^2, row-major).
  [[nodiscard]] constexpr uint32_t mixWidth() const noexcept {
    return streams * streams + 2 * streams;
  }
};

class Mhc final {
public:
  // The fp32 projections these ops' logits and head projection rows come
  // from (mhc_project): the mHC mapping's [mixWidth][streamWidth] and the
  // HC head's [streams][streamWidth] F32 weights, whose fp32 input the
  // bf16-input GGUF float kernels cannot read. `weights` is the fp32
  // [columns][shape.width()] plane, `logits` the fp32 [rows][columns]
  // scratch. One threadgroup per (column, row).
  static void addProject(metal::CommandGraph &graph,
                         metal::MetalBuffer streams,
                         metal::MetalBuffer weights, uint32_t columns,
                         metal::MetalBuffer logits, MhcShape shape,
                         uint32_t rows);
  // A module's whole stage in one dispatch (mhc_module): the streams'
  // projection logits, gated mappings (Sinkhorn h_res included) and the
  // h_pre aggregate into the bf16 `aggregated` rows the consumer's layer
  // norm reads — the three dispatches addProject + addMappings +
  // addWeightedSum otherwise pay per module. `weights` is the mapping's
  // fp32 [mixWidth][shape.width()] plane, `mappings` the fp32
  // [rows][mixWidth] buffer expand-mix reads, `aggregated` bf16
  // [rows][hidden].
  static void addModule(metal::CommandGraph &graph,
                      metal::MetalBuffer streams, metal::MetalBuffer weights,
                      metal::MetalBuffer alphas, metal::MetalBuffer bias,
                      metal::MetalBuffer mappings,
                      metal::MetalBuffer aggregated, MhcShape shape,
                      uint32_t rows);
  // fp32 bytes of `rows` rows of mapping values — the caller-allocated
  // mappings [rows][mixWidth] buffer's size, and equally the logits
  // buffer's and (generously) the headWeights buffer's.
  [[nodiscard]] static constexpr uint64_t mappingBytes(MhcShape shape,
                                                       uint64_t rows) noexcept {
    return rows * shape.mixWidth() * sizeof(float);
  }
  // mappings[r] = the gates of logits[r] scaled by the streams' rms
  // inverse: h_pre = sigmoid + eps, h_post = 2 sigmoid, h_res the
  // residual logits' row-softmaxed matrix after
  // shape.sinkhornIterations Sinkhorn normalizations. `alphas` is the
  // packed [pre, post, res] triple, `bias` [mixWidth] fp32.
  static void addMappings(metal::CommandGraph &graph,
                          metal::MetalBuffer streams,
                          metal::MetalBuffer logits,
                          metal::MetalBuffer alphas,
                          metal::MetalBuffer bias,
                          metal::MetalBuffer mappings, MhcShape shape,
                          uint32_t rows);
  // out bf16 [rows][hidden] = sum_s weights[r][s] * streams[r][s][d].
  // `weights` rows are weightsStride fp32 values apart and the slice
  // starts at weightOffset: pass (mixWidth(), 0) for the aggregate over
  // a mappings row's h_pre, (streams, 0) for a standalone [rows][streams]
  // buffer such as the head weights.
  static void addWeightedSum(metal::CommandGraph &graph,
                             metal::MetalBuffer streams,
                             metal::MetalBuffer weights,
                             uint32_t weightsStride, uint32_t weightOffset,
                             metal::MetalBuffer out, MhcShape shape,
                             uint32_t rows);
  // streams[r][s] = sum_t h_res[t][s] * streams[r][t] + h_post[s] *
  // branch[r], in place; branch is the block's bf16 output and mappings
  // the buffer addMappings wrote.
  static void addExpandMix(metal::CommandGraph &graph,
                           metal::MetalBuffer streams,
                           metal::MetalBuffer branch,
                           metal::MetalBuffer mappings, MhcShape shape,
                           uint32_t rows);
  // headWeights fp32 [rows][streams] = sigmoid(proj * r * scale + base)
  // + eps, r the streams' rms inverse; `proj` is the head projection's
  // fp32 rows, `base` [streams] and `scale` [1] fp32. Feed the result to
  // addWeightedSum for the contraction itself.
  static void addHeadWeights(metal::CommandGraph &graph,
                             metal::MetalBuffer streams,
                             metal::MetalBuffer proj,
                             metal::MetalBuffer base,
                             metal::MetalBuffer scale,
                             metal::MetalBuffer headWeights, MhcShape shape,
                             uint32_t rows);
};

} // namespace richengine::ops
