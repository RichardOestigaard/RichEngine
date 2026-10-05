#include "RoPE.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace richengine::ops {

void RoPE::addTables(
    metal::CommandGraph &graph, metal::MetalBuffer targetPositions,
    metal::MetalBuffer draftPositions,
    metal::MetalBuffer targetInverseFrequencies,
    metal::MetalBuffer draftInverseFrequencies,
    metal::MetalBuffer targetCosine, metal::MetalBuffer targetSine,
    metal::MetalBuffer draftCosine, metal::MetalBuffer draftSine,
    RopeTableParams rows, uint32_t maximumRows) {
  // Either side may be empty: a tree batch builds the draft's tables before
  // its forward and the target's after the tree input pass wrote them.
  if ((!rows.target_rows && !rows.draft_rows) ||
      rows.target_rows > maximumRows || rows.draft_rows > maximumRows) {
    throw std::invalid_argument("invalid RoPE table row count");
  }
  const uint64_t elements =
      std::max<uint64_t>(uint64_t{rows.target_rows} * rows.target_dims,
                         uint64_t{rows.draft_rows} * 64);
  graph.add("rope_build_tables",
            {std::move(targetPositions), std::move(draftPositions),
             std::move(targetInverseFrequencies),
             std::move(draftInverseFrequencies), std::move(targetCosine),
             std::move(targetSine), std::move(draftCosine),
             std::move(draftSine)},
            rows, {(elements + 255) / 256, 1, 1}, {256, 1, 1});
}

} // namespace richengine::ops
