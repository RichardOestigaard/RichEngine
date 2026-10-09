#include "model/RuntimeImpl.hpp"

namespace richengine::model {

  void Runtime::Impl::addRopeTables(CommandGraph &graph, MetalBuffer targetPositions,
                     uint32_t targetRows, MetalBuffer draftPositions,
                     uint32_t draftRows, MetalBuffer targetCos,
                     MetalBuffer targetSin, MetalBuffer draftCos,
                     MetalBuffer draftSin, MetalBuffer targetCosAlt,
                     MetalBuffer targetSinAlt) const {
    // A Null draft has no rotary pairs: its IF slot is unallocated, so the
    // draft side borrows the target's (it stays unwritten at 0 draft rows).
    const MetalBuffer draftIf =
        prefillArena->get(geometry.draftRotaryPairs()
                              ? PrefillTensor::DraftInverseFrequencies
                              : PrefillTensor::TargetInverseFrequencies);
    // The dual-geometry target's second target table: the same positions at
    // the alternate pairs and theta; its draft side stays unwritten.
    if (targetCosAlt) {
      ops::RoPE::addTables(
          graph, targetPositions, draftPositions,
          prefillArena->get(PrefillTensor::TargetInverseFrequencies2),
          draftIf,
          std::move(targetCosAlt), std::move(targetSinAlt), draftCos,
          draftSin,
          {targetRows, 0, geometry.target.altRotaryPairs,
           geometry.target.ropeAxes},
          kPrefillRows);
    }
    ops::RoPE::addTables(
        graph, std::move(targetPositions), std::move(draftPositions),
        prefillArena->get(PrefillTensor::TargetInverseFrequencies),
        draftIf,
        std::move(targetCos), std::move(targetSin), std::move(draftCos),
        std::move(draftSin),
        {targetRows, draftRows, geometry.target.rotaryPairs,
         geometry.target.ropeAxes},
        kPrefillRows);
  }

  void Runtime::Impl::encodeBatchEmbedding(CommandGraph &graph, DecodeTensor tokens,
                            DecodeTensor output, uint32_t lanes,
                            uint32_t rowFactor) {
    if (!lanes || lanes * rowFactor > kLaneCount)
      throw std::invalid_argument("invalid embedding batch width");
    const uint32_t storage = lanes * rowFactor;
    const uint32_t rows = lanes * rowFactor * kDecodeRows;
    targetModel.addEmbedding(graph, decodeArena->packed(tokens, storage),
                             decodeArena->packed(output, storage), rows);
  }

} // namespace richengine::model
