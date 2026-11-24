#include "model/RuntimeImpl.hpp"

namespace richengine::model {

  void Runtime::Impl::addRopeTables(CommandGraph &graph, MetalBuffer targetPositions,
                     uint32_t targetRows, MetalBuffer draftPositions,
                     uint32_t draftRows, MetalBuffer targetCos,
                     MetalBuffer targetSin, MetalBuffer draftCos,
                     MetalBuffer draftSin, MetalBuffer targetCosAlt,
                     MetalBuffer targetSinAlt,
                     uint32_t maximumRows) const {
    // A Null draft has no rotary pairs: its IF slot is unallocated, so the
    // draft side borrows the target's (it stays unwritten at 0 draft rows).
    const MetalBuffer draftIf =
        prefillArena->get(geometry.draftRotaryPairs()
                              ? PrefillTensor::DraftInverseFrequencies
                              : PrefillTensor::TargetInverseFrequencies);
    // The dual-geometry target's second target table: the same positions at
    // the alternate pairs and theta; its draft side stays unwritten. A
    // draft-only call (the tree path builds the target's tables later)
    // skips it — a 0/0 request is meaningless.
    if (targetCosAlt && targetRows) {
      ops::RoPE::addTables(
          graph, targetPositions, draftPositions,
          prefillArena->get(PrefillTensor::TargetInverseFrequencies2),
          draftIf,
          std::move(targetCosAlt), std::move(targetSinAlt), draftCos,
          draftSin,
          {targetRows, 0, geometry.target.altRotaryPairs,
           geometry.target.ropeAxes, geometry.draftRotaryPairs()},
          maximumRows);
    }
    // A NoPE target (nemotron) leaves its rope tables unallocated: with a
    // rope-less draft there is nothing to build, and with a roped draft
    // the target's zero-dims side borrows the draft's slots — the kernel
    // writes nothing for it.
    if (!geometry.target.rotaryPairs) {
      if (!geometry.draftRotaryPairs())
        return;
      ops::RoPE::addTables(
          graph, std::move(targetPositions), std::move(draftPositions),
          draftIf, draftIf, draftCos, draftSin, std::move(draftCos),
          std::move(draftSin),
          {targetRows, draftRows, 0, geometry.target.ropeAxes,
           geometry.draftRotaryPairs()},
          maximumRows);
      return;
    }
    ops::RoPE::addTables(
        graph, std::move(targetPositions), std::move(draftPositions),
        prefillArena->get(PrefillTensor::TargetInverseFrequencies),
        draftIf,
        std::move(targetCos), std::move(targetSin), std::move(draftCos),
        std::move(draftSin),
        {targetRows, draftRows, geometry.target.rotaryPairs,
         geometry.target.ropeAxes, geometry.draftRotaryPairs()},
        maximumRows);
  }

  void Runtime::Impl::encodeBatchEmbedding(CommandGraph &graph, DecodeTensor tokens,
                            DecodeTensor output, uint32_t lanes,
                            uint32_t rowFactor, uint32_t rowStride) {
    if (!lanes || lanes * rowFactor > kLaneCount)
      throw std::invalid_argument("invalid embedding batch width");
    const uint32_t storage = lanes * rowFactor;
    const uint32_t rows = lanes * rowFactor * rowStride;
    targetModel.addEmbedding(graph, decodeArena->packed(tokens, storage),
                             decodeArena->packed(output, storage), rows);
  }

  // The draft input rows' gather: the draft's own embed_tokens when it
  // ships one (has_embed_tokens — a DSpark draft's trained mask row is
  // not the target table's), else the target's.
  void Runtime::Impl::encodeDraftInputEmbedding(CommandGraph &graph,
                            DecodeTensor tokens, DecodeTensor output,
                            uint32_t lanes, uint32_t rowFactor,
                            uint32_t rowStride) {
    const ops::EmbeddingWeights *own = nullptr;
    if (const auto *dspark = std::get_if<DSparkDraft>(&draftModel);
        dspark && dspark->weights().tokenEmbedding.outputSize)
      own = &dspark->weights().tokenEmbedding;
    if (!own) {
      encodeBatchEmbedding(graph, tokens, output, lanes, rowFactor, rowStride);
      return;
    }
    if (!lanes || lanes * rowFactor > kLaneCount)
      throw std::invalid_argument("invalid embedding batch width");
    const uint32_t storage = lanes * rowFactor;
    const uint32_t rows = lanes * rowFactor * rowStride;
    ops::Embedding::add(graph, decodeArena->packed(tokens, storage), *own,
                        decodeArena->packed(output, storage), rows);
  }

} // namespace richengine::model
