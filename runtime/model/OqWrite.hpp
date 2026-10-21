#pragma once

// The writer of an Oq section (SectionKind::Oq, planned by AffineTarget.cpp
// against the checkpoint's per-module quantization): a run of segments in
// the MDGG0001 layout — a GgufTensorDescriptor, the plane0 code tiles and
// the meta plane, each at a 16 KiB boundary — which the block readers
// (model/WeightStore.cpp, OqTargetFormat) consume as QuantizedSegments.
//
// The parts tile the section's rows in order. A run of consecutive
// quantized parts whose row counts all fill whole 256-row tiles becomes one
// segment per part at the part's own format; a run holding an unaligned
// part becomes one segment at the run's finest format — its maximum bits,
// its minimum group size — whose repack re-emits codes at the target width
// and replicates meta over the target's units, both lossless. An
// unquantized part is one F32 float segment. The token table's section is
// its descriptor and the native rows the embedding kernels gather.

#include "model/AffinePreparation.hpp"

#include <cstdint>
#include <functional>
#include <vector>

namespace richengine::model::affine::oq {

// The section bytes the parts' segments occupy, planned before the section
// is appended to its image.
[[nodiscard]] uint64_t sectionBytes(const Section &section);

// writeAffineImage's SectionKind::Oq writer: the segments' descriptors,
// plane0/meta tiles and float or native rows.
void addTasks(const Section &section, uint8_t *at,
              std::vector<std::function<void(std::vector<uint8_t> &)>> &tasks);

} // namespace richengine::model::affine::oq
