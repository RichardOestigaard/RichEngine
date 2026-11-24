#include "ops/Mhc.hpp"

#include "metal/abi/Mhc.h"
#include "ops/KernelNames.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace richengine::ops {
namespace {

constexpr uint32_t kThreads = RICHENGINE_MHC_ROW_THREADS;

void requireShape(MhcShape shape) {
  if (!shape.hidden || !shape.streams ||
      shape.streams > RICHENGINE_MHC_MAXIMUM_STREAMS ||
      !shape.sinkhornIterations)
    throw std::invalid_argument("invalid mHC shape");
}

uint32_t groups(uint32_t width) { return (width + kThreads - 1) / kThreads; }

MhcParams mhcParams(MhcShape shape, uint32_t rows) {
  return {rows, shape.hidden, shape.streams, shape.sinkhornIterations,
          shape.mixWidth()};
}

} // namespace

void Mhc::addProject(metal::CommandGraph &graph, metal::MetalBuffer streams,
                     metal::MetalBuffer weights, uint32_t columns,
                     metal::MetalBuffer logits, MhcShape shape,
                     uint32_t rows) {
  requireShape(shape);
  if (!columns || columns > RICHENGINE_MHC_MAXIMUM_MIX_WIDTH)
    throw std::invalid_argument("invalid mHC projection width");
  const MhcProjectParams params{rows, columns, shape.width(), 0};
  graph.add(std::string(kMhcProject),
            {std::move(streams), std::move(weights), std::move(logits)},
            params, {columns, rows, 1}, {kThreads, 1, 1});
}

void Mhc::addModule(metal::CommandGraph &graph, metal::MetalBuffer streams,
                    metal::MetalBuffer weights, metal::MetalBuffer alphas,
                    metal::MetalBuffer bias, metal::MetalBuffer mappings,
                    metal::MetalBuffer aggregated, MhcShape shape,
                    uint32_t rows) {
  requireShape(shape);
  if (shape.mixWidth() > RICHENGINE_MHC_MAXIMUM_MIX_WIDTH)
    throw std::invalid_argument("invalid mHC projection width");
  graph.add(std::string(kMhcModule),
            {std::move(streams), std::move(weights), std::move(alphas),
             std::move(bias), std::move(mappings), std::move(aggregated)},
            mhcParams(shape, rows), {rows, 1, 1}, {kThreads, 1, 1});
}

void Mhc::addMappings(metal::CommandGraph &graph, metal::MetalBuffer streams,
                      metal::MetalBuffer logits, metal::MetalBuffer alphas,
                      metal::MetalBuffer bias, metal::MetalBuffer mappings,
                      MhcShape shape, uint32_t rows) {
  requireShape(shape);
  graph.add(std::string(kMhcMappings),
            {std::move(streams), std::move(logits), std::move(alphas),
             std::move(bias), std::move(mappings)},
            mhcParams(shape, rows), {rows, 1, 1}, {kThreads, 1, 1});
}

void Mhc::addWeightedSum(metal::CommandGraph &graph,
                         metal::MetalBuffer streams,
                         metal::MetalBuffer weights, uint32_t weightsStride,
                         uint32_t weightOffset, metal::MetalBuffer out,
                         MhcShape shape, uint32_t rows) {
  requireShape(shape);
  if (weightsStride < uint64_t{weightOffset} + shape.streams)
    throw std::invalid_argument("invalid mHC weights slice");
  const MhcSumParams sum{rows, shape.hidden, shape.streams, weightsStride,
                         weightOffset};
  graph.add(std::string(kMhcWeightedSum),
            {std::move(streams), std::move(weights), std::move(out)}, sum,
            {rows, groups(shape.hidden), 1}, {kThreads, 1, 1});
}

void Mhc::addExpandMix(metal::CommandGraph &graph, metal::MetalBuffer streams,
                       metal::MetalBuffer branch, metal::MetalBuffer mappings,
                       MhcShape shape, uint32_t rows) {
  requireShape(shape);
  graph.add(std::string(kMhcExpandMix),
            {std::move(streams), std::move(branch), std::move(mappings)},
            mhcParams(shape, rows), {rows, groups(shape.hidden), 1},
            {kThreads, 1, 1});
}

void Mhc::addHeadWeights(metal::CommandGraph &graph,
                         metal::MetalBuffer streams, metal::MetalBuffer proj,
                         metal::MetalBuffer base, metal::MetalBuffer scale,
                         metal::MetalBuffer headWeights, MhcShape shape,
                         uint32_t rows) {
  requireShape(shape);
  graph.add(std::string(kMhcHeadWeights),
            {std::move(streams), std::move(proj), std::move(base),
             std::move(scale), std::move(headWeights)},
            mhcParams(shape, rows), {rows, 1, 1}, {kThreads, 1, 1});
}

} // namespace richengine::ops
