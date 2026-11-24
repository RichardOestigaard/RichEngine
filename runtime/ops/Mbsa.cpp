#include "ops/Mbsa.hpp"

#include "metal/abi/Mbsa.h"
#include "ops/BufferExtent.hpp"
#include "ops/KernelNames.hpp"
#include "ops/LaneBindings.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace richengine::ops {

namespace {

constexpr uint64_t kQueryRowBytes =
    MBSA_INDEX_HEADS * MBSA_INDEX_DIMENSION * sizeof(uint16_t);

// The lanes, rows and positions every MBSA dispatch shares.
uint32_t mbsaLanes(const MbsaRows &rows) {
  if (!rows.rows || rows.firstPositions.empty() ||
      rows.firstPositions.size() > RICHENGINE_MAXIMUM_BATCH_WIDTH ||
      rows.index.tables.size() != rows.firstPositions.size())
    throw std::invalid_argument("invalid MBSA rows");
  return static_cast<uint32_t>(rows.firstPositions.size());
}

} // namespace

void Mbsa::addProject(metal::CommandGraph &graph, metal::MetalBuffer input,
                      metal::MetalBuffer queryWeights,
                      metal::MetalBuffer keyWeights,
                      const NormWeights &queryNorm,
                      const NormWeights &keyNorm, metal::MetalBuffer ropeCos,
                      metal::MetalBuffer ropeSin, metal::MetalBuffer queries,
                      const MbsaRows &rows, uint32_t inputWidth) {
  const uint32_t lanes = mbsaLanes(rows);
  if (!queryNorm.float32 || !keyNorm.float32)
    throw std::invalid_argument("the MBSA indexer norms are not F32");
  if (!inputWidth || inputWidth > MBSA_INDEX_INPUT_WIDTH || inputWidth % 4)
    throw std::invalid_argument("invalid MBSA projection input width");
  const uint64_t laneRows = uint64_t{lanes} * rows.rows;
  requireBytes(input, laneRows * inputWidth * sizeof(uint16_t),
               "MBSA projection input");
  requireBytes(queryWeights,
               MBSA_INDEX_HEADS * MBSA_INDEX_DIMENSION * uint64_t{inputWidth} *
                   sizeof(float),
               "MBSA query weights");
  requireBytes(keyWeights, MBSA_INDEX_DIMENSION * uint64_t{inputWidth} *
                               sizeof(float),
               "MBSA key weights");
  requireBytes(queryNorm.buffer, MBSA_INDEX_DIMENSION * sizeof(float), "MBSA query norm");
  requireBytes(keyNorm.buffer, MBSA_INDEX_DIMENSION * sizeof(float), "MBSA key norm");
  requireBytes(ropeCos, laneRows * MBSA_ROPE_PAIRS * sizeof(float), "MBSA RoPE cosine");
  requireBytes(ropeSin, laneRows * MBSA_ROPE_PAIRS * sizeof(float), "MBSA RoPE sine");
  requireBytes(queries, laneRows * kQueryRowBytes, "MBSA index queries");
  MbsaIndexParams params{.lanes = lanes,
                         .rows = rows.rows,
                         .first_position = {},
                         .index_offset = rows.index.offset,
                         .input_width = inputWidth};
  std::ranges::copy(rows.firstPositions, std::begin(params.first_position));
  const auto tables = laneBindings(rows.index.tables);
  graph.add(std::string(kMbsaIndexProject),
            {std::move(input), std::move(queryWeights), std::move(keyWeights),
             queryNorm.buffer, keyNorm.buffer, std::move(ropeCos),
             std::move(ropeSin), std::move(queries), tables[0], tables[1],
             tables[2], tables[3]},
            params, {rows.rows, MBSA_INDEX_HEADS + 1, lanes}, {MBSA_INDEX_DIMENSION, 1, 1});
}

void Mbsa::addRoute(metal::CommandGraph &graph, metal::MetalBuffer queries,
                    metal::MetalBuffer lists, metal::MetalBuffer counts, uint32_t list,
                    uint32_t listStride, uint32_t rowTiles,
                    std::span<const uint32_t> activeRows, const MbsaRows &rows) {
  const uint32_t lanes = mbsaLanes(rows);
  if (!rowTiles || !listStride || activeRows.size() != lanes || !lists || !counts)
    throw std::invalid_argument("invalid MBSA route rows");
  for (uint32_t lane = 0; lane < lanes; ++lane)
    if (activeRows[lane] > rows.rows)
      throw std::invalid_argument("an MBSA lane's live rows exceed its rows");
  // The route reads index keys at most first_position + rows - 1 tokens
  // back; the queries cover the lanes' rows as the project wrote them.
  const uint64_t laneRows = uint64_t{lanes} * rows.rows;
  requireBytes(queries, laneRows * kQueryRowBytes, "MBSA index queries");
  // The last list a dispatch can write: lane lanes - 1's last tile's last
  // group, (list + lanes * rowTiles) * MBSA_INDEX_HEADS exclusive.
  const uint64_t lastList = (uint64_t{list} + uint64_t{lanes} * rowTiles) * MBSA_INDEX_HEADS;
  requireBytes(lists, lastList * listStride * sizeof(uint32_t), "MBSA block list");
  requireBytes(counts, lastList * sizeof(uint32_t), "MBSA block count");
  MbsaRouteParams params{.lanes = lanes,
                         .rows = rows.rows,
                         .first_position = {},
                         .active_rows = {},
                         .row_tiles = rowTiles,
                         .list = list,
                         .list_stride = listStride,
                         .index_offset = rows.index.offset};
  std::ranges::copy(rows.firstPositions, std::begin(params.first_position));
  std::ranges::copy(activeRows, std::begin(params.active_rows));
  const auto tables = laneBindings(rows.index.tables);
  graph.add(std::string(kMbsaRoute),
            {std::move(queries), tables[0], tables[1], tables[2], tables[3],
             std::move(lists), std::move(counts)},
            params, {rowTiles, MBSA_INDEX_HEADS, lanes}, {MBSA_ROUTE_THREADS, 1, 1});
}

} // namespace richengine::ops
