#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "ops/KernelNames.hpp"
#include "ops/PagedAttention.hpp"
#include "tuning/HostKvExtents.hpp"
#include "tuning/LinearNumerics.hpp"

#include "NormReference.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

using namespace richengine;
using ops::tuning::HostKvExtents;
using ops::tuning::bf16ToFloat;
using ops::tuning::floatToBf16;
using ops::tuning::ulpBf16;

static_assert(!std::is_aggregate_v<ops::PrefillAttentionPlan> &&
              !std::is_default_constructible_v<ops::PrefillAttentionPlan> &&
              !std::is_copy_assignable_v<ops::PrefillAttentionPlan>);
static_assert(!std::is_aggregate_v<ops::VerifyAttentionPlan> &&
              !std::is_default_constructible_v<ops::VerifyAttentionPlan> &&
              !std::is_copy_assignable_v<ops::VerifyAttentionPlan>);

using richengine::test::require;

template <class Function> void rejects(Function function) {
  try {
    function();
  } catch (const std::invalid_argument &) {
    return;
  }
  throw std::runtime_error("invalid attention plan was accepted");
}

void checkPrefillSlotOrientation(uint32_t queryHeads, kv::Layout layout) {
  bool unequalAxes = false, partialTile = false, multipleSplits = false;
  // Enumerate nonsquare grids and partial final tiles. Each logical partial
  // belongs to exactly one query tile, KV head and balanced history split.
  for (const uint32_t rows : {17U, 257U, 2048U}) {
    const auto plan = ops::PagedAttention::prefillPlan(rows, queryHeads, layout);
    const uint32_t tiles = (rows + 7) / 8;
    require(plan.splitGroups.x == layout.kvHeads &&
                plan.splitGroups.y == tiles && plan.splitGroups.z == plan.splits,
            "prefill split axes must be KV head, query tile, split");
    unequalAxes |= tiles != plan.splits;
    partialTile |= rows % 8 != 0;
    multipleSplits |= plan.splits > 1;
    const uint64_t slots = uint64_t{tiles} * layout.kvHeads * plan.splits;
    std::vector<uint8_t> visits(slots, 0);
    for (uint32_t z = 0; z < plan.splitGroups.z; ++z)
      for (uint32_t y = 0; y < plan.splitGroups.y; ++y)
        for (uint32_t x = 0; x < plan.splitGroups.x; ++x) {
          // group.y selects the query tile and group.z its balanced split;
          // scratch is [tile][KV head][split].
          const uint64_t slot = (uint64_t{y} * layout.kvHeads + x) * plan.splits + z;
          require(slot < visits.size() && visits[slot] == 0,
                  "prefill grid aliases or exceeds a logical partial slot");
          ++visits[slot];
          require(slot % plan.splits == z &&
                      (slot / plan.splits) % layout.kvHeads == x &&
                      slot / (uint64_t{plan.splits} * layout.kvHeads) == y,
                  "prefill dispatch orientation changed logical scratch ownership");
        }
    require(std::all_of(visits.begin(), visits.end(),
                        [](uint8_t count) { return count == 1; }),
            "prefill grid omitted a logical partial slot");
  }
  require(unequalAxes && partialTile && multipleSplits,
          "prefill slot orientation cases omitted unequal axes, partial tiles or splits");
}

// The production GQA geometries as (query heads, KV heads, head dimension)
// triples. kv4 carries two group sizes, each its own kernel family; the
// d64/d128 cores cover the LFM2, Granite and dense layouts.
constexpr std::array<std::array<uint32_t, 3>, 7> kGeometries{
    {{24, 4, 256}, {16, 4, 256}, {16, 2, 256}, {16, 2, 128},
     {32, 8, 64}, {40, 8, 64}, {32, 8, 128}}};

// The declared attention core each fixture geometry binds (the family
// makers declare these; the fixtures mirror them).
[[nodiscard]] constexpr kv::Core coreFor(uint32_t queryHeads,
                                         uint32_t kvHeads,
                                         uint32_t headDim) noexcept {
  if (headDim == 128) {
    if (kvHeads == 2) return kv::Core::Kv2Group8D128;
    return queryHeads == 32 ? kv::Core::Kv8Group4D128 : kv::Core::Unspecified;
  }
  if (headDim == 64)
    return queryHeads == 32 ? kv::Core::Kv8Group4D64 : kv::Core::Kv8Group5D64;
  if (kvHeads == 4)
    return queryHeads == 24 ? kv::Core::Kv4Group6 : kv::Core::Kv4Group4;
  return kv::Core::Kv2Group8;
}

[[nodiscard]] kv::Layout testLayout(uint32_t queryHeads, uint32_t kvHeads,
                                    uint32_t headDim,
                                    kv::Format format) noexcept {
  kv::Layout layout{1, kvHeads, headDim, format};
  layout.attentionCore = coreFor(queryHeads, kvHeads, headDim);
  return layout;
}

std::string geometrySuffix(uint32_t queryHeads, kv::Layout layout) {
  if (layout.headDimension == 128)
    return layout.kvHeads == 2 ? "_hd128" : "_k8q4d128";
  if (layout.headDimension == 64)
    return queryHeads == 32 ? "_hd64" : "_k8q5d64";
  return layout.kvHeads == 2    ? "_kv2_g8"
         : queryHeads == 16     ? "_kv4_g4"
                                : "";
}

void checkPlans(uint32_t queryHeads, kv::Layout layout) {
  const std::string geometrySuffix = ::geometrySuffix(queryHeads, layout);
  const std::array<uint32_t, 1> zeroHistory{};
  // RICHENGINE_ATTENTION_ACC16's _a16 twins: every geometry here is a dense,
  // windowless, non-Gemma layout, so the plans take the tag wherever a twin
  // was emitted — every prefill split, and the verify _rP/_m8 forms (the
  // R2-off base name keeps fp32).
  const bool acc16 = ops::PagedAttention::attentionAcc16Enabled();
  const std::string prefillSplit = std::string(
      layout.format == kv::Format::Int8      ? ops::kPrefillAttentionQ8Split
      : layout.format == kv::Format::Int4    ? ops::kPrefillAttentionInt4Split
                                             : ops::kPrefillAttentionBf16Split) +
      geometrySuffix + (acc16 ? "_a16" : "");
  const std::string prefillReduce = std::string(ops::kPrefillAttentionReduce) + geometrySuffix;
  checkPrefillSlotOrientation(queryHeads, layout);
  for (uint32_t rows = 1; rows <= 2048; ++rows) {
    const auto plan = ops::PagedAttention::prefillPlan(rows, queryHeads, layout);
    const uint32_t tiles = (rows + 7) / 8;
    const uint32_t splits = std::clamp(32U / tiles, 1U, 32U);
    require(plan.rows == rows && plan.splits == splits, "prefill plan lost actual rows");
    require(plan.splitPipeline == prefillSplit && plan.reducePipeline == prefillReduce,
            "prefill plan runs the wrong pipelines");
    require(plan.splitGroups.x == layout.kvHeads &&
                plan.splitGroups.y == tiles && plan.splitGroups.z == splits &&
                plan.reduceGroups.x == layout.kvHeads &&
                plan.reduceGroups.y == 8 * queryHeads / layout.kvHeads &&
                plan.reduceGroups.z == tiles,
            "prefill split/reduce geometry disagrees");
    const uint64_t fused = uint64_t{tiles} * splits * 8 * queryHeads;
    require(plan.workspace.partialsBytes == fused * layout.headDimension * 2 &&
                plan.workspace.statisticsBytes == fused * 2 * 4,
            "prefill split dispatch and exact scratch disagree");
    const auto bound = ops::PagedAttention::prefillWorkspace(rows, queryHeads, layout);
    require(bound.partialsBytes >= plan.workspace.partialsBytes &&
                bound.statisticsBytes >= plan.workspace.statisticsBytes,
            "prefill arena omitted a valid shorter plan");
  }
  // At a wide chain stride the verify splits fused-row passes across
  // threadgroups: _r2 pairs for the kv4 layouts (M=64/96 halve to ~8-13 KB
  // scratch, three resident groups), the four-way _r4 for kv2/g8 whose
  // M=128 tile still caps residency halved. An eight-row chain is the
  // narrow _m8 tile.
  const bool narrow = kv::kVerifyRows == 8;
  const bool chainR2 = !narrow && !richengine::tuning().verifySplitR2Off;
  const uint32_t splitPasses =
      narrow ? 0
      : (layout.attentionCore == kv::Core::Kv2Group8 ||
         layout.attentionCore == kv::Core::Kv2Group8D128)
          ? 4
          : 2;
  const std::string verifySplit = std::string(
      layout.format == kv::Format::Int8      ? ops::kVerifyAttentionQ8Split
      : layout.format == kv::Format::Int4    ? ops::kVerifyAttentionInt4Split
                                             : ops::kVerifyAttentionBf16Split) +
      geometrySuffix +
      (narrow ? "_m8" : "_r" + std::to_string(splitPasses)) +
      (acc16 && (narrow || chainR2) ? "_a16" : "");
  const std::string verifyReduce = std::string(ops::kVerifyAttentionReduce) +
                                   geometrySuffix + (narrow ? "_m8" : "");
  for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
    const std::array<uint32_t, 4> histories{0, 31, 16384, 131072};
    const auto plan = ops::PagedAttention::verifyPlan(lanes, queryHeads, layout,
                                                      std::span(histories).first(lanes));
    require(plan.splitPipeline == verifySplit && plan.reducePipeline == verifyReduce,
            "verify plan runs the wrong pipelines");
    uint32_t maximum = 0;
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      const uint32_t expected = kv::verifyAttentionSplits(histories[lane]);
      require(plan.laneSplits[lane] == expected && expected >= kv::kVerifySplits &&
                  expected <= kv::kVerifyMaximumSplits,
              "verify lane split count does not follow its own history");
      maximum = std::max(maximum, expected);
    }
    const uint64_t fused =
        uint64_t{lanes} * kv::kVerifyRows * kv::kVerifyMaximumSplits * queryHeads;
    require(plan.splits == maximum &&
                plan.workspace.partialsBytes == fused * layout.headDimension * 2 &&
                plan.workspace.statisticsBytes == fused * 2 * 4 &&
                plan.splitGroups.x == layout.kvHeads * (splitPasses ? splitPasses : 1) &&
                plan.splitGroups.y == plan.splits &&
                plan.splitGroups.z == lanes &&
                plan.reduceGroups.y == kv::kVerifyRows * queryHeads / layout.kvHeads &&
                plan.reduceGroups.z == lanes,
            "verify split/reduce/scratch disagree");
    require(plan.laneSplits[0] == kv::kVerifySplits &&
                (lanes < 4 || plan.laneSplits[3] == kv::kVerifyPolicySplits),
            "verify partition changed");
  }
  rejects([&] { (void)ops::PagedAttention::prefillPlan(0, queryHeads, layout); });
  rejects([&] { (void)ops::PagedAttention::prefillPlan(2049, queryHeads, layout); });
  rejects([&] { (void)ops::PagedAttention::verifyPlan(0, queryHeads, layout, zeroHistory); });
  rejects([&] { (void)ops::PagedAttention::verifyPlan(5, queryHeads, layout, zeroHistory); });
  rejects([&] {
    const std::array<uint32_t, 2> two{};
    (void)ops::PagedAttention::verifyPlan(3, queryHeads, layout, two);
  });
  rejects([&] {
    const std::array<uint32_t, 4> padded{};
    (void)ops::PagedAttention::verifyPlan(3, queryHeads, layout, padded);
  });
  rejects([&] {
    const std::array<uint32_t, 1> beyond{kv::kMaximumPhysicalTokens};
    (void)ops::PagedAttention::verifyPlan(1, queryHeads, layout, beyond);
  });
  rejects([&] { (void)ops::PagedAttention::verifyPlan(1, queryHeads + 1, layout, zeroHistory); });
}

// The attention layer under test is the second of a pool's two, so its region
// starts past the first one's in every extent.
constexpr uint32_t kLayer = 1;

struct Case final {
  uint32_t queryHeads;
  kv::Layout layout;
  uint32_t lanes;
  uint32_t rows;
  uint32_t stride;
  HostKvExtents pool;
  RichKvLayer layer{};
  metal::MetalBuffer keys;
  metal::MetalBuffer values;
  metal::MetalBuffer queries;
  std::array<metal::MetalBuffer, 4> tables;
  // Each lane's page ids, which its table holds as entries.
  std::array<std::vector<uint32_t>, 4> pages;
  std::array<kv::ChunkedPrefillParams, 4> stores{};
  // The q8q run: the oracle quantizes each fused query row to int8 codes
  // with the int8-activation rule (kernels/common/int8_activations.h).
  bool int8Queries = false;

  uint64_t queryIndex(uint32_t lane, uint32_t head, uint32_t row,
                      uint32_t dimension) const {
    const uint32_t group = queryHeads / layout.kvHeads;
    return (((uint64_t{lane} * layout.kvHeads + head / group) * stride + row) *
                group +
            head % group) *
               layout.headDimension +
           dimension;
  }

  uint32_t page(uint32_t lane, uint32_t token) const { return pages[lane][token / 32]; }

  static float packed4Code(uint8_t byte, uint32_t nibble) {
    const int code = int(byte >> (nibble * 4)) & 0xF;
    return float(code - (code & 0x8 ? 16 : 0));
  }

  // Raw stored codes and their fp32 scales, matching the GPU's fold order:
  // dot of raw codes, then one multiply by the token scale.
  float keyScale(uint32_t lane, uint32_t head, uint32_t token) const {
    if (layout.format == kv::Format::BFloat16) return 1.0f;
    const uint32_t id = page(lane, token);
    return pool.slab<float>(kLayer, RICHENGINE_KV_KEY_SCALES, id)[
        richengine_kv_scale_element(head, token % 32)];
  }

  float valueScale(uint32_t lane, uint32_t head, uint32_t token) const {
    if (layout.format == kv::Format::BFloat16) return 1.0f;
    const uint32_t id = page(lane, token);
    return pool.slab<float>(kLayer, RICHENGINE_KV_VALUE_SCALES, id)[
        richengine_kv_scale_element(head, token % 32)];
  }

  float keyCode(uint32_t lane, uint32_t head, uint32_t token,
                uint32_t dimension) const {
    const uint32_t id = page(lane, token);
    const uint64_t index =
        richengine_kv_key_element_dim(head, token % 32, dimension,
                                      layout.headDimension);
    if (layout.format == kv::Format::BFloat16)
      return bf16ToFloat(pool.slab<uint16_t>(kLayer, RICHENGINE_KV_KEYS, id)[index]);
    if (layout.format == kv::Format::Int4)
      return packed4Code(
          pool.slab<uint8_t>(kLayer, RICHENGINE_KV_KEYS, id)[index / 2],
          uint32_t(index & 1));
    return float(pool.slab<int8_t>(kLayer, RICHENGINE_KV_KEYS, id)[index]);
  }

  float valueCode(uint32_t lane, uint32_t head, uint32_t token,
                  uint32_t dimension) const {
    const uint32_t id = page(lane, token);
    const uint64_t index =
        richengine_kv_value_element_dim(head, token % 32, dimension,
                                        layout.headDimension);
    if (layout.format == kv::Format::BFloat16)
      return bf16ToFloat(pool.slab<uint16_t>(kLayer, RICHENGINE_KV_VALUES, id)[index]);
    if (layout.format == kv::Format::Int4) {
      const uint64_t byte =
          richengine_kv_key_element_dim(head, token % 32, dimension,
                                        layout.headDimension) /
          2;
      return packed4Code(
          pool.slab<uint8_t>(kLayer, RICHENGINE_KV_VALUES, id)[byte], dimension & 1);
    }
    return float(pool.slab<int8_t>(kLayer, RICHENGINE_KV_VALUES, id)[index]);
  }

  float key(uint32_t lane, uint32_t head, uint32_t token,
             uint32_t dimension) const {
    const uint32_t id = page(lane, token);
    const uint64_t index =
        richengine_kv_key_element_dim(head, token % 32, dimension,
                                      layout.headDimension);
    const uint64_t scale = richengine_kv_scale_element(head, token % 32);
    if (layout.format == kv::Format::BFloat16)
      return bf16ToFloat(pool.slab<uint16_t>(kLayer, RICHENGINE_KV_KEYS, id)[index]);
    if (layout.format == kv::Format::Int4)
      return packed4Code(pool.slab<uint8_t>(kLayer, RICHENGINE_KV_KEYS, id)[index / 2],
                         uint32_t(index & 1)) *
             pool.slab<float>(kLayer, RICHENGINE_KV_KEY_SCALES, id)[scale];
    return pool.slab<int8_t>(kLayer, RICHENGINE_KV_KEYS, id)[index] *
           pool.slab<float>(kLayer, RICHENGINE_KV_KEY_SCALES, id)[scale];
  }

  float value(uint32_t lane, uint32_t head, uint32_t token,
               uint32_t dimension) const {
    const uint32_t id = page(lane, token);
    const uint64_t index =
        richengine_kv_value_element_dim(head, token % 32, dimension,
                                        layout.headDimension);
    const uint64_t scale = richengine_kv_scale_element(head, token % 32);
    if (layout.format == kv::Format::BFloat16)
      return bf16ToFloat(pool.slab<uint16_t>(kLayer, RICHENGINE_KV_VALUES, id)[index]);
    if (layout.format == kv::Format::Int4) {
      // Values pack dim pairs at one token, token-major like the keys:
      // byte [head][token][dim/2].
      const uint64_t byte =
          richengine_kv_key_element_dim(head, token % 32, dimension,
                                        layout.headDimension) /
          2;
      return packed4Code(pool.slab<uint8_t>(kLayer, RICHENGINE_KV_VALUES, id)[byte],
                         dimension & 1) *
             pool.slab<float>(kLayer, RICHENGINE_KV_VALUE_SCALES, id)[scale];
    }
    return pool.slab<int8_t>(kLayer, RICHENGINE_KV_VALUES, id)[index] *
           pool.slab<float>(kLayer, RICHENGINE_KV_VALUE_SCALES, id)[scale];
  }
};

metal::MetalBuffer allocate(metal::MetalBackend &backend, uint64_t bytes) {
  if (!bytes) return {};
  auto buffer = test::sharedBuffer(backend, bytes);
  std::memset(buffer.contents(), 0, bytes);
  return buffer;
}

// The lanes' pages are mixed over three or more extents of the pool, boundary
// pages first; oneExtent puts the same pages in one extent, which must give
// the same bits.
Case makeCase(metal::MetalBackend &backend, uint32_t queryHeads,
               kv::Layout layout, uint32_t lanes, uint32_t rows,
               uint32_t history, bool verify, bool oneExtent = false) {
  std::array<uint32_t, 4> historyLengths{};
  std::array<uint32_t, 4> pageCounts{};
  uint32_t allPages = 0;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    // Verify lanes differ by more than a split's worth of pages, so long
    // histories give each lane its own split count under one slot stride.
    historyLengths[lane] = history + (verify ? lane * 649 : 0);
    pageCounts[lane] = (historyLengths[lane] + rows + 31) / 32;
    allPages += pageCounts[lane];
  }
  const auto spread = HostKvExtents::spread(allPages + 2);
  const std::vector<uint32_t> ids =
      HostKvExtents::mixedPages(spread, allPages, history + rows + lanes);
  kv::Layout poolLayout = layout;
  poolLayout.attentionLayers = kLayer + 1;
  Case data{queryHeads, layout, lanes, rows, (rows + 31) / 32 * 32,
            oneExtent ? HostKvExtents(backend, poolLayout,
                                            spread.extentPages * spread.extents, 1)
                      : HostKvExtents(backend, poolLayout, spread.extentPages,
                                            spread.extents),
            {}, {}, {}, {}, {}, {}, {}};
  data.layer = data.pool.layer(kLayer);
  data.keys = allocate(backend, uint64_t{lanes} * layout.kvHeads * data.stride *
                                    layout.headDimension * 2);
  data.values = allocate(backend, data.keys.sizeBytes());
  data.queries = allocate(backend, uint64_t{lanes} * queryHeads * data.stride *
                                      layout.headDimension * 2);
  uint32_t firstPage = 0;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    data.pages[lane].assign(ids.begin() + firstPage,
                            ids.begin() + firstPage + pageCounts[lane]);
    firstPage += pageCounts[lane];
    data.tables[lane] = allocate(backend, uint64_t{pageCounts[lane]} * sizeof(RichKvPage));
    data.pool.writeTable(data.pages[lane], data.tables[lane].contents());
    data.stores[lane] =
        verify ? ops::PagedAttention::verifyParams(historyLengths[lane],
                                                   pageCounts[lane], rows)
               : ops::PagedAttention::prefillParams(historyLengths[lane], rows, data.stride,
                                                    pageCounts[lane]);
    for (uint32_t token = 0; token < historyLengths[lane]; ++token) {
      const uint32_t id = data.page(lane, token);
      for (uint32_t head = 0; head < layout.kvHeads; ++head) {
        const uint64_t slot = richengine_kv_scale_element(head, token % 32);
        if (layout.format == kv::Format::Int4) {
          // INT4 codes are 16x smaller than INT8's; a realistic scale keeps
          // decoded magnitudes comparable so output rounding noise is too.
          data.pool.slab<float>(kLayer, RICHENGINE_KV_KEY_SCALES, id)[slot] = 0.096f;
          data.pool.slab<float>(kLayer, RICHENGINE_KV_VALUE_SCALES, id)[slot] = 0.112f;
        } else if (layout.format == kv::Format::Int8) {
          data.pool.slab<float>(kLayer, RICHENGINE_KV_KEY_SCALES, id)[slot] = 0.006f;
          data.pool.slab<float>(kLayer, RICHENGINE_KV_VALUE_SCALES, id)[slot] = 0.007f;
        }
        for (uint32_t dimension = 0; dimension < layout.headDimension;
             dimension += 2) {
          const auto keyCode = [&](uint32_t d) {
            return layout.format == kv::Format::Int4
                       ? int((token * 37 + head * 101 + d * 17 + token * d * 3 +
                              lane * 7) % 15) - 7
                       : int((token * 37 + head * 101 + d * 17 + token * d * 3 +
                              lane * 7) % 255) - 127;
          };
          const auto valueCode = [&](uint32_t d) {
            return layout.format == kv::Format::Int4
                       ? int((token * 53 + head * 79 + d * 29 + token * d * 5 +
                              lane * 19) % 15) - 7
                       : int((token * 53 + head * 79 + d * 29 + token * d * 5 +
                              lane * 19) % 255) - 127;
          };
          if (layout.format == kv::Format::Int4) {
            const uint64_t keyByte =
                richengine_kv_key_element_dim(head, token % 32, dimension,
                                              layout.headDimension) /
                2;
            const uint64_t valueByte =
                richengine_kv_key_element_dim(head, token % 32, dimension,
                                              layout.headDimension) /
                2;
            data.pool.slab<uint8_t>(kLayer, RICHENGINE_KV_KEYS, id)[keyByte] =
                uint8_t((keyCode(dimension) & 0xF) |
                        (keyCode(dimension + 1) & 0xF) << 4);
            data.pool.slab<uint8_t>(kLayer, RICHENGINE_KV_VALUES, id)[valueByte] =
                uint8_t((valueCode(dimension) & 0xF) |
                        (valueCode(dimension + 1) & 0xF) << 4);
            continue;
          }
          for (const uint32_t d : {dimension, dimension + 1}) {
            const int key = keyCode(d);
            const int value = valueCode(d);
            const uint64_t keyIndex =
                richengine_kv_key_element_dim(head, token % 32, d,
                                              layout.headDimension);
            const uint64_t valueIndex =
                richengine_kv_value_element_dim(head, token % 32, d,
                                                layout.headDimension);
            if (layout.format == kv::Format::Int8) {
              data.pool.slab<int8_t>(kLayer, RICHENGINE_KV_KEYS, id)[keyIndex] = key;
              data.pool.slab<int8_t>(kLayer, RICHENGINE_KV_VALUES, id)[valueIndex] = value;
            } else {
              data.pool.slab<uint16_t>(kLayer, RICHENGINE_KV_KEYS, id)[keyIndex] =
                  floatToBf16(key * 0.006f);
              data.pool.slab<uint16_t>(kLayer, RICHENGINE_KV_VALUES, id)[valueIndex] =
                  floatToBf16(value * 0.007f);
            }
          }
        }
      }
    }
    for (uint32_t row = 0; row < rows; ++row) {
      for (uint32_t head = 0; head < layout.kvHeads; ++head) {
        const uint64_t base = (uint64_t{lane} * layout.kvHeads + head) *
                              data.stride * layout.headDimension;
        for (uint32_t dimension = 0; dimension < layout.headDimension;
             ++dimension) {
          static_cast<uint16_t *>(data.keys.contents())
              [base + row * layout.headDimension + dimension] =
              floatToBf16(float(int((row * 37 + head * 101 + dimension * 17) % 255) - 127) * 0.006f);
          static_cast<uint16_t *>(data.values.contents())[base + dimension * data.stride + row] =
              floatToBf16(float(int((row * 53 + head * 79 + dimension * 29) % 255) - 127) * 0.007f);
        }
      }
      for (uint32_t head = 0; head < queryHeads; ++head)
        for (uint32_t dimension = 0; dimension < layout.headDimension;
             ++dimension)
          static_cast<uint16_t *>(data.queries.contents())[
              data.queryIndex(lane, head, row, dimension)] =
              floatToBf16(float(int((row * 43 + head * 67 + dimension * 11 +
                              head * dimension * 7) % 1019) - 509) / 1018.0f);
    }
  }
  for (uint32_t lane = lanes; lane < 4; ++lane) {
    data.tables[lane] = data.tables[0];
    data.pages[lane] = data.pages[0];
  }
  return data;
}

// Independent scalar softmax over the exact KV pages produced by the store.
// Sampling query rows bounds the full-2048 oracle cost. Each prefill candidate
// and host chunk is checked independently against its own causal KV history.
void checkReference(const Case &data, const std::vector<uint16_t> &actual) {
  double dot = 0, actualSquared = 0, expectedSquared = 0;
  float maximumError = 0;
  double signedError = 0;
  uint64_t count = 0;
  uint32_t worstRow = 0, worstHead = 0, worstDim = 0;
  std::vector<uint32_t> selectedRows{0, std::min(7U, data.rows - 1),
                                     data.rows / 2, data.rows - 1};
  std::sort(selectedRows.begin(), selectedRows.end());
  selectedRows.erase(std::unique(selectedRows.begin(), selectedRows.end()), selectedRows.end());
  for (uint32_t lane = 0; lane < data.lanes; ++lane) {
    for (uint32_t row : selectedRows) {
      for (uint32_t head : {0U, data.queryHeads - 1}) {
        const uint32_t kvHead = head / (data.queryHeads / data.layout.kvHeads);
        const uint32_t tokens = data.stores[lane].committed_tokens + row + 1;
        // The kernels' default qk scale when the layout carries none.
        const float scoreScale =
            data.layout.headDimension == 64    ? 0.125f
            : data.layout.headDimension == 128 ? 0.08838834764831845f
                                               : 0.0625f;
        // A q8q fused row's INT8 codes and scale, by the int8-activation
        // rule the kernel applies (kernels/common/int8_activations.h):
        // q = clamp(rint(x * (127 / m)), -127, 127), the row standing for
        // q * (m / 127). The quotients are correctly rounded on both sides,
        // so the codes and scale match the kernel's bit for bit.
        std::vector<int> queryCodes;
        float queryScale = 1.0f;
        if (data.int8Queries) {
          queryCodes.resize(data.layout.headDimension);
          float largest = 0.0f;
          for (uint32_t dimension = 0; dimension < data.layout.headDimension;
               ++dimension) {
            const float value = bf16ToFloat(static_cast<const uint16_t *>(
                data.queries.contents())[data.queryIndex(lane, head, row,
                                                         dimension)]);
            require(std::isfinite(value), "q8q oracle query is nonfinite");
            largest = std::max(largest, std::abs(value));
          }
          const bool coded = largest > 0x1p-100f;
          const float inverse = coded ? 127.0f / largest : 0.0f;
          queryScale = coded ? largest / 127.0f : 0.0f;
          for (uint32_t dimension = 0; dimension < data.layout.headDimension;
               ++dimension) {
            const float value = bf16ToFloat(static_cast<const uint16_t *>(
                data.queries.contents())[data.queryIndex(lane, head, row,
                                                         dimension)]);
            queryCodes[dimension] = coded
                ? int(std::clamp(std::rint(value * inverse), -127.0f, 127.0f))
                : 0;
          }
        }
        const float effectiveScale = scoreScale * queryScale;
        std::vector<float> scores(tokens);
        float maximum = -std::numeric_limits<float>::infinity();
        for (uint32_t token = 0; token < tokens; ++token) {
          float score = 0;
          if (data.int8Queries) {
            // The kernel's int32 product is exact below 2^24; the double
            // accumulation is exact well past it.
            double dot = 0;
            for (uint32_t dimension = 0; dimension < data.layout.headDimension;
                 ++dimension)
              dot += double(queryCodes[dimension]) *
                     data.keyCode(lane, kvHead, token, dimension);
            score = float(dot);
          } else {
            for (uint32_t dimension = 0; dimension < data.layout.headDimension;
                 ++dimension)
              score += bf16ToFloat(static_cast<const uint16_t *>(
                               data.queries.contents())[
                               data.queryIndex(lane, head, row, dimension)]) *
                       data.keyCode(lane, kvHead, token, dimension);
          }
          scores[token] = score * data.keyScale(lane, kvHead, token) *
                          effectiveScale;
          maximum = std::max(maximum, scores[token]);
        }
        float denominator = 0;
        for (float &score : scores) {
          score = std::exp(score - maximum);
          denominator += score;
        }
        for (uint32_t dimension = 0; dimension < data.layout.headDimension;
             ++dimension) {
          float expected = 0;
          for (uint32_t token = 0; token < tokens; ++token)
            expected += scores[token] * data.valueScale(lane, kvHead, token) *
                        data.valueCode(lane, kvHead, token, dimension);
          expected /= denominator;
          const float value = bf16ToFloat(actual[data.queryIndex(lane, head, row, dimension)]);
          require(std::isfinite(value), "attention output is nonfinite");
          const float error = std::abs(value - expected);
          if (error > maximumError)
            worstRow = row, worstHead = head, worstDim = dimension;
          maximumError = std::max(maximumError, error);
          signedError += value - expected;
          ++count;
          dot += value * expected;
          actualSquared += value * value;
          expectedSquared += expected * expected;
        }
      }
    }
  }
  const double cosine = dot / std::sqrt(actualSquared * expectedSquared);
  // INT4's prob-times-scale fold lands bf16 steps 16x coarser than INT8's at
  // the same decoded magnitudes, so its oracle cosine is inherently lower;
  // errors stay unbiased and near zero (meanSigned ~0).
  const double bound = data.layout.format == kv::Format::Int4 ? 0.998 : 0.9995;
  // RICHENGINE_ATTENTION_ACC16's _a16 splits accumulate the PV product in
  // bfloat: every page iteration rounds the running sum, so long histories
  // drift further from the fp64 oracle than the fp32 accumulator's. The
  // bounds below only reject catastrophic error; the sweep benchmark
  // quantifies the real quality cost.
  // q8q keeps the fp32 accumulator even with the flag on, so its oracle
  // keeps the tight bounds.
  const bool acc16 =
      ops::PagedAttention::attentionAcc16Enabled() && !data.int8Queries;
  const float errorBound = acc16 ? 0.15f : 0.02f;
  const double cosineBound =
      acc16 ? (data.layout.format == kv::Format::Int4 ? 0.95 : 0.97) : bound;
  if (acc16)
    std::cerr << "acc16 oracle: history=" << data.stores[0].committed_tokens
              << " rows=" << data.rows << " format="
              << kv::formatName(data.layout.format)
              << " maxErr=" << maximumError << " cosine=" << cosine
              << " meanSigned=" << signedError / count << '\n';
  if (!(maximumError < errorBound && cosine > cosineBound)) {
    std::cerr << "oracle debug: history=" << data.stores[0].committed_tokens
              << " rows=" << data.rows << " maxErr=" << maximumError
              << " cosine=" << cosine << " meanSigned=" << signedError / count
              << " worst: row=" << worstRow << " head=" << worstHead
              << " dim=" << worstDim << '\n';
    throw std::runtime_error(
        "attention candidate failed scalar KV oracle: history=" +
        std::to_string(data.stores[0].committed_tokens) + " rows=" +
        std::to_string(data.rows) + " maximum_absolute_error=" +
        std::to_string(maximumError) + " cosine=" + std::to_string(cosine));
  }
}

void checkEquivalent(const Case &data, const std::vector<uint16_t> &baseline,
                      const std::vector<uint16_t> &candidate) {
  float maximumError = 0;
  double dot = 0, baselineSquared = 0, candidateSquared = 0;
  for (uint32_t lane = 0; lane < data.lanes; ++lane)
    for (uint32_t head = 0; head < data.queryHeads; ++head)
      for (uint32_t row = 0; row < data.rows; ++row)
        for (uint32_t dimension = 0; dimension < data.layout.headDimension;
             ++dimension) {
          const uint64_t index = data.queryIndex(lane, head, row, dimension);
          const float left = bf16ToFloat(baseline[index]);
          const float right = bf16ToFloat(candidate[index]);
          require(std::isfinite(right), "attention candidate is nonfinite");
          maximumError = std::max(maximumError, std::abs(left - right));
          dot += left * right;
          baselineSquared += left * left;
          candidateSquared += right * right;
        }
  const double cosine = dot / std::sqrt(baselineSquared * candidateSquared);
  if (!(maximumError < 0.02f && cosine > 0.9995))
    throw std::runtime_error(
        "attention candidate differs from baseline: history=" +
        std::to_string(data.stores[0].committed_tokens) + " rows=" +
        std::to_string(data.rows) + " maximum_absolute_error=" +
        std::to_string(maximumError) + " cosine=" + std::to_string(cosine));
}

// Construct expected extents from the source bits, independently of GPU
// stores: every byte of every extent but the stored rows' slots stays.
std::vector<std::vector<std::byte>> expectedBf16Store(const Case &data) {
  std::vector<std::vector<std::byte>> result;
  for (uint32_t extent = 0; extent < data.pool.extentCount(); ++extent) {
    const auto bytes = data.pool.bytes(extent);
    result.emplace_back(bytes.begin(), bytes.end());
  }
  for (unsigned tensor = 0; tensor < 2; ++tensor) {
    const auto *source = static_cast<const uint16_t *>(
        (tensor ? data.values : data.keys).contents());
    for (uint32_t lane = 0; lane < data.lanes; ++lane) {
      for (uint32_t row = 0; row < data.stores[lane].chunk_tokens; ++row) {
        const uint32_t token = data.stores[lane].committed_tokens + row;
        const uint32_t id = data.page(lane, token);
        const auto *page = data.pool.slab<uint16_t>(
            kLayer, tensor ? RICHENGINE_KV_VALUES : RICHENGINE_KV_KEYS, id);
        const uint32_t extent = id / data.pool.extentPages();
        for (uint32_t head = 0; head < data.layout.kvHeads; ++head)
          for (uint32_t d = 0; d < data.layout.headDimension; ++d) {
            const uint64_t element =
                tensor ? richengine_kv_value_element_dim(head, token % 32, d,
                                                         data.layout.headDimension)
                       : richengine_kv_key_element_dim(head, token % 32, d,
                                                       data.layout.headDimension);
            const auto offset = reinterpret_cast<const std::byte *>(page + element) -
                                data.pool.bytes(extent).data();
            const uint64_t base = (uint64_t{lane} * data.layout.kvHeads + head) *
                                  data.stride * data.layout.headDimension;
            const uint64_t input =
                base + (tensor ? d * data.stride + row
                               : row * data.layout.headDimension + d);
            std::memcpy(result[extent].data() + offset, source + input, sizeof(uint16_t));
          }
      }
    }
  }
  return result;
}

void checkBf16Store(const Case &data, const std::vector<std::vector<std::byte>> &expected) {
  for (uint32_t extent = 0; extent < data.pool.extentCount(); ++extent) {
    const auto bytes = data.pool.bytes(extent);
    require(std::equal(bytes.begin(), bytes.end(), expected[extent].begin()),
            "BF16 store changed source bits, history, another layer or an unused page slot");
  }
}

// The packed-INT4 store computed independently: each stored row's symmetric
// [-7, 7] codes, fp32 scale and nibble placement, over the current extents.
std::vector<std::vector<std::byte>> expectedInt4Store(const Case &data) {
  std::vector<std::vector<std::byte>> result;
  for (uint32_t extent = 0; extent < data.pool.extentCount(); ++extent) {
    const auto bytes = data.pool.bytes(extent);
    result.emplace_back(bytes.begin(), bytes.end());
  }
  for (unsigned tensor = 0; tensor < 2; ++tensor) {
    const auto *source = static_cast<const uint16_t *>(
        (tensor ? data.values : data.keys).contents());
    for (uint32_t lane = 0; lane < data.lanes; ++lane) {
      for (uint32_t row = 0; row < data.stores[lane].chunk_tokens; ++row) {
        const uint32_t token = data.stores[lane].committed_tokens + row;
        const uint32_t id = data.page(lane, token);
        const uint32_t extent = id / data.pool.extentPages();
        for (uint32_t head = 0; head < data.layout.kvHeads; ++head) {
          const uint64_t base = (uint64_t{lane} * data.layout.kvHeads + head) *
                                data.stride * data.layout.headDimension;
          const auto element = [&](uint32_t d) {
            return bf16ToFloat(
                source[base + (tensor ? d * data.stride + row
                                     : row * data.layout.headDimension + d)]);
          };
          float maximum = 0;
          for (uint32_t d = 0; d < data.layout.headDimension; ++d)
            maximum = std::max(maximum, std::abs(element(d)));
          const float scale = maximum == 0.0f ? 0.0f : maximum / 7.0f;
          const auto code = [&](uint32_t d) {
            return maximum == 0.0f
                       ? 0
                       : std::clamp(int(std::rint(element(d) * 7.0f / maximum)),
                                    -7, 7);
          };
          const uint64_t scaleIndex = richengine_kv_scale_element(head, token % 32);
          const auto *scalePage = data.pool.slab<float>(
              kLayer, tensor ? RICHENGINE_KV_VALUE_SCALES : RICHENGINE_KV_KEY_SCALES, id);
          const auto scaleOffset =
              reinterpret_cast<const std::byte *>(scalePage + scaleIndex) -
              data.pool.bytes(extent).data();
          std::memcpy(result[extent].data() + scaleOffset, &scale, sizeof(float));
          for (uint32_t d = 0; d < data.layout.headDimension; d += 2) {
            const uint64_t element =
                richengine_kv_key_element_dim(head, token % 32, d,
                                              data.layout.headDimension) /
                2;
            const auto *page = data.pool.slab<uint8_t>(
                kLayer, tensor ? RICHENGINE_KV_VALUES : RICHENGINE_KV_KEYS, id);
            const auto offset =
                reinterpret_cast<const std::byte *>(page + element) -
                data.pool.bytes(extent).data();
            result[extent][offset] = std::byte(
                uint8_t((code(d) & 0xF) | (code(d + 1) & 0xF) << 4));
          }
        }
      }
    }
  }
  return result;
}

void checkInt4Store(const Case &data,
                    const std::vector<std::vector<std::byte>> &expected) {
  // Byte-exact except inside stored slots: the GPU may evaluate the
  // quantization arithmetic one ulp differently, so a stored code may sit
  // one nibble step away and a stored scale one ulp away. Zero-expected
  // bytes are outside every stored slot and must stay zero.
  uint64_t near = 0;
  const uint64_t db = data.layout.dataBytesPerLayerPage();
  const uint64_t sb = data.layout.scaleBytesPerLayerPage();
  const uint64_t ep = data.pool.extentPages();
  const uint64_t layerRegion = ep * 2 * (db + sb);
  const auto isScale = [&](size_t i) {
    const uint64_t local = i % layerRegion;
    return (local >= ep * db && local < ep * (db + sb)) ||
           local >= ep * (2 * db + sb);
  };
  for (uint32_t extent = 0; extent < data.pool.extentCount(); ++extent) {
    const auto bytes = data.pool.bytes(extent);
    const auto &want = expected[extent];
    for (size_t i = 0; i < bytes.size(); ++i) {
      if (bytes[i] == want[i])
        continue;
      if (isScale(i)) {
        const size_t base = i & ~size_t{3};
        float actualScale, expectedScale;
        std::memcpy(&actualScale, bytes.data() + base, sizeof(float));
        std::memcpy(&expectedScale, want.data() + base, sizeof(float));
        require(std::abs(actualScale - expectedScale) <=
                    std::nextafterf(expectedScale, INFINITY) - expectedScale,
                "INT4 store scale differs by more than an ulp");
        continue;
      }
      const auto nibble = [](uint8_t byte, uint shift) {
        const int code = int(byte >> shift) & 0xF;
        return code - (code & 0x8 ? 16 : 0);
      };
      const uint8_t actual = uint8_t(bytes[i]), target = uint8_t(want[i]);
      if (!(std::abs(nibble(actual, 0) - nibble(target, 0)) <= 1 &&
            std::abs(nibble(actual, 4) - nibble(target, 4)) <= 1)) {
        std::cerr << "INT4 store code mismatch: extent=" << extent << " byte=" << i
                  << " actual=0x" << std::hex << int(actual) << " expected=0x"
                  << int(target) << std::dec << '\n';
        require(false,
                "INT4 store code differs by more than one quantization step");
      }
      ++near;
    }
  }
  require(near <= data.lanes * data.rows * data.layout.kvHeads,
          "INT4 store quantization differs everywhere");
}

void checkBf16StoreEdges(metal::MetalBackend &backend) {
  for (const auto &[heads, kvHeads, headDim] : kGeometries) {
    auto data = makeCase(backend, heads,
                         testLayout(heads, kvHeads, headDim,
                                    kv::Format::BFloat16),
                         1, 33, 31, false);
    // Signed zero, subnormals, large finite values, infinities and NaN payloads.
    constexpr std::array<uint16_t, 12> bits{0, 0x8000, 1, 0x8001, 0x007f, 0x0080,
                                          0x4960, 0x7f7f, 0xff7f, 0x7f80, 0xff80, 0x7fc3};
    for (auto buffer : {data.keys, data.values}) {
      auto *source = static_cast<uint16_t *>(buffer.contents());
      for (uint64_t i = 0; i < buffer.sizeBytes() / 2; ++i) source[i] = bits[i % bits.size()];
    }
    const auto expected = expectedBf16Store(data);
    metal::CommandGraph graph;
    ops::PagedAttention::addPrefillStore(graph, data.layer, data.keys, data.values,
                                        data.tables[0], data.stores[0], data.layout);
    (void)backend.submitCommand(graph.dispatches());
    checkBf16Store(data, expected);
  }
}

enum class Phase : uint8_t { Prefill, Verify };

template <Phase phase>
std::vector<uint16_t> run(metal::MetalBackend &backend, Case &data, bool testBounds,
                          uint32_t int8QueriesFromPages = 0) {
  constexpr bool prefill = phase == Phase::Prefill;
  const auto plan = [&] {
    if constexpr (prefill)
      return ops::PagedAttention::prefillPlan(
          data.rows, data.queryHeads, data.layout, false,
          data.stores[0].committed_tokens, int8QueriesFromPages);
    else {
      std::vector<uint32_t> histories;
      for (uint32_t lane = 0; lane < data.lanes; ++lane)
        histories.push_back(data.stores[lane].committed_tokens);
      return ops::PagedAttention::verifyPlan(data.lanes, data.queryHeads, data.layout,
                                            histories, false, 0, data.rows, false,
                                            int8QueriesFromPages);
    }
  }();
  constexpr uint64_t guardBytes = 256;
  const std::array sizes{plan.workspace.partialsBytes, plan.workspace.statisticsBytes,
                         data.queries.sizeBytes()};
  std::array<metal::MetalBuffer, 3> backing, views;
  for (size_t i = 0; i < sizes.size(); ++i) {
    backing[i] = allocate(backend, sizes[i] + 2 * guardBytes);
    std::memset(backing[i].contents(), 0xa5, sizes[i] + 2 * guardBytes);
    views[i] = backend.view(backing[i], guardBytes, sizes[i]);
    std::memset(views[i].contents(), 0, sizes[i]);
  }
  const auto partials = views[0], statistics = views[1], output = views[2];
  auto encode = [&](metal::CommandGraph &graph, metal::MetalBuffer partialBuffer,
                     metal::MetalBuffer statisticsBuffer) {
    if constexpr (prefill) {
      ops::PagedAttention::addPrefill(graph, data.layer, data.queries, output,
                                      partialBuffer, statisticsBuffer, data.tables[0],
                                      data.stores[0], plan);
    } else {
      ops::PagedVerifyBuffers buffers{data.keys, data.values, data.queries,
                                      partialBuffer, statisticsBuffer, output, data.tables, {}};
      ops::PagedAttention::addVerify(graph, data.layer, buffers,
                                     std::span(data.stores).first(plan.lanes), plan);
    }
  };
  if (testBounds) {
    metal::CommandGraph shortGraph;
    if constexpr (prefill) {
      auto mismatch = data.stores[0];
      mismatch.chunk_tokens = plan.rows == 1 ? 2 : plan.rows - 1;
      rejects([&] {
        ops::PagedAttention::addPrefill(shortGraph, data.layer, data.queries, output,
                                       partials, statistics, data.tables[0], mismatch, plan);
      });
      require(shortGraph.empty(), "mismatched prefill plan partially encoded a graph");
    }
    rejects([&] {
      encode(shortGraph, backend.view(partials, 0, partials.sizeBytes() - 4),
             statistics);
    });
    require(shortGraph.empty(), "undersized partial scratch partially encoded a graph");
    rejects([&] {
      encode(shortGraph, partials,
             backend.view(statistics, 0, statistics.sizeBytes() - 4));
    });
    require(shortGraph.empty(), "undersized statistic scratch partially encoded a graph");
  }
  metal::CommandGraph graph;
  if constexpr (prefill)
    ops::PagedAttention::addPrefillStore(graph, data.layer, data.keys, data.values,
                                        data.tables[0], data.stores[0], data.layout);
  encode(graph, partials, statistics);
  if constexpr (prefill) {
    require(graph.dispatches().size() == 3,
            "production prefill should encode store/split/reduce");
    const auto checkDispatch = [&](const auto &dispatch, auto groups, std::string_view pipeline) {
      require(dispatch.pipelineName == pipeline && dispatch.threadgroups.x == groups.x &&
                  dispatch.threadgroups.y == groups.y && dispatch.threadgroups.z == groups.z &&
                  dispatch.bytes.size() == 1 &&
                  dispatch.bytes[0].sizeBytes == sizeof(kv::PrefillAttentionParams),
              "production prefill dispatch departed from its plan");
      kv::PrefillAttentionParams params;
      std::memcpy(&params, dispatch.bytes[0].data, sizeof(params));
      require(params.committed_tokens == data.stores[0].committed_tokens &&
                  params.rows == data.rows &&
                  params.chunk_stride == data.stride &&
                  params.page_table_entries == data.stores[0].page_table_entries &&
                  params.kv.extent_pages == data.layer.extent_pages &&
                  params.kv.offset == data.layer.offset &&
                  params.split_count == plan.splits,
              "recorded prefill ABI does not describe the actual split plan");
    };
    checkDispatch(graph.dispatches()[1], plan.splitGroups, plan.splitPipeline);
    checkDispatch(graph.dispatches()[2], plan.reduceGroups, plan.reducePipeline);
  } else {
    require(graph.dispatches().size() == 3, "production verify should encode store/split/reduce");
    const auto &split = graph.dispatches()[1];
    const auto &reduce = graph.dispatches()[2];
    require(split.pipelineName == plan.splitPipeline && reduce.pipelineName == plan.reducePipeline &&
                split.threadgroups.y == plan.splits && split.threadgroups.z == plan.splitGroups.z &&
                reduce.threadgroups.y == plan.reduceGroups.y &&
                reduce.threadgroups.z == plan.reduceGroups.z,
            "production verify encoding departed from its plan");
  }
  const auto expected =
      data.layout.format == kv::Format::BFloat16 ? expectedBf16Store(data)
      : data.layout.format == kv::Format::Int4   ? expectedInt4Store(data)
                                                 : std::vector<std::vector<std::byte>>{};
  (void)backend.submitCommand(graph.dispatches());
  if (data.layout.format == kv::Format::BFloat16) checkBf16Store(data, expected);
  if (data.layout.format == kv::Format::Int4) checkInt4Store(data, expected);
  for (size_t i = 0; i < sizes.size(); ++i) {
    const auto *bytes = static_cast<const uint8_t *>(backing[i].contents());
    for (uint64_t byte = 0; byte < guardBytes; ++byte)
      require(bytes[byte] == 0xa5 && bytes[guardBytes + sizes[i] + byte] == 0xa5,
              "attention scratch/output write canary changed");
  }
  const auto *values = static_cast<const uint16_t *>(output.contents());
  return {values, values + output.sizeBytes() / 2};
}

void checkPrefill(metal::MetalBackend &backend, uint32_t heads, kv::Layout layout,
                   uint32_t history, uint32_t rows) {
  auto data = makeCase(backend, heads, layout, 1, rows, history, false);
  const auto output = run<Phase::Prefill>(backend, data, true);
  checkReference(data, output);
  checkEquivalent(data, output, run<Phase::Prefill>(backend, data, false));
  auto oneExtent = makeCase(backend, heads, layout, 1, rows, history, false, true);
  require(run<Phase::Prefill>(backend, oneExtent, false) == output,
          "prefill attention over extents differs from one extent of the same pages");
  if (rows == 1057) {
    // Reuse the identical packed BF16 inputs and KV history across unaligned
    // host chunks, checking each path against its independent causal oracle.
    const auto copy = [](metal::MetalBuffer buffer) {
      const auto *begin = static_cast<const uint16_t *>(buffer.contents());
      return std::vector<uint16_t>(begin, begin + buffer.sizeBytes() / 2);
    };
    const auto keys = copy(data.keys), values = copy(data.values), queries = copy(data.queries);
    uint32_t offset = 0;
    for (uint32_t chunk : {3U, 5U, 31U, 509U, 509U}) {
      data.rows = chunk;
      data.stores[0].committed_tokens = history + offset;
      data.stores[0].chunk_tokens = chunk;
      for (uint32_t head = 0; head < layout.kvHeads; ++head)
        for (uint32_t row = 0; row < chunk; ++row)
          for (uint32_t d = 0; d < layout.headDimension; ++d) {
            const uint64_t base =
                uint64_t{head} * data.stride * layout.headDimension;
            static_cast<uint16_t *>(data.keys.contents())
                [base + row * layout.headDimension + d] =
                keys[base + (offset + row) * layout.headDimension + d];
            static_cast<uint16_t *>(data.values.contents())[base + d * data.stride + row] =
                values[base + d * data.stride + offset + row];
          }
      for (uint32_t head = 0; head < heads; ++head)
        for (uint32_t row = 0; row < chunk; ++row)
          for (uint32_t d = 0; d < layout.headDimension; ++d)
            static_cast<uint16_t *>(data.queries.contents())[data.queryIndex(0, head, row, d)] =
                queries[data.queryIndex(0, head, offset + row, d)];
      const auto chunkOutput = run<Phase::Prefill>(backend, data, true);
      checkReference(data, chunkOutput);
      checkEquivalent(data, chunkOutput, run<Phase::Prefill>(backend, data, false));
      offset += chunk;
    }
    require(offset == rows, "chunk comparison dropped logical query rows");
  }
  std::cout << "paged prefill: format=" << kv::formatName(layout.format) << " q=" << heads << " history=" << history
            << " rows=" << rows << " PASS\n";
}

void checkVerify(metal::MetalBackend &backend, uint32_t heads, kv::Layout layout,
                  uint32_t history, uint32_t lanes,
                  uint32_t rows = kv::kVerifyRows) {
  auto data = makeCase(backend, heads, layout, lanes, rows, history, true);
  const auto output = run<Phase::Verify>(backend, data, true);
  checkReference(data, output);
  auto oneExtent = makeCase(backend, heads, layout, lanes, rows, history, true, true);
  require(run<Phase::Verify>(backend, oneExtent, false) == output,
          "verify attention over extents differs from one extent of the same pages");
  std::cout << "paged verify: format=" << kv::formatName(layout.format) << " q=" << heads << " history=" << history
            << " lanes=" << lanes << " rows=" << rows << " PASS\n";
}

// The q/k RMS norms, RoPE and V copy of the attention prepare, prefill and
// verify, against fp64 with norm weights in bf16 or F32 (a GGUF's). Past the
// rotary pairs a row holds the norm rounded once to bf16; each rotated value
// is within an ulp of the fp64 rotation of the bf16-rounded norms plus an ulp
// of the larger input, which covers the fp32 kernel rounding a norm to the
// other bf16 neighbour and a rotation that cancels.
void checkProjection(metal::MetalBackend &backend, uint32_t queryHeads, kv::Layout layout,
                     bool float32, bool verify) {
  constexpr uint32_t kDim = 256, kPairs = 32;
  const uint32_t kvHeads = layout.kvHeads, group = queryHeads / kvHeads;
  const uint32_t lanes = verify ? 3 : 1, rows = verify ? kv::kVerifyRows : 37,
                 stride = verify ? kv::kVerifyChunkStride : 64;
  const uint32_t packedWidth = 2 * queryHeads * kDim + 2 * kvHeads * kDim;
  auto packed = allocate(backend, uint64_t{lanes} * rows * packedWidth * 2);
  auto ropeCos = allocate(backend, uint64_t{lanes} * rows * kPairs * 4);
  auto ropeSin = allocate(backend, ropeCos.sizeBytes());
  auto queries = allocate(backend, uint64_t{lanes} * queryHeads * stride * kDim * 2);
  auto keys = allocate(backend, uint64_t{lanes} * kvHeads * stride * kDim * 2);
  auto values = allocate(backend, keys.sizeBytes());
  uint32_t state = 0x2545F491U + queryHeads + (verify ? 7 : 0);
  const auto unit = [&] {
    state = state * 1664525U + 1013904223U;
    return double(state >> 8) / double(1U << 23) - 1.0;
  };
  auto *packedData = static_cast<uint16_t *>(packed.contents());
  for (uint64_t i = 0; i < packed.sizeBytes() / 2; ++i)
    packedData[i] = floatToBf16(float(2 * unit()));
  for (uint64_t i = 0; i < ropeCos.sizeBytes() / 4; ++i) {
    const double angle = 3.14159265358979 * unit();
    static_cast<float *>(ropeCos.contents())[i] = float(std::cos(angle));
    static_cast<float *>(ropeSin.contents())[i] = float(std::sin(angle));
  }
  const auto weight = [&](uint32_t) { return float(1.0 + 0.3 * unit()); };
  const ops::NormWeights queryNorm = test::makeNormWeights(backend, kDim, float32, weight);
  const ops::NormWeights keyNorm = test::makeNormWeights(backend, kDim, float32, weight);
  const auto addProjection = [&](metal::CommandGraph &graph, const ops::NormWeights &keyWeights) {
    if (verify)
      ops::PagedAttention::addVerifyProjection(graph, packed, queryNorm, keyWeights, ropeCos, ropeSin,
                                               queries, keys, values, queryHeads, layout, lanes);
    else
      ops::PagedAttention::addPrefillProjection(graph, packed, queryNorm, keyWeights, ropeCos, ropeSin,
                                                queries, keys, values, rows, stride, queryHeads,
                                                layout);
  };
  metal::CommandGraph graph;
  addProjection(graph, keyNorm);
  (void)backend.submitCommand(graph.dispatches());

  const auto *queryData = static_cast<const uint16_t *>(queries.contents());
  const auto *keyData = static_cast<const uint16_t *>(keys.contents());
  const auto *valueData = static_cast<const uint16_t *>(values.contents());
  for (uint32_t lane = 0; lane < lanes; ++lane)
    for (uint32_t row = 0; row < rows; ++row) {
      const uint64_t packedRow = (uint64_t{lane} * rows + row) * packedWidth;
      const uint64_t rope = (uint64_t{lane} * rows + row) * kPairs;
      for (uint32_t head = 0; head < queryHeads + kvHeads; ++head) {
        const bool query = head < queryHeads;
        const uint32_t h = query ? head : head - queryHeads;
        const uint16_t *source = packedData + packedRow +
                                 (query ? h * 2 * kDim : 2 * queryHeads * kDim + h * kDim);
        const uint16_t *out =
            query ? queryData + (((uint64_t{lane} * kvHeads + h / group) * stride + row) * group +
                                 h % group) * kDim
                  : keyData + ((uint64_t{lane} * kvHeads + h) * stride + row) * kDim;
        const std::vector<double> normalized =
            test::rmsNorm(source, query ? queryNorm : keyNorm, kDim);
        for (uint32_t d = 2 * kPairs; d < kDim; ++d)
          require(test::roundedOnceToBf16(out[d], normalized[d]),
                  "attention prepare norm differs from the fp64 reference");
        for (uint32_t d = 0; d < kPairs; ++d) {
          const double first = bf16ToFloat(floatToBf16(float(normalized[d])));
          const double second = bf16ToFloat(floatToBf16(float(normalized[d + kPairs])));
          const double c = static_cast<const float *>(ropeCos.contents())[rope + d];
          const double s = static_cast<const float *>(ropeSin.contents())[rope + d];
          const double rotated[2] = {first * c - second * s, second * c + first * s};
          for (uint32_t half = 0; half < 2; ++half)
            require(std::fabs(bf16ToFloat(out[d + half * kPairs]) - rotated[half]) <=
                        ulpBf16(float(rotated[half])) +
                            ulpBf16(float(std::max(std::fabs(first), std::fabs(second)))),
                    "attention prepare rotation differs from the fp64 reference");
        }
        if (!query)
          for (uint32_t d = 0; d < kDim; ++d)
            require(valueData[((uint64_t{lane} * kvHeads + h) * kDim + d) * stride + row] ==
                        source[kvHeads * kDim + d],
                    "attention prepare value copy differs");
      }
    }
  // One dispatch normalizes the queries and the keys, so it takes one norm type.
  if (float32) {
    metal::CommandGraph rejected;
    try {
      addProjection(rejected, {keyNorm.buffer, false});
      throw std::runtime_error("mixed q/k norm types were accepted");
    } catch (const std::invalid_argument &error) {
      require(std::string_view(error.what()) == "query and key norms differ in type",
              "mixed q/k norm types rejected for the wrong reason");
    }
  }
  std::cout << "attention prepare: q=" << queryHeads << (verify ? " verify" : " prefill")
            << (float32 ? " f32" : " bf16") << " norms PASS\n";
}

// The q8q gate's plan names, without a dispatch: the pages-per-split
// threshold and its committed-history and override terms, the format and
// layout exclusions, and the default-off contract every caller keeps.
void checkQ8qGate() {
  const auto dense = testLayout(24, 4, 256, kv::Format::Int8);
  const auto hasQ8q = [](const std::string &name) {
    return name.find("q8q") != std::string::npos;
  };
  // Prefill: 2048 rows is a single split reading every page — over any
  // sane threshold; 64 rows splits four ways and reads a page or two.
  require(hasQ8q(ops::PagedAttention::prefillPlan(2048, 24, dense, false, 0, 8)
                    .splitPipeline),
          "q8q prefill gate missed a long dense INT8 chunk");
  require(!hasQ8q(ops::PagedAttention::prefillPlan(2048, 24, dense).splitPipeline),
          "q8q prefill ran without its pages-per-split threshold");
  require(!hasQ8q(ops::PagedAttention::prefillPlan(64, 24, dense, false, 0, 8)
                     .splitPipeline),
          "q8q prefill ran below its pages-per-split threshold");
  require(hasQ8q(ops::PagedAttention::prefillPlan(64, 24, dense, false, 100000, 8)
                    .splitPipeline),
          "q8q prefill ignored the chunk's committed history");
  require(hasQ8q(ops::PagedAttention::prefillPlan(64, 24, dense, false, 0, 1)
                    .splitPipeline),
          "q8q pages threshold override was ignored");
  for (const auto format : {kv::Format::Int4, kv::Format::BFloat16})
    require(!hasQ8q(ops::PagedAttention::prefillPlan(
                        2048, 24, testLayout(24, 4, 256, format), false, 0, 8)
                    .splitPipeline),
            "q8q prefill ran on non-INT8 pages");
  auto windowed = dense;
  windowed.windowTokens = 512;
  require(!hasQ8q(ops::PagedAttention::prefillPlan(2048, 24, windowed, false, 0, 8)
                     .splitPipeline),
          "q8q prefill ran a windowed layer");
  auto duo = dense;
  duo.duo = {128, 512, 0xf};
  require(hasQ8q(ops::PagedAttention::prefillPlan(2048, 24, duo, false, 65536,
                                               8)
                     .splitPipeline),
          "q8q prefill skipped a DuoAttention layer");
  auto sparseLayout = testLayout(24, 2, 256, kv::Format::Int8);
  sparseLayout.attentionCore = kv::Core::Kv2Group12H256;
  sparseLayout.valuesTokenMajor = true;
  require(!hasQ8q(ops::PagedAttention::prefillPlan(2048, 24, sparseLayout, true,
                                                0, 8)
                     .splitPipeline),
          "q8q prefill ran a sparse layer");
  auto gemma = testLayout(16, 8, 256, kv::Format::Int8);
  gemma.attentionCore = kv::Core::GemmaKv8H256;
  require(!hasQ8q(ops::PagedAttention::prefillPlan(2048, 16, gemma, false, 0, 8)
                     .splitPipeline),
          "q8q prefill ran a Gemma core");
  // Verify: the longest lane's busiest split decides; 16K tokens is far
  // over the threshold, 1K is under it, and tree verify has no q8q kernels.
  const std::array<uint32_t, 1> longHistory{16384};
  const std::array<uint32_t, 1> shortHistory{1023};
  require(hasQ8q(ops::PagedAttention::verifyPlan(1, 24, dense, longHistory, false,
                                              0, 16, false, 8)
                    .splitPipeline),
          "q8q verify gate missed a long dense INT8 lane");
  require(!hasQ8q(ops::PagedAttention::verifyPlan(1, 24, dense, longHistory)
                     .splitPipeline),
          "q8q verify ran without its pages-per-split threshold");
  require(!hasQ8q(ops::PagedAttention::verifyPlan(1, 24, dense, shortHistory,
                                               false, 0, 16, false, 8)
                     .splitPipeline),
          "q8q verify ran below its pages-per-split threshold");
  require(!hasQ8q(ops::PagedAttention::verifyPlan(1, 24, dense, longHistory, true,
                                               0, 16, false, 8)
                     .splitPipeline),
          "q8q verify ran a tree lane");
  require(hasQ8q(ops::PagedAttention::verifyPlan(1, 24, duo, longHistory, false,
                                              0, 16, false, 8)
                    .splitPipeline),
          "q8q verify skipped a DuoAttention layer");
  std::cout << "q8q attention gate: PASS\n";
}

// The history term of the dense prefill splits: pages-per-split raises the
// shared count with the committed history, bounded by the workspace's tile
// product, and sparse and windowed layouts keep the shared splits their
// kernels were measured at. DuoAttention's _duo splits partition each
// head's own page range, so they take the history-scaled count too.
void checkPrefillSplitHistory() {
  const auto dense = testLayout(24, 4, 256, kv::Format::Int8);
  // 256 rows is 32 tiles: the shared policy splits once; the workspace
  // holds eight splits of it (256 tile-products over 32 tiles).
  require(ops::PagedAttention::prefillPlan(256, 24, dense).splits == 1,
          "prefill shared splits changed");
  require(ops::PagedAttention::prefillPlan(256, 24, dense, false, 131072, 0,
                                         128)
              .splits == 8,
          "prefill history splits missed the workspace bound");
  require(ops::PagedAttention::prefillPlan(256, 24, dense, false, 16384, 0,
                                         128)
              .splits == 4,
          "prefill history splits did not scale with the pages");
  require(ops::PagedAttention::prefillPlan(256, 24, dense, false, 131072, 0,
                                         0)
              .splits == 1,
          "prefill pages-per-split 0 kept the history term");
  auto sparseLayout = testLayout(24, 2, 256, kv::Format::Int8);
  sparseLayout.attentionCore = kv::Core::Kv2Group12H256;
  sparseLayout.valuesTokenMajor = true;
  require(ops::PagedAttention::prefillPlan(256, 24, sparseLayout, true,
                                         131072, 0, 128)
              .splits == 1,
          "prefill sparse splits took the history term");
  auto windowed = dense;
  windowed.windowTokens = 512;
  require(ops::PagedAttention::prefillPlan(256, 24, windowed, false, 131072,
                                         0, 128)
              .splits == 1,
          "prefill windowed splits took the history term");
  auto duo = dense;
  duo.duo = {128, 512, 0xf};
  require(ops::PagedAttention::prefillPlan(256, 24, duo, false, 131072, 0,
                                         128)
              .splits == 8,
          "prefill DuoAttention splits skipped the history term");
  // The kv4/g4 core (Ornith's full-attention layers) compiles the same
  // _duo family: the plan must select its kernels, not throw.
  auto duoG4 = testLayout(16, 4, 256, kv::Format::Int8);
  duoG4.duo = {128, 512, 0xf};
  const auto g4Plan = ops::PagedAttention::prefillPlan(
      256, 16, duoG4, false, 131072, 0, 128);
  require(g4Plan.splitPipeline.find("_kv4_g4_duo") != std::string::npos &&
              g4Plan.splits == 8,
          "prefill Kv4Group4 DuoAttention plan missed its _duo splits");
  require(ops::PagedAttention::verifyPlan(1, 16, duoG4,
                                        std::array<uint32_t, 1>{16384},
                                        false, 0, 16, false, 8)
              .splitPipeline.find("_kv4_g4_duo") != std::string::npos,
          "verify Kv4Group4 DuoAttention plan missed its _duo splits");
  // The kv2/g8 core (the qwen3_6moe family's full-attention layers): the
  // chain's fused-128 tile takes the _r4 pass splits like the dense
  // layout, and the M2-halved plain splits cover the non-chain path.
  auto duoG8 = testLayout(16, 2, 256, kv::Format::Int8);
  duoG8.duo = {128, 512, 0x3};
  const auto g8Plan = ops::PagedAttention::prefillPlan(
      256, 16, duoG8, false, 131072, 0, 128);
  require(g8Plan.splitPipeline.find("_kv2_g8_duo") != std::string::npos &&
              g8Plan.splits == 8,
          "prefill Kv2Group8 DuoAttention plan missed its _duo splits");
  require(ops::PagedAttention::verifyPlan(1, 16, duoG8,
                                        std::array<uint32_t, 1>{16384},
                                        false, 0, 16, false, 8)
              .splitPipeline.find("_kv2_g8_duo_r4") != std::string::npos,
          "verify Kv2Group8 DuoAttention plan missed its _duo_r4 splits");
  std::cout << "prefill split history: PASS\n";
}

// The q8q kernels end to end: the plans pick the _q8q splits and the scalar
// oracle — quantizing each fused query row by the same int8-activation rule —
// bounds the output. Prefill covers the plain splits; verify covers the
// chain's _r2/_r4 pass splits and the block-8 draft's _m8 tile.
void checkQ8q(metal::MetalBackend &backend, uint32_t queryHeads,
              kv::Layout layout) {
  constexpr uint32_t kPagesPerSplit = 8;
  {
    auto data = makeCase(backend, queryHeads, layout, 1, 2048, 4096, false);
    data.int8Queries = true;
    const auto plan = ops::PagedAttention::prefillPlan(
        data.rows, queryHeads, layout, false, data.stores[0].committed_tokens,
        kPagesPerSplit);
    require(plan.splitPipeline.find("q8q") != std::string::npos,
            "q8q prefill plan did not select the q8q split");
    checkReference(data, run<Phase::Prefill>(backend, data, true, kPagesPerSplit));
  }
  for (const uint32_t rows : {kv::kVerifyRows, 8U}) {
    auto data = makeCase(backend, queryHeads, layout, 2, rows, 16384, true);
    data.int8Queries = true;
    const auto plan = ops::PagedAttention::verifyPlan(
        data.lanes, queryHeads, layout,
        std::span<const uint32_t>(
            std::array<uint32_t, 2>{data.stores[0].committed_tokens,
                                    data.stores[1].committed_tokens}),
        false, 0, rows, false, kPagesPerSplit);
    require(plan.splitPipeline.find("q8q") != std::string::npos,
            "q8q verify plan did not select the q8q split");
    const auto output = run<Phase::Verify>(backend, data, true, kPagesPerSplit);
    checkReference(data, output);
    auto oneExtent = makeCase(backend, queryHeads, layout, 2, rows, 16384,
                              true, true);
    oneExtent.int8Queries = true;
    require(run<Phase::Verify>(backend, oneExtent, false, kPagesPerSplit) ==
                output,
            "q8q verify over extents differs from one extent of the same pages");
  }
  std::cout << "q8q attention: q=" << queryHeads << " PASS\n";
}

} // namespace

int main(int argc, char **argv) {
  try {
    require(argc >= 1 && argc <= 3, "usage: paged-attention-plan [METALLIB [--long]]");
    for (auto format : {kv::Format::Int8, kv::Format::Int4, kv::Format::BFloat16})
      for (const auto &[heads, kvHeads, headDim] : kGeometries)
        checkPlans(heads, testLayout(heads, kvHeads, headDim, format));
    checkQ8qGate();
    checkPrefillSplitHistory();
    if (argc == 1) {
      std::cout << "paged attention plans: CPU PASS\n";
      return 0;
    }
    metal::MetalBackend backend(argv[1]);
    checkBf16StoreEdges(backend);
    // The prepare kernels do not depend on the KV format; their rope pair
    // count is the head dimension's, so only the d256 geometries exercise
    // the projection reference.
    for (const auto &[heads, kvHeads, headDim] : kGeometries)
      if (headDim == 256)
        for (bool float32 : {false, true})
          for (bool verify : {false, true})
            checkProjection(backend, heads,
                            testLayout(heads, kvHeads, headDim,
                                       kv::Format::Int8),
                            float32, verify);
    if (argc == 3 && std::string_view(argv[2]) == "--long") {
      for (auto format : {kv::Format::Int8, kv::Format::Int4, kv::Format::BFloat16})
        for (const auto &[heads, kvHeads, headDim] : kGeometries)
          for (uint32_t history : {131072U, 260096U}) {
            const kv::Layout layout = testLayout(heads, kvHeads, headDim, format);
            auto prefill = makeCase(backend, heads, layout, 1, 2048, history, false);
            checkReference(prefill, run<Phase::Prefill>(backend, prefill, true));
            auto verify = makeCase(backend, heads, layout, 4, kv::kVerifyRows, history, true);
            checkReference(verify, run<Phase::Verify>(backend, verify, true));
            std::cout << "long attention: format=" << kv::formatName(format)
                      << " q=" << heads << " history=" << history << " PASS\n" << std::flush;
          }
      std::cout << "long paged attention plans: PASS\n";
      return 0;
    }
    require(argc == 2, "usage: paged-attention-plan [METALLIB [--long]]");
    for (auto format : {kv::Format::Int8, kv::Format::Int4, kv::Format::BFloat16})
    for (const auto &[heads, kvHeads, headDim] : kGeometries) {
      const kv::Layout layout = testLayout(heads, kvHeads, headDim, format);
      for (const auto [history, rows] :
           std::array<std::array<uint32_t, 2>, 10>{{{0, 1}, {33, 7}, {255, 17},
                                                  {1023, 8}, {0, 2048},
                                                  {4093, 1057}, {16383, 257},
                                                  {100, 77}, {8064, 130},
                                                  {6145, 2048}}})
        checkPrefill(backend, heads, layout, history, rows);
      for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
        checkVerify(backend, heads, layout, 0, lanes);
        checkVerify(backend, heads, layout, 1023, lanes);
        // The production block-8 verify stride (trained-block drafts):
        // dead rows beyond the live eight are the kernels' problem.
        checkVerify(backend, heads, layout, 0, lanes, 8);
        checkVerify(backend, heads, layout, 1023, lanes, 8);
      }
      checkVerify(backend, heads, layout, 16384, 2);
      checkVerify(backend, heads, layout, 16384, 2, 8);
      if (format == kv::Format::Int8) checkQ8q(backend, heads, layout);
    }
    std::cout << "paged attention plans: PASS\n";
  } catch (const std::exception &error) {
    std::cerr << "paged attention plans: FAIL: " << error.what() << '\n';
    return 1;
  }
}
