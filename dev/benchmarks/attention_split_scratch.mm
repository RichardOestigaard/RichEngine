// Scratch A/B for attention history-split granularity: encodes the same
// store/split/reduce dispatches PagedAttention::addPrefill/addVerify emit,
// but with a caller-chosen split count, so pages-per-split can be swept
// without touching engine sources. NOT a correctness oracle — it times the
// production kernels; the attention kernel tests are the oracle.
//
// usage: attention-split-scratch METALLIB --shape 27b|35b|minicpm5|lfm2
//        --phase prefill|verify --rows N --history H --lanes N
//        --splits LIST --repeat N
#include "DispatchReplay.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/KvExtent.h"
#include "metal/abi/PagedAttention.h"
#include "ops/PagedAttention.hpp"
#include "tuning/AttentionFixture.hpp"

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace richengine;
using namespace richengine::ops;
using tuning::AttentionFixture;
using tuning::AttentionFixturePlan;
using tuning::AttentionShape;

constexpr uint32_t kMaximumLanes = RICHENGINE_MAXIMUM_BATCH_WIDTH;
constexpr uint32_t kVerifyRows = RICHENGINE_TARGET_VERIFY_ROWS;

struct Geo final {
  AttentionShape shape;
  kv::Core core;
  std::string suffix;        // kernelSuffix of the layout's core
  uint32_t group;            // queryHeads / kvHeads
  uint32_t fusedPrefill;     // 8 * group
  uint32_t fusedVerify;      // 16 * group
  uint32_t passes;           // verify _rP passes (0 = plain split)
  uint32_t splitGroupX;      // verify grid.x = kvHeads * passes (or kvHeads)
  std::string storeSuffix;
};

Geo geoOf(const std::string &shape, kv::Format format) {
  if (shape == "minicpm5")
    return {{16, 2, 128, format}, kv::Core::Kv2Group8D128, "_hd128", 8, 64, 128, 4, 8,
            "_hd128"};
  if (shape == "lfm2")
    return {{32, 8, 64, format}, kv::Core::Kv8Group4D64, "_hd64", 4, 32, 64, 2, 16,
            "_hd64"};
  if (shape == "27b")
    return {{24, 4, 256, format}, kv::Core::Kv4Group6, "", 6, 48, 96, 2, 8, ""};
  return {{16, 2, 256, format}, kv::Core::Kv2Group8, "_kv2_g8", 8, 64, 128, 4, 8,
          "_kv2_g8"};
}

uint32_t shapeLayers(const std::string &shape) {
  if (shape == "minicpm5") return 42;
  if (shape == "lfm2") return 8;
  return shape == "27b" ? 16 : 10;
}

std::string fmtTag(kv::Format format) {
  return format == kv::Format::Int8 ? "q8" : "bf16";
}

// The kernels the production plan names for this geometry (windowless,
// dense, int8/bf16 — no _swa/_duo/_m8/_m2/_sparse variants).
std::string prefillSplitName(const Geo &g, kv::Format format) {
  return std::string("prefill_attention_") + fmtTag(format) + "_split" + g.suffix;
}
std::string prefillReduceName(const Geo &g) {
  return std::string("prefill_attention_reduce") + g.suffix;
}
std::string verifySplitName(const Geo &g, kv::Format format) {
  std::string name = std::string("verify_attention_") + fmtTag(format) + "_split" + g.suffix;
  if (g.passes) name += g.passes == 4 ? "_r4" : "_r2";
  return name;
}
std::string verifyReduceName(const Geo &g) {
  return std::string("verify_attention_reduce") + g.suffix;
}
std::string verifyStoreName(const Geo &g, kv::Format format) {
  return std::string("verify_attention_") + fmtTag(format) + "_store" + g.storeSuffix;
}

AttentionWorkspace scratchForPrefill(const Geo &g, uint32_t tiles,
                                     uint32_t splitsMax) {
  const uint64_t slots = uint64_t{tiles} * splitsMax * g.shape.kvHeads;
  return {slots * g.fusedPrefill * g.shape.headDimension * 2,
          slots * g.fusedPrefill * 2 * sizeof(float)};
}
AttentionWorkspace scratchForVerify(const Geo &g, uint32_t lanes,
                                    uint32_t splitsMax) {
  const uint64_t slots = uint64_t{lanes} * g.shape.kvHeads * splitsMax;
  return {slots * g.fusedVerify * g.shape.headDimension * 2,
          slots * g.fusedVerify * 2 * sizeof(float)};
}

RichKvLayer layerOf(const kv::Layout &pool, uint32_t extentPages,
                    uint32_t layer) {
  return richengine_kv_layer(extentPages, uint32_t(pool.dataBytesPerLayerPage()),
                             uint32_t(pool.scaleBytesPerLayerPage()), layer);
}

// Store + split + reduce of one prefill chunk with `splits` history splits,
// mirroring PagedAttention::addPrefill's dense path.
void addPrefillVariant(metal::CommandGraph &graph, const AttentionFixture &f,
                       const Geo &g, kv::Layout layout, RichKvLayer layer,
                       uint32_t rows, uint32_t stride, uint32_t history,
                       uint32_t pages, uint32_t splits) {
  const kv::ChunkedPrefillParams chunk =
      PagedAttention::prefillParams(history, rows, stride, pages);
  PagedAttention::addPrefillStore(
      graph, layer, f.buffer(AttentionFixturePlan::Tensor::ChunkKeys),
      f.buffer(AttentionFixturePlan::Tensor::ChunkValues),
      f.buffer(AttentionFixturePlan::table(0)), chunk, layout);
  const uint32_t tiles = kv::prefillAttentionTiles(rows);
  kv::PrefillAttentionParams params{history, rows, stride, pages, layer,
                                    splits, 0.0f, 0, 0};
  graph.addPatchable(prefillSplitName(g, layout.format),
                     {f.buffer(AttentionFixturePlan::Tensor::Queries),
                      f.buffer(AttentionFixturePlan::Tensor::Partials),
                      f.buffer(AttentionFixturePlan::Tensor::Statistics),
                      f.buffer(AttentionFixturePlan::table(0))},
                     params, {g.shape.kvHeads, tiles, splits});
  graph.addPatchable(prefillReduceName(g),
                     {f.buffer(AttentionFixturePlan::Tensor::Partials),
                      f.buffer(AttentionFixturePlan::Tensor::Statistics),
                      f.buffer(AttentionFixturePlan::Tensor::Output)},
                     params, {g.shape.kvHeads, g.fusedPrefill, tiles},
                     {g.shape.headDimension, 1, 1});
}

// Store + split + reduce of a verify step with `splits` splits per lane,
// mirroring PagedAttention::addVerify's dense path.
void addVerifyVariant(metal::CommandGraph &graph, const AttentionFixture &f,
                      const Geo &g, kv::Layout layout, RichKvLayer layer,
                      uint32_t lanes, uint32_t history,
                      std::span<const uint32_t> lanePages, uint32_t splits) {
  std::array<kv::ChunkedPrefillParams, kMaximumLanes> stores{};
  std::array<kv::VerifyAttentionParams, kMaximumLanes> attention{};
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    stores[lane] =
        PagedAttention::verifyParams(history, lanePages[lane]);
    stores[lane].kv = layer;
    attention[lane] = {history, lanePages[lane], layer, splits, splits,
                       kVerifyRows, kVerifyRows, 0.0f, 0, 0};
  }
  // Lanes past the plan's lanes bind lane 0's table, as the fixture's
  // padded-lane aliasing does.
  std::array<metal::MetalBuffer, kMaximumLanes> tables{};
  for (uint32_t lane = 0; lane < kMaximumLanes; ++lane)
    tables[lane] = f.buffer(
        AttentionFixturePlan::table(lane < lanes ? lane : 0));
  graph.addPatchable(verifyStoreName(g, layout.format),
                     {f.buffer(AttentionFixturePlan::Tensor::ChunkKeys),
                      f.buffer(AttentionFixturePlan::Tensor::ChunkValues),
                      tables[0], tables[1], tables[2], tables[3]},
                     stores,
                     {uint64_t{lanes} * 2 * kVerifyRows * g.shape.kvHeads, 1, 1},
                     {g.shape.headDimension, 1, 1});
  graph.addPatchable(verifySplitName(g, layout.format),
                     {f.buffer(AttentionFixturePlan::Tensor::Queries),
                      f.buffer(AttentionFixturePlan::Tensor::Partials),
                      f.buffer(AttentionFixturePlan::Tensor::Statistics),
                      tables[0], tables[1], tables[2], tables[3]},
                     attention, {g.splitGroupX, splits, lanes});
  graph.addPatchable(verifyReduceName(g),
                     {f.buffer(AttentionFixturePlan::Tensor::Partials),
                      f.buffer(AttentionFixturePlan::Tensor::Statistics),
                      f.buffer(AttentionFixturePlan::Tensor::Output)},
                     attention, {g.shape.kvHeads, g.fusedVerify, lanes},
                     {g.shape.headDimension, 1, 1});
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

uint32_t parseCount(std::string_view text, uint32_t minimum, uint32_t maximum,
                    std::string_view option) {
  uint32_t value = 0;
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
      value < minimum || value > maximum)
    throw std::invalid_argument(std::string(option) + " requires integers from " +
                                std::to_string(minimum) + " to " +
                                std::to_string(maximum));
  return value;
}
std::vector<uint32_t> parseList(const std::string &text, uint32_t minimum,
                                uint32_t maximum, std::string_view option) {
  std::vector<uint32_t> values;
  size_t start = 0;
  while (start <= text.size()) {
    const size_t comma = text.find(',', start);
    const std::string item = text.substr(
        start, comma == std::string::npos ? std::string::npos : comma - start);
    values.push_back(parseCount(item, minimum, maximum, option));
    if (comma == std::string::npos) break;
    start = comma + 1;
  }
  return values;
}

} // namespace

int main(int argc, const char *argv[]) {
  try {
    if (argc < 2) {
      std::cerr << "usage: attention-split-scratch METALLIB --shape S --phase P "
                   "--rows N --history H --lanes N --splits LIST --repeat N\n";
      return 64;
    }
    std::string shape = "35b", phase = "prefill";
    uint32_t rows = 0, history = 0, lanes = 1, repeat = 5;
    std::vector<uint32_t> splits;
    for (int index = 2; index < argc; index += 2) {
      const std::string option(argv[index]);
      if (index + 1 >= argc)
        throw std::invalid_argument(option + " requires a value");
      const std::string value(argv[index + 1]);
      if (option == "--shape") shape = value;
      else if (option == "--phase") phase = value;
      else if (option == "--rows") rows = parseCount(value, 1, 2048, option);
      else if (option == "--history")
        history = parseCount(value, 0, kv::kMaximumPhysicalTokens - 2048, option);
      else if (option == "--lanes") lanes = parseCount(value, 1, kMaximumLanes, option);
      else if (option == "--repeat") repeat = parseCount(value, 1, 1000, option);
      else if (option == "--splits")
        splits = parseList(value, 1, phase == "prefill" ? 32 : 128, option);
      else throw std::invalid_argument("unknown option " + option);
    }
    if (phase != "prefill" && phase != "verify")
      throw std::invalid_argument("--phase takes prefill or verify");
    if (phase == "prefill" && !rows)
      throw std::invalid_argument("--rows is required for prefill");
    if (phase == "verify") rows = kVerifyRows;
    const Geo g = geoOf(shape, kv::Format::Int8);
    const bool prefill = phase == "prefill";
    const uint32_t splitsMax = *std::max_element(splits.begin(), splits.end());

    metal::MetalBackend backend(argv[1]);
    std::cerr << "device " << backend.capabilities().deviceName
              << ", shape " << shape << " " << phase << " rows=" << rows
              << " history=" << history << " lanes=" << lanes << "\n";

    // Fixture: the production plans' largest workspace covers every split.
    AttentionFixturePlan::Histories histories{};
    std::fill_n(histories.begin(), lanes, history);
    // The verify plan's workspace bound covers the maximum split count for
    // every lane regardless of the candidate count.
    const AttentionWorkspace scratch =
        prefill ? scratchForPrefill(g, kv::prefillAttentionTiles(rows), splitsMax)
                : scratchForVerify(g, lanes, kv::kVerifyMaximumSplits);
    kv::Layout poolLayout{shapeLayers(shape), g.shape.kvHeads,
                          g.shape.headDimension, kv::Format::Int8};
    poolLayout.attentionCore = g.core;
    const uint32_t extentPages =
        poolLayout.maximumExtentPages();
    AttentionFixturePlan plan = AttentionFixturePlan::make(
        g.shape, lanes, rows, histories, scratch,
        {2, 1, extentPages});
    AttentionFixture fixture(backend, plan, "split-scratch");
    if (!fixture.fill())
      throw std::runtime_error("fixture fill stopped");
    kv::Layout layout{1, g.shape.kvHeads, g.shape.headDimension,
                      kv::Format::Int8};
    layout.attentionCore = g.core;
    const RichKvLayer layer =
        layerOf(poolLayout, extentPages, 1);

    // Lane page counts for verify params.
    std::vector<uint32_t> lanePages(lanes);
    for (uint32_t lane = 0; lane < lanes; ++lane)
      lanePages[lane] = (history + rows + kv::kPageTokens - 1) / kv::kPageTokens;

    struct Variant {
      std::string name;
      uint32_t splits;
      metal::CommandGraph graph;
    };
    std::vector<Variant> variants;
    // The production plan first: validates the hand-encoding against the
    // real dispatch and gives the shipped policy's number in the same run.
    {
      metal::CommandGraph graph;
      if (prefill) {
        const auto attention =
            PagedAttention::prefillPlan(rows, g.shape.queryHeads, layout);
        fixture.addGraph(graph, attention);
      } else {
        const auto attention = PagedAttention::verifyPlan(
            lanes, g.shape.queryHeads, layout,
            std::span(histories).first(lanes));
        fixture.addGraph(graph, attention);
      }
      variants.push_back({"prod", 0, std::move(graph)});
    }
    for (uint32_t s : splits) {
      metal::CommandGraph graph;
      if (prefill)
        addPrefillVariant(graph, fixture, g, layout, layer, rows, plan.stride,
                          history, plan.pages[0], s);
      else
        addVerifyVariant(graph, fixture, g, layout, layer, lanes, history,
                         lanePages, s);
      variants.push_back({"s" + std::to_string(s), s, std::move(graph)});
    }

    // Warm every variant, then alternate order.
    double warmup = 0.0;
    while (warmup < 1.5)
      for (auto &v : variants)
        warmup += backend.submitCommand(v.graph.dispatches()).gpuSeconds;

    std::vector<std::vector<double>> fused(variants.size());
    std::vector<std::map<std::string, std::vector<double>>> perPipe(
        variants.size());
    for (uint32_t round = 0; round < repeat; ++round)
      for (size_t offset = 0; offset < variants.size(); ++offset) {
        const size_t i = (round + offset) % variants.size();
        fused[i].push_back(
            backend.submitCommand(variants[i].graph.dispatches()).gpuSeconds *
            1000.0);
      }
    for (uint64_t round = 0; round <= repeat; ++round)
      for (size_t offset = 0; offset < variants.size(); ++offset) {
        const size_t i = (round + offset) % variants.size();
        const auto run =
            benchmark::replayDispatches(backend, variants[i].graph.dispatches());
        if (round)
          for (const auto &[name, seconds] : run)
            perPipe[i][name].push_back(seconds * 1000.0);
      }
    std::cout << "{\"shape\":\"" << shape << "\",\"phase\":\"" << phase
              << "\",\"rows\":" << rows << ",\"history\":" << history
              << ",\"lanes\":" << lanes << ",\"variants\":[";
    for (size_t i = 0; i < variants.size(); ++i) {
      std::cout << (i ? "," : "") << "{\"name\":\"" << variants[i].name
                << "\",\"splits\":" << variants[i].splits
                << ",\"fused_ms\":" << median(fused[i]) << ",\"pipelines\":{";
      bool first = true;
      for (const auto &[name, samples] : perPipe[i]) {
        std::cout << (first ? "" : ",") << "\"" << name << "\":"
                  << median(samples);
        first = false;
      }
      std::cout << "}}";
      std::cerr << variants[i].name << " fused=" << median(fused[i])
                << " ms\n";
    }
    std::cout << "]}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "attention-split-scratch: " << error.what() << '\n';
    return 70;
  }
}
