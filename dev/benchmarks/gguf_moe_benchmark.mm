// GPU time of one sparse MoE layer at the 35B shape (hidden 2048, 256 experts, top 8, intermediate 512) in GGUF
// (by default the 35B UD-Q4_K_M's Q4_K gate/up and Q5_K down experts, a Q8_0 shared expert, F32 router) against the
// affine Q4 layer, on the device's plans and the other GGUF tile: a 2048-row prefill chunk, decode B1-B4 and prefill
// chunks of their rows, prefill chunks of 64 to 512 rows, then every dispatch of the long chunk, B1 and B4 replayed as
// its own command. The numbers behind the MoE plans of ops/MoE.cpp:
//   gguf-moe-benchmark <metallib> [rounds] [gate/up format] [down format]
// Printed are medians of GPU ms per layer over `rounds` (20) commands. Every row routes to 8 experts of a pool of 24
// per request lane (decode, short chunks) or of all 256 (chunks of 64 rows or more), identically for both formats:
// expert e scores 4 x[e], so the first 256 inputs pick the routes. The weights exceed the system cache, so each layer
// streams its experts from DRAM.
//   gguf-moe-benchmark <metallib> int8 [rounds] [gate/up format] [down format] [shared format] [a1b]
// - prefill chunks of 64 to 2048 rows routed at random on the staged expert passes and on the int8 passes
//   (MoeConfig::int8Passes, kernels/shared/moe_int8.metal: the MLX affine formats gguf_prefill_int8_format takes,
//   af4 throughout by default), with the routes per expert, the rows each routed expert takes on average and the
//   passes the device's plan picks (ExecutionPlans::moeConfig's gate). The mode arms
//   RICHENGINE_PREFILL_FAST_INT8 so the gate's model-policy term holds.
//   gguf-moe-benchmark <metallib> bolt [rounds]
// - same-graph direct/indirect A/B of two latent routed-layer shapes from
//   Bolt 74B-A9B 24GB GGUF, with synthetic weights and routing inputs.
#include "../tests/engine/AffineQ4Fixture.hpp"
#include "../tests/engine/GgufFormatReference.hpp"
#include "DispatchReplay.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "ops/ExecutionPlans.hpp"
#include "ops/MoE.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using richengine::metal::BufferStorage;
using richengine::metal::CommandGraph;
using richengine::metal::MetalBackend;
using richengine::metal::MetalBuffer;
using richengine::ops::BlockMoeWeights;
using richengine::ops::MoE;
using richengine::ops::MoeBuffers;
using richengine::ops::MoeConfig;
using richengine::ops::MoeScratchField;
using richengine::ops::kMoeScratchFields;
using richengine::ops::MoeGgufTile;
using richengine::ops::MoePlan;
using richengine::ops::MoeShape;
using richengine::ops::MoeWeights;
using richengine::ops::QuantizedSegment;
using richengine::ops::WeightLayout;
using namespace gguf_reference;

float bf16(double value) { return float(__bf16(float(value))); }

MetalBuffer upload(MetalBackend &backend, const void *data, uint64_t bytes, const char *label) {
  MetalBuffer buffer = backend.allocateBuffer(bytes, BufferStorage::Shared, label);
  std::memcpy(buffer.contents(), data, bytes);
  return buffer;
}

MetalBuffer zeros(MetalBackend &backend, uint64_t bytes, const char *label) {
  if (!bytes) return {};
  MetalBuffer buffer = backend.allocateBuffer(bytes, BufferStorage::Shared, label);
  std::memset(buffer.contents(), 0, bytes);
  return buffer;
}

MetalBuffer bfloatBuffer(MetalBackend &backend, const std::vector<float> &values, const char *label) {
  std::vector<__bf16> bits(values.begin(), values.end());
  return upload(backend, bits.data(), bits.size() * 2, label);
}

// The segments of a repacked [rows, K] tensor in format f and of [rows, K] floats.
QuantizedSegment planeSegment(MetalBackend &backend, Fmt f, const Packed &planes, uint32_t rows, uint32_t K) {
  return QuantizedSegment::planes(
      f, rows, K, upload(backend, planes.w0.data(), planes.w0.size(), "gguf-plane0"),
      kQuantFormats[f].plane1_bytes ? upload(backend, planes.w1.data(), planes.w1.size(), "gguf-plane1")
                                    : MetalBuffer{},
      upload(backend, planes.meta.data(), planes.meta.size(), "gguf-meta"));
}
QuantizedSegment floatSegment(MetalBackend &backend, const std::vector<float> &values, uint32_t rows,
                              uint32_t K) {
  return QuantizedSegment::floats(rows, K, upload(backend, values.data(), values.size() * sizeof(float),
                                                  "gguf-floats"));
}

void allocate(MetalBackend &backend, MoeBuffers &m, const MoePlan &plan) {
  const auto &w = plan.workspace();
  for (const MoeScratchField &field : kMoeScratchFields)
    m.scratch.*field.buffer = zeros(backend, w.*field.bytes, "moe-scratch");
}

int boltTiming(MetalBackend &backend, uint32_t rounds) {
  if (!rounds)
    throw std::invalid_argument("bolt rounds must be positive");
  constexpr uint32_t hidden = 4096, latent = 1024, topK = 9;
  constexpr uint32_t maxExperts = 496, maxIntermediate = 2560;
  const MoeShape shape{.hiddenSize = hidden,
                       .experts = maxExperts,
                       .expertsPerToken = topK,
                       .expertIntermediateSize = maxIntermediate,
                       .weightLayout = WeightLayout::Block32,
                       .expertFormat = GGUF_FMT_IQ2XXS,
                       .sharedExpert = false,
                       .ungatedExperts = true,
                       .expertWeightScale = 5.0f,
                       .latentSize = latent};
  const richengine::ops::ExecutionPlans plans(backend.capabilities());
  std::mt19937 rng(9);
  const auto expertPlanes = [&](uint32_t rows, uint32_t width) {
    auto native = makeNative(IQ2XXS, rows, width, rng, [&] { return f2h(0.001f); });
    return planeSegment(backend, IQ2XXS, repack(IQ2XXS, native, rows, width, nullptr), rows, width);
  };
  const auto median = [](std::vector<double> &samples) {
    std::sort(samples.begin(), samples.end());
    const size_t middle = samples.size() / 2;
    return (samples[middle - 1] + samples[middle]) / 2;
  };
  std::printf("%s: Bolt latent routed MoE, IQ2_XXS, %u samples per arm per case\n",
              backend.capabilities().deviceName.c_str(), 2 * rounds);
  // These are layers 1 and 48 of the 24GB GGUF. The shared expert and
  // latent projections run outside MoE::add on Bolt and are not timed.
  for (const auto [experts, intermediate] :
       {std::pair{240u, 2048u}, std::pair{496u, 2560u}}) {
    std::vector<float> router(uint64_t{experts} * hidden), bias(experts);
    for (uint32_t e = 0; e < experts; ++e)
      router[uint64_t{e} * hidden + e] = 4.0f;
    BlockMoeWeights block;
    block.router = floatSegment(backend, router, experts, hidden);
    block.expertBias = floatSegment(backend, bias, 1, experts);
    block.up.routed = expertPlanes(experts * intermediate, latent);
    block.down.routed = expertPlanes(experts * latent, intermediate);
    const MoeWeights weights = block;
    for (const auto [lanes, stride] :
         {std::pair{1u, 1u}, std::pair{4u, 1u}, std::pair{1u, 8u}}) {
      const uint32_t rows = lanes * stride;
      const auto plan = plans.moeDecode(shape, lanes, stride);
      MoeBuffers b;
      std::vector<float> x(uint64_t{rows} * hidden, -0.25f),
          expertInput(uint64_t{rows} * latent);
      std::uniform_real_distribution<float> value(-1.0f, 1.0f);
      for (float &v : expertInput) v = value(rng);
      // Nine routes per token, with overlap inside a packed row group.
      for (uint32_t r = 0; r < rows; ++r)
        for (uint32_t k = 0; k < topK; ++k)
          x[uint64_t{r} * hidden + (r * 3 + k) % experts] = 1.0f - 0.05f * k;
      b.input = bfloatBuffer(backend, x, "bolt-input");
      b.expertInput = bfloatBuffer(backend, expertInput, "bolt-latent");
      b.residual = zeros(backend, uint64_t{rows} * latent * 2, "bolt-residual");
      b.output = zeros(backend, uint64_t{rows} * latent * 2, "bolt-output");
      allocate(backend, b, plan);
      CommandGraph graph;
      MoE::add(graph, b, weights, plan);
      std::vector<richengine::metal::ComputeDispatch> direct(graph.dispatches().begin(),
                                                             graph.dispatches().end());
      size_t indirectPasses = 0;
      for (auto &d : direct) {
        if (!d.indirectGrid) continue;
        ++indirectPasses;
        d.indirectGrid = {};
      }
      if (indirectPasses != 2)
        throw std::runtime_error("expected two indirect routed expert passes");
      const auto check = [&](std::span<const richengine::metal::ComputeDispatch> dispatches,
                             int sentinel) {
        for (const auto &buffer : {b.output, b.scratch.expertIntermediate, b.scratch.expertOutput})
          std::memset(buffer.contents(), sentinel, buffer.sizeBytes());
        static_cast<void>(backend.submitCommand(dispatches));
        const auto *bits = static_cast<const uint16_t *>(b.output.contents());
        return std::vector<uint16_t>(bits, bits + uint64_t{rows} * latent);
      };
      const auto reference = check(direct, 0x3c);
      if (reference != check(graph.dispatches(), 0xa5))
        throw std::runtime_error("Bolt latent routed output differs between dispatch modes");
      const auto live = *static_cast<const uint32_t *>(b.scratch.tileCount.contents());
      if (!live || live > plan.maximumTiles())
        throw std::runtime_error("invalid Bolt tile count");
      // Two warmups per arm, then ABBA; use GPU time, not graph encoding.
      std::array<std::vector<double>, 2> samples;
      for (uint32_t r = 0; r <= rounds; ++r) {
        for (const unsigned arm : {0u, 1u, 1u, 0u}) {
          const auto dispatches = arm ? graph.dispatches()
              : std::span<const richengine::metal::ComputeDispatch>(direct);
          const auto ms = backend.submitCommand(dispatches).gpuSeconds * 1000.0;
          if (r) samples[arm].push_back(ms);
        }
      }
      const double a = median(samples[0]), bms = median(samples[1]);
      std::printf("  E%u I%u B%u stride%u tiles %u/%u: direct %.5f ms, indirect %.5f ms, "
                  "latency reduction %+.2f%%, bitwise match\n",
                  experts, intermediate, lanes, stride, live, plan.maximumTiles(),
                  a, bms, 100.0 * (1.0 - bms / a));
    }
  }
  return 0;
}

int timing(MetalBackend &backend, uint32_t rounds, bool int8, Fmt gateUpFormat, Fmt downFormat, Fmt sharedFormat,
           bool a1b) {
  // The 35B shape by default; `a1b` is LFM2.5-8B-A1B's sparse block: 32
  // experts, top 4, intermediate 1792, sigmoid-bias router, no shared expert.
  const uint32_t H = 2048, I = a1b ? 1792 : 512, E = a1b ? 32 : 256,
                 TOPK = a1b ? 4 : 8, kRowsMax = 2048;
  // gate and up hold most of the routed expert weights.
  const MoeShape affineShape{H, E, TOPK, I}, ggufShape{H, E, TOPK, I, WeightLayout::Block32,
                                                       uint32_t(gateUpFormat), !a1b};
  std::mt19937 local(9);
  // Affine: random Q4 slabs with finite scales, the router and shared gate of the routing fixture.
  const auto affineExperts = [&](uint32_t experts, uint32_t n, uint32_t k) {
    const uint64_t parameters = uint64_t{n} * k / 64;
    const uint16_t scale = std::bit_cast<uint16_t>(__bf16(0.01f)), bias = std::bit_cast<uint16_t>(__bf16(-0.05f));
    return richengine::test::expertSlabs(
        backend, experts, n, k, "affine-experts", [&](uint32_t, const richengine::test::AffineQ4Planes &planes) {
          for (uint64_t i = 0; i < parameters * 32; ++i) planes.weights[i] = uint8_t(local());
          std::fill_n(planes.scales, parameters, scale);
          std::fill_n(planes.biases, parameters, bias);
        });
  };
  const auto affineRouter = [&](bool routes) {
    const uint64_t elements = uint64_t{256} * H;
    MetalBuffer weights = zeros(backend, elements, "router-weights"),
                scales = zeros(backend, elements / 32, "router-scales"),
                biases = zeros(backend, elements / 32, "router-biases");
    auto *w = static_cast<uint8_t *>(weights.contents());
    auto *sc = static_cast<__bf16 *>(scales.contents());
    for (uint32_t e = 0; e < 256; ++e) {
      if (!routes) break;
      w[(uint64_t(e / 64) * 256 + e) * 64 + e % 64] = 1;
      sc[(e / 64) * 256 + e] = __bf16(4.0f);
    }
    return richengine::ops::Q8Projection{{weights, scales, biases}, 256, H};
  };
  MoeWeights affine = richengine::ops::AffineMoeWeights{
      .router = affineRouter(true),
      .expertGate = affineExperts(E, I, H),
      .expertUp = affineExperts(E, I, H),
      .expertDown = affineExperts(E, H, I),
      .sharedGate = affineExperts(1, I, H),
      .sharedUp = affineExperts(1, I, H),
      .sharedDown = affineExperts(1, H, I),
      .sharedScalarGate = affineRouter(false),
  };
  // GGUF: the same routing in an F32 router.
  const auto planes = [&](Fmt f, uint32_t rows, uint32_t k) {
    std::uniform_real_distribution<float> d(0.0005f, 0.004f);
    const std::vector<uint8_t> native = makeNative(f, rows, k, local, [&] { return f2h(d(local)); });
    return planeSegment(backend, f, repack(f, native, rows, k, nullptr), rows, k);
  };
  std::vector<float> router(uint64_t{E} * H, 0.0f), sharedGate(H, 0.0f), bias(E, 0.0f);
  for (uint32_t e = 0; e < E; ++e) router[uint64_t{e} * H + e] = 4.0f;
  const QuantizedSegment routerSegment = floatSegment(backend, router, E, H);
  const QuantizedSegment sharedGateSegment =
      a1b ? QuantizedSegment{} : floatSegment(backend, sharedGate, 1, H);
  const QuantizedSegment biasSegment =
      a1b ? floatSegment(backend, bias, 1, E) : QuantizedSegment{};
  MoeWeights gguf;
  gguf = BlockMoeWeights{routerSegment, sharedGateSegment,
                             {planes(gateUpFormat, E * I, H), planes(sharedFormat, I, H)},
                             {planes(gateUpFormat, E * I, H), planes(sharedFormat, I, H)},
                             {planes(downFormat, E * H, I), planes(sharedFormat, H, I)},
                             biasSegment};
  // Rows route to 8 of a pool of 24 experts per lane of 8 rows (decode) or of all 256 (prefill).
  const auto input = [&](uint32_t rows, uint32_t pool) {
    std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
    std::vector<float> x(uint64_t{rows} * H);
    std::vector<uint32_t> experts(E);
    std::iota(experts.begin(), experts.end(), 0u);
    for (uint32_t r = 0; r < rows; ++r) {
      if (r % 8 == 0) std::shuffle(experts.begin(), experts.end(), local);
      std::vector<uint32_t> candidates(experts.begin(), experts.begin() + pool);
      std::shuffle(candidates.begin(), candidates.end(), local);
      for (uint32_t k = 0; k < H; ++k) x[uint64_t{r} * H + k] = bf16(k < E ? 0.05f * unit(local) : unit(local));
      for (uint32_t rank = 0; rank < TOPK; ++rank) x[uint64_t{r} * H + candidates[rank]] = bf16(1.0f - 0.05f * rank);
    }
    return x;
  };
  MoeBuffers b;
  b.residual = zeros(backend, uint64_t{kRowsMax} * H * 2, "residual");
  b.output = zeros(backend, uint64_t{kRowsMax} * H * 2, "output");
  const richengine::ops::ExecutionPlans plans(backend.capabilities());
  const auto time = [&](const MoeWeights &weights, const MoePlan &plan) {
    allocate(backend, b, plan);
    CommandGraph graph;
    MoE::add(graph, b, weights, plan);
    std::vector<double> samples;
    for (uint32_t i = 0; i < rounds + 1; ++i) {
      const double seconds = backend.submitCommand(graph.dispatches()).gpuSeconds;
      if (i) samples.push_back(seconds * 1e3);   // the first round warms the pipelines
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
  };
  const MoeGgufTile device = richengine::ops::moeGgufTile(
      richengine::ops::DevicePolicy::of(backend.capabilities()), ggufShape);
  const MoeGgufTile other = device == MoeGgufTile::Register ? MoeGgufTile::Staged : MoeGgufTile::Register;
  printf("%s, GPU family %u, %u cores: median GPU ms per MoE layer of %u rounds\n",
         backend.capabilities().deviceName.c_str(), backend.capabilities().appleGpuFamily,
         backend.capabilities().gpuCoreCount, rounds);
  if (int8) {
    // The staged prefill passes against the int8 ones: forced off and on at
    // the same plan, as upstream's `int8` mode A/Bs them. The "plan" column
    // asks the real gate — moeConfig over a shape whose int8Experts every
    // resident AF block sets (TargetModelGeometry::moeInt8Experts).
    const auto passes = [&](uint32_t rows, bool on) {
      MoeConfig config = plans.moePrefill(ggufShape, rows).configuration();
      config.int8Passes = on;
      return MoE::prefillPlan(ggufShape, rows, config);
    };
    MoeShape int8Shape = ggufShape;
    int8Shape.int8Experts = true;
    // Warm the clocks as the default mode's 2048-row chunk does.
    b.input = bfloatBuffer(backend, input(kRowsMax, E), "input");
    static_cast<void>(time(gguf, passes(kRowsMax, false)));
    printf("  rows | routes per expert | rows per routed expert | staged ms | int8 ms | int8/staged | tile | "
           "plan\n");
    for (const uint32_t rows : {64u, 96u, 128u, 160u, 192u, 256u, 320u, 384u, 512u, 768u, 1024u, 2048u}) {
      const MoePlan stagedPlan = passes(rows, false), int8Plan = passes(rows, true);
      b.input = bfloatBuffer(backend, input(rows, E), "input");
      const double staged = time(gguf, stagedPlan), quantized = time(gguf, int8Plan),
                   staged2 = time(gguf, stagedPlan), quantized2 = time(gguf, int8Plan);
      const double before = (staged + staged2) / 2, after = (quantized + quantized2) / 2;
      // The rows each routed expert of this chunk's routes took on average:
      // its routes over the distinct experts the last run selected (the
      // routes are the staged and int8 plans' alike — the same input rows).
      const auto *selected = static_cast<const uint32_t *>(b.scratch.selectedExperts.contents());
      std::vector<uint8_t> seen(E, 0);
      uint32_t distinct = 0;
      for (uint64_t route = 0; route < uint64_t{rows} * ggufShape.routesPerToken(); ++route)
        if (selected[route] < E && !seen[selected[route]]++) ++distinct;
      const double perExpert = double(rows) * TOPK / std::max(distinct, 1u);
      printf("  %4u | %5.2f | %6.2f | %.4f | %.4f | %.3f | M%u | %s\n", rows, double(rows) * TOPK / E,
             perExpert, before, after, after / before,
             uint32_t(stagedPlan.configuration().expertTile),
             plans.moePrefill(int8Shape, rows).configuration().int8Passes ? "int8" : "staged");
    }
    return 0;
  }
  // Prefill chunks first: they also bring the GPU clocks up for the short decode layers.
  b.input = bfloatBuffer(backend, input(kRowsMax, E), "input");
  const double affinePrefill = time(affine, plans.moePrefill(affineShape, kRowsMax));
  printf("  prefill %u rows: affine %.3f  gguf %.3f\n", kRowsMax, affinePrefill,
         time(gguf, plans.moePrefill(ggufShape, kRowsMax)));
  for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
    b.input = bfloatBuffer(backend, input(lanes * 8, a1b ? E : 24), "input");
    MoeConfig config = plans.moeDecode(ggufShape, lanes, 8).configuration();
    config.ggufTile = other;
    if (other == MoeGgufTile::Register) config.expertTile = richengine::ops::MoeExpertTile::M8;
    const double a = time(affine, plans.moeDecode(affineShape, lanes, 8));
    const double g = time(gguf, plans.moeDecode(ggufShape, lanes, 8));
    const double o = time(gguf, MoE::decodePlan(ggufShape, lanes, 8, config));
    const double ap = time(affine, plans.moePrefill(affineShape, lanes * 8));
    const double gp = time(gguf, plans.moePrefill(ggufShape, lanes * 8));
    printf("  decode B%u: affine %.3f  gguf %s %.3f  gguf %s %.3f  | prefill chunk of %u rows: affine %.3f  "
           "gguf %.3f\n",
           lanes, a, device == MoeGgufTile::Register ? "register" : "staged", g,
           other == MoeGgufTile::Register ? "register" : "staged", o, lanes * 8, ap, gp);
  }
  for (const uint32_t rows : {64u, 128u, 256u, 512u}) {
    b.input = bfloatBuffer(backend, input(rows, E), "input");
    const double ap = time(affine, plans.moePrefill(affineShape, rows));
    const double gp = time(gguf, plans.moePrefill(ggufShape, rows));
    printf("  prefill chunk of %u rows: affine %.3f  gguf %.3f\n", rows, ap, gp);
  }
  // Where the time goes: every dispatch replayed as its own command.
  for (const auto &[label, weights, plan] :
       {std::tuple{"affine prefill", &affine, plans.moePrefill(affineShape, kRowsMax)},
        std::tuple{"gguf prefill", &gguf, plans.moePrefill(ggufShape, kRowsMax)},
        std::tuple{"affine B1", &affine, plans.moeDecode(affineShape, 1, 8)},
        std::tuple{"gguf B1", &gguf, plans.moeDecode(ggufShape, 1, 8)},
        std::tuple{"affine B4", &affine, plans.moeDecode(affineShape, 4, 8)},
        std::tuple{"gguf B4", &gguf, plans.moeDecode(ggufShape, 4, 8)}}) {
    b.input = bfloatBuffer(backend, input(plan.rows(), plan.rows() == kRowsMax || a1b ? E : 24), "input");
    allocate(backend, b, plan);
    CommandGraph graph;
    MoE::add(graph, b, *weights, plan);
    std::map<std::string, double> spent;
    for (uint32_t i = 0; i < rounds; ++i)
      for (const auto &[name, seconds] : richengine::benchmark::replayDispatches(backend, graph.dispatches()))
        spent[name] += seconds * 1e3 / rounds;
    printf("  %s:", label);
    for (const auto &[name, ms] : spent) printf(" %s %.3f", name.c_str(), ms);
    printf("\n");
  }
  return 0;
}

} // namespace

int main(int argc, const char *argv[]) {
  if (argc > 2 && std::string_view(argv[2]) == "bolt") {
    if (argc > 4) {
      std::cerr << "usage: gguf-moe-benchmark <metallib> bolt [rounds]\n";
      return 2;
    }
    setenv("RICHENGINE_MOE_INDIRECT_DISPATCH", "1", 1);
    @autoreleasepool {
      try {
        MetalBackend backend(argv[1]);
        return boltTiming(backend, argc == 4 ? std::stoul(argv[3]) : 20);
      } catch (const std::exception &error) {
        std::cerr << "gguf-moe-benchmark: " << error.what() << '\n';
        return 1;
      }
    }
  }
  // `int8` runs the opt-in RICHENGINE_PREFILL_FAST_INT8 passes, whose plan
  // gate reads modelPolicy().prefillFastInt8 — arm it before anything reads
// tuning() or modelPolicy(), as gguf-prefill-int8's binary does.
  const bool int8 = argc > 2 && std::string_view(argv[2]) == "int8";
  if (int8) setenv("RICHENGINE_PREFILL_FAST_INT8", "1", 1);
  @autoreleasepool {
    const bool a1b = argc > 1 && std::string_view(argv[argc - 1]) == "a1b";
    std::vector<const char *> args(argv, argv + argc - (a1b ? 1 : 0));
    if (int8) args.erase(args.begin() + 2);
    const size_t count = args.size();
    // The int8 passes take the MLX affine formats alone
    // (gguf_prefill_int8_format): af4 throughout by default.
    const Fmt gateUp = count > 3 ? fmtNamed(args[3]) : int8 ? AF4 : Q4K,
              down = count > 4 ? fmtNamed(args[4]) : int8 ? AF4 : Q5K,
              shared = count > 5 ? fmtNamed(args[5]) : int8 ? AF4 : Q80;
    if (count < 2 || count > 6 || gateUp == FMT_COUNT || down == FMT_COUNT || shared == FMT_COUNT) {
      std::cerr << "usage: gguf-moe-benchmark <metallib> [int8] [rounds] [gate/up format] [down format] "
                   "[shared format] [a1b]\n";
      return 2;
    }
    try {
      MetalBackend backend(args[1]);
      return timing(backend, count > 2 ? std::stoul(args[2]) : 20, int8, gateUp, down, shared, a1b);
    } catch (const std::exception &error) {
      std::cerr << "gguf-moe-benchmark: " << error.what() << '\n';
      return 1;
    }
  }
}
