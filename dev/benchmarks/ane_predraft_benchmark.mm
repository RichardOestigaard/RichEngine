// A/B: the predraft predictor's transport (docs/ANE_EXPERIMENTS.md) —
// CoreML's predict() path of MLMultiArray allocation, feature provider and
// output copies, against the same network as an in-memory MIL program
// evaluated over preallocated IOSurfaces.
//
// The model stands in for RICHENGINE_ANE_PREDRAFT's contract (docs/
// ANE_DRAFTING.md): inputs anchor, position, retained [4] int32 and hidden
// [4,8,2048] fp16, output proposals [4,7]. Both sides run two matmuls with a
// gelu between, 34 MFLOP — small enough that transport dominates.
//   ane-predraft-benchmark <metallib> <model.mlmodel|mlpackage|mlmodelc> [rounds]
#include "AwakeClock.hpp"
#include "ane/Handoff.hpp"
#include "ane/Program.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "model/AnePredictor.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace {

using richengine::AwakeClock;
using richengine::metal::CommandGraph;
using richengine::metal::MetalBackend;
using richengine::model::AneDType;
using richengine::model::AnePredictor;
using richengine::model::AneTensor;
using namespace richengine;
using Element = ane::Surface::Element;

// C: CoreAI's public framework through the @_cdecl shim (same .aimodel
// contract); run() copies into preallocated NDArrays, encode() binds
// caller-filled MTLBuffers. Both specialized for the Neural Engine.
extern "C" int coreai_shim_load(const char *path);
extern "C" int coreai_shim_eval_run(const void *hidden, const void *ids, void *out);
extern "C" int coreai_shim_eval_encode(const void *hidden, const void *ids, void *out);

constexpr uint32_t kLanes = 4, kRows = 8, kHidden = 2048, kMid = 256, kProposals = 7;
constexpr auto kHandoffBound = std::chrono::seconds(2);
constexpr auto kProgramLimit = std::chrono::seconds(120);

// predraft<ios18>(x [1,1,32,2048] fp16, w1 [1,1,2048,256], w2 [1,1,256,224]):
// hidden @ W1, gelu, @ W2, slice [4,7] out — the same op count as the CoreML
// model. The three int32 ids are not part of what the A/B prices (transport),
// so this side binds only what the matmuls need.
std::string predraftMil(const ane::Surface &hidden, const ane::Surface &w1s, const ane::Surface &w2s,
                        const ane::Surface &output) {
  const auto param = [](const ane::Surface &s, uint32_t rows, uint32_t width, const char *name) {
    return s.bufferType(rows, width) + " " + name;
  };
  const auto input = [](uint32_t rows, uint32_t width, const char *name) {
    return "        tensor<fp16, [1, 1, " + std::to_string(rows) + ", " + std::to_string(width) + "]> " + name +
           "_t = tensor_buffer_to_tensor<ios17>(input = " + name + ");\n";
  };
  // Weights bake transposed: ANE's matmul takes no transpose flags, so the
  // token-major surface multiplies W^T directly (x[32,2048] @ w1[2048,256]).
  std::string body = input(32, 2048, "x") + input(2048, 256, "w1") + input(256, 224, "w2");
  const auto f16 = [&](const char *name, uint32_t height, uint32_t width, const std::string &expression) {
    body += "        tensor<fp16, [1, 1, " + std::to_string(height) + ", " + std::to_string(width) + "]> " + name +
            " = " + expression + ";\n";
  };

  f16("a", 32, 256, "matmul(transpose_x = bool(false), transpose_y = bool(false), x = x_t, y = w1_t)");
  // gelu ≈ x/2 (1 + tanh(0.7978845608 (x + 0.044715 x^3))), each op a named
  // tensor — the ANE compiler rejects nested expressions.
  f16("a2", 32, 256, "mul(x = a, y = a)");
  f16("a3", 32, 256, "mul(x = a2, y = a)");
  f16("a4", 32, 256, "mul(x = a3, y = fp16(0x1.222p-4))");
  f16("a5", 32, 256, "add(x = a, y = a4)");
  f16("ai", 32, 256, "mul(x = a5, y = fp16(0x1.98p-1))");
  f16("at", 32, 256, "tanh(x = ai)");
  f16("ap", 32, 256, "add(x = at, y = fp16(0x1p0))");
  f16("ah", 32, 256, "mul(x = a, y = fp16(0x1p-1))");
  f16("g", 32, 256, "mul(x = ah, y = ap)");
  f16("lg", 32, 224, "matmul(transpose_x = bool(false), transpose_y = bool(false), x = g, y = w2_t)");

  f16("ls", 4, 7, "slice_by_size(x = lg, begin = tensor<int32, [4]>([0, 0, 0, 0]), size = tensor<int32, [4]>([1, 1, "
                "4, 7]))");
  body += "        " + output.bufferType(4, 7) +
          " y = tensor_to_tensor_buffer<ios17>(input = ls, interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), "
          "strides = tensor<int64, [4]>(" +
          output.strides(4) + "));\n";
  return "    func predraft<ios18>(" + param(hidden, 32, 2048, "x") + ", " + param(w1s, 2048, 256, "w1") + ", " +
         param(w2s, 256, 224, "w2") + ") {\n" + body + "    } -> (y);\n";
}

double median(std::vector<double> &values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

int run(MetalBackend &backend, std::string_view modelPath, uint32_t rounds) {
  if (const std::optional<std::string> why = ane::unavailable()) {
    std::cerr << "ane-predraft-benchmark: Neural Engine interface unavailable: " << *why << '\n';
    return 1;
  }

  // ---------- A: the production CoreML path ----------
  std::string error;
  std::unique_ptr<AnePredictor> predictor = AnePredictor::load(modelPath, error);
  if (!predictor) {
    std::cerr << "ane-predraft-benchmark: " << error << '\n';
    return 1;
  }
  std::mt19937 generator(20261220);
  std::array<int32_t, kLanes> ids{};
  for (int32_t &id : ids) id = int32_t(generator() % 128000);
  std::vector<_Float16> hiddenData(size_t{kLanes} * kRows * kHidden);
  for (_Float16 &value : hiddenData)
    value = _Float16(std::uniform_real_distribution<float>(-0.5f, 0.5f)(generator));
  std::vector<int32_t> proposals(size_t{kLanes} * kProposals, -1);
  AneTensor anchorIn{"anchor", AneDType::Int32, {kLanes}, ids.data()};
  AneTensor positionIn{"position", AneDType::Int32, {kLanes}, ids.data()};
  AneTensor hiddenIn{"hidden", AneDType::Float16, {kLanes, kRows, kHidden}, hiddenData.data()};
  AneTensor retainedIn{"retained", AneDType::Int32, {kLanes}, ids.data()};
  AneTensor out{"proposals", AneDType::Int32, {kLanes, kProposals}, proposals.data()};
  const std::array<AneTensor, 4> inputs{anchorIn, positionIn, hiddenIn, retainedIn};
  std::array<AneTensor, 1> outputs{out};

  const auto predict = [&] {
    if (!predictor->predict(std::span<const AneTensor>(inputs), std::span<AneTensor>(outputs), error))
      throw std::runtime_error("predict: " + error);
  };
  for (uint32_t i = 0; i < 5; ++i) predict();
  std::vector<double> coremlMs;
  for (uint32_t round = 0; round < rounds; ++round) {
    const auto start = AwakeClock::now();
    predict();
    coremlMs.push_back(std::chrono::duration<double, std::milli>(AwakeClock::now() - start).count());
  }

  // ---------- B: in-memory MIL over surfaces ----------
  const auto surface = [&](uint32_t rows, uint32_t width) {
    return ane::Surface::create(backend, rows, width, Element::Float16);
  };
  ane::Surface hidden = surface(32, 2048);

  ane::Surface output = surface(4, 7);

  // Weights ride surfaces, transposed — a const() BLOBFILE tensor as a matmul
  // operand made the ANE compiler reject the program (it accepts const() only
  // as conv weight, as the dense split's rotation shows).
  std::vector<_Float16> w1(uint64_t{kMid} * kHidden), w2(uint64_t{224} * kMid);
  for (_Float16 &v : w1) v = _Float16(std::normal_distribution<float>(0.f, 0.01f)(generator));
  for (_Float16 &v : w2) v = _Float16(std::normal_distribution<float>(0.f, 0.05f)(generator));
  ane::Surface w1Surface = surface(2048, 256), w2Surface = surface(256, 224);
  {
    std::memcpy(w1Surface.buffer.contents(), w1.data(), w1.size() * 2);
    std::memcpy(w2Surface.buffer.contents(), w2.data(), w2.size() * 2);
  }
  const std::string mil = "program(1.3)\n{\n" +
                          predraftMil(hidden, w1Surface, w2Surface, output) + "}\n";
  if (std::getenv("ANE_PREDRAFT_DUMP")) std::fprintf(stderr, "%s\n", mil.c_str());
  // No const() weights — the blob is the empty header only.
  const std::vector<uint8_t> blob = ane::constantBlob({});
  ane::Program program(mil, blob, {kProgramLimit, [] { return false; }});
  const uint32_t procedure = program.procedure("predraft");
  const std::vector<std::string> &names = program.inputs(procedure);
  std::vector<ane::Surface> bound;
  for (const std::string &name : names) {
    const ane::Surface *found = nullptr;
    if (name == "x") found = &hidden;
    if (name == "w1") found = &w1Surface;
    if (name == "w2") found = &w2Surface;
    if (!found) throw std::runtime_error("unexpected input " + name);
    bound.push_back(*found);
  }
  const ane::Program::Binding binding = program.bind(procedure, bound, output);
  ane::Handoff handoff(backend, kHandoffBound);
  // A command needs a dispatch; one residency_kick thread satisfies it.
  const metal::MetalBuffer kickTarget = backend.allocateBuffer(4, metal::BufferStorage::Shared, "kick");

  // hidden: token-major [32 rows x 2048] = 128KB contiguous (rowBytes already
  // aligned); the id surfaces hold 4 halves at 64B stride.
  const auto fill = [&] {
    _Float16 *rows = static_cast<_Float16 *>(hidden.buffer.contents());
    const uint32_t stride = hidden.strideBytes / 2;
    for (uint32_t t = 0; t < 32; ++t)
      std::memcpy(rows + uint64_t{t} * stride, hiddenData.data() + uint64_t{t} * kHidden, kHidden * 2);
  };
  const auto eval = [&] {
    CommandGraph graph;
    graph.add("residency_kick", {kickTarget}, {1, 1, 1}, {1, 1, 1});
    const uint64_t ready = handoff.next(), done = handoff.next();
    graph.signal(handoff.event(), ready);
    graph.wait(handoff.event(), done);
    handoff.queue(ready, done,
                  [&](const metal::SharedEvent &event, uint64_t wait, uint64_t signal, ane::Handoff::Report report) {
                    program.enqueue(binding, event, wait, signal, std::move(report));
                  });
    static_cast<void>(backend.submitCommandAsync(graph.command()).wait());
    if (!handoff.finish()) throw std::runtime_error("ANE eval failed (" + handoff.reason() + ")");
  };
  for (uint32_t i = 0; i < 5; ++i) {
    fill();
    eval();
  }
  std::vector<double> writeMs, milMs;
  for (uint32_t round = 0; round < rounds; ++round) {
    const auto start = AwakeClock::now();
    fill();
    const auto wrote = AwakeClock::now();
    eval();
    const auto finished = AwakeClock::now();
    writeMs.push_back(std::chrono::duration<double, std::milli>(wrote - start).count());
    milMs.push_back(std::chrono::duration<double, std::milli>(finished - wrote).count());
  }

  // ---------- C: CoreAI (public framework, NE specialization) ----------
  // Same contract as A, model authored through coreai-torch to .aimodel.
  // ANE_PREDRAFT_AIMODEL points at the asset; skipped when unset.
  std::vector<double> runMs, encodeMs;
  if (const char *aiModel = std::getenv("ANE_PREDRAFT_AIMODEL")) {
    if (!coreai_shim_load(aiModel))
      throw std::runtime_error("coreai_shim_load failed for " + std::string(aiModel));
    // The shim's ids are fp32 (the .aimodel signature); hidden is the same
    // fp16 rows. out is fp16 [4,7] — CoreML's fp32 [4,7,1] copy size is
    // comparable enough for a transport A/B.
    std::array<float, kLanes * 3> ids32{};
    for (uint32_t i = 0; i < kLanes; ++i) {
      ids32[i] = ids32[i + kLanes] = ids32[i + 2 * kLanes] = static_cast<float>(ids[i]);
    }
    std::array<_Float16, kLanes * kProposals> out16{};
    const auto coreai = [&](int (*fn)(const void *, const void *, void *)) {
      if (!fn(hiddenData.data(), ids32.data(), out16.data()))
        throw std::runtime_error("coreai eval failed");
    };
    for (uint32_t i = 0; i < 5; ++i) coreai(coreai_shim_eval_run);
    for (uint32_t round = 0; round < rounds; ++round) {
      const auto start = AwakeClock::now();
      coreai(coreai_shim_eval_run);
      runMs.push_back(std::chrono::duration<double, std::milli>(AwakeClock::now() - start).count());
    }
    for (uint32_t i = 0; i < 5; ++i) coreai(coreai_shim_eval_encode);
    for (uint32_t round = 0; round < rounds; ++round) {
      const auto start = AwakeClock::now();
      coreai(coreai_shim_eval_encode);
      encodeMs.push_back(std::chrono::duration<double, std::milli>(AwakeClock::now() - start).count());
    }
  }

  const double a = median(coremlMs), write = median(writeMs), evalMs = median(milMs);
  std::printf("predraft transport A/B/C (%u rounds, hidden 4x8x2048 fp16, 34 MFLOP model)\n\n", rounds);
  std::printf("  %-34s %9.3f ms/call\n", "A: CoreML predict() (MLMultiArray)", a);
  std::printf("  %-34s %9.3f ms/call\n", "B: MIL surfaces + eval", write + evalMs);
  std::printf("  %-34s %9.3f\n", "    surface writes", write);
  std::printf("  %-34s %9.3f\n", "    eval round-trip", evalMs);
  if (!runMs.empty()) {
    const double r = median(runMs), e = median(encodeMs);
    std::printf("  %-34s %9.3f ms/call\n", "C1: CoreAI run() (NDArray)", r);
    std::printf("  %-34s %9.3f ms/call\n", "C2: CoreAI encode() (MTLBuffer)", e);
    std::printf("\nB/A = %.2fx, C1/A = %.2fx, C2/A = %.2fx\n", (write + evalMs) / a, r / a, e / a);
  } else {
    std::printf("\nB/A = %.2fx; saving %.3f ms per call\n", (write + evalMs) / a, a - write - evalMs);
  }
  return 0;
}

} // namespace

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    if (argc < 3 || argc > 4) {
      std::cerr << "usage: ane-predraft-benchmark <metallib> <model> [rounds]\n";
      return 2;
    }
    try {
      MetalBackend backend(argv[1]);
      return run(backend, argv[2], argc > 3 ? std::stoul(argv[3]) : 30);
    } catch (const std::exception &error) {
      std::cerr << "ane-predraft-benchmark: " << error.what() << '\n';
      return 1;
    }
  }
}
