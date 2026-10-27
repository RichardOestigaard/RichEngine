#include "ops/CoreAiFfn.hpp"

#include "Checked.hpp"
#include "StderrLine.hpp"
#include "metal/abi/AneFfn.h"
#include "metal/abi/QuantFormat.h"
#include "ops/BufferExtent.hpp"
#include "ops/CoreAiBridge.h"
#include "ops/KernelNames.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <iomanip>
#include <mach-o/dyld.h>
#include <optional>
#include <random>
#include <semaphore>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace richengine::ops {
namespace {

// The channels a split moves in, as ops::AneFfn's: whole blocks of the
// intermediate rotation for the ANE, whole 256-row tiles of the weight
// planes for the GPU.
constexpr uint32_t kChannelUnit = std::max(QUANT_TILE_ROWS, ANE_FFN_INTERMEDIATE_BLOCK);
constexpr uint32_t kQuantGroup = 64;
// How long an evaluation may run once the GPU has raised its `ready`, as
// ops::AneFfn's (the Metal event-wait timeout there bounds it too).
constexpr auto kHandoffBound = std::chrono::seconds(2);

struct ShimApi {
  decltype(&coreai_ffn_open) open = nullptr;
  decltype(&coreai_ffn_start) start = nullptr;
  decltype(&coreai_ffn_close) close = nullptr;
};

// The shim beside the executable, or at RICHENGINE_COREAI_SHIM. Where the
// process lacks it — a Mac without the framework, a build without the
// dylib — the split is simply unavailable.
const ShimApi *shim() {
  static const std::optional<ShimApi> api = []() -> std::optional<ShimApi> {
    std::vector<std::string> candidates;
    if (const char *env = std::getenv("RICHENGINE_COREAI_SHIM")) candidates.emplace_back(env);
    uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    std::string exe(size, '\0');
    if (_NSGetExecutablePath(exe.data(), &size) == 0)
      candidates.emplace_back(exe.substr(0, exe.find_last_of('/') + 1) + "librichengine_coreai.dylib");
    candidates.emplace_back("librichengine_coreai.dylib");
    for (const std::string &candidate : candidates) {
      void *library = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
      if (!library) continue;
      ShimApi loaded{reinterpret_cast<decltype(&coreai_ffn_open)>(dlsym(library, "coreai_ffn_open")),
                     reinterpret_cast<decltype(&coreai_ffn_start)>(dlsym(library, "coreai_ffn_start")),
                     reinterpret_cast<decltype(&coreai_ffn_close)>(dlsym(library, "coreai_ffn_close"))};
      if (loaded.open && loaded.start && loaded.close) return loaded;
      dlclose(library);
    }
    return std::nullopt;
  }();
  return api ? &*api : nullptr;
}

// The .aimodel directory of the split's functions
// (dev/tools/gen_coreai_ffn.py writes one).
const char *assetPath() { return std::getenv("RICHENGINE_COREAI_ASSET"); }

// The GPU's channels of `intermediate` with the ANE taking `aneUnits`, as
// ops::AneFfn's.
uint32_t gpuChannels(uint32_t intermediate, uint32_t aneUnits) {
  if (!aneUnits || aneUnits >= intermediate / kChannelUnit)
    throw std::invalid_argument("ANE FFN split leaves the GPU or the ANE no channels");
  return intermediate - aneUnits * kChannelUnit;
}

enum class Rotation : uint8_t { Inputs, Intermediate };
uint32_t blockOf(Rotation rotation) {
  return rotation == Rotation::Inputs ? ANE_FFN_INPUT_BLOCK : ANE_FFN_INTERMEDIATE_BLOCK;
}
std::string kernel(const char *name, const char *suffix, Rotation rotation) {
  return std::string(name) + suffix + (rotation == Rotation::Inputs ? "_inputs" : "_intermediate");
}

// What CoreAiFfn::verify fills the model's output with before a check: an
// fp16 NaN, which the join flags where the evaluation leaves it.
constexpr uint16_t kHalfNaN = 0x7e00;
// The most the split's output may differ from the GPU's alone, as the RMS
// of their difference relative to the RMS of what the ANE adds over the
// GPU's part: AneFfn's bound, which int8 weights at fp16 activations stay
// well within (verify measures about 1%).
constexpr double kMaximumAneError = 0.10;

// The leading `count` values of the CPU-visible bf16 `buffer`.
std::vector<float> bf16Values(const metal::MetalBuffer &buffer, uint64_t count) {
  const auto *bits = static_cast<const uint16_t *>(buffer.contents());
  if (!bits || buffer.sizeBytes() < count * sizeof(uint16_t))
    throw std::invalid_argument("CoreAI FFN check reads rows the CPU cannot");
  std::vector<float> values(count);
  for (uint64_t i = 0; i < count; ++i) values[i] = std::bit_cast<float>(uint32_t{bits[i]} << 16);
  return values;
}

} // namespace

const char *CoreAiFfn::unsupported(std::span<const SwiGluProjections> layers) {
  if (layers.empty()) return "CoreAI FFN split has no layers";
  const uint32_t hidden = layers.front().gate->inputSize, intermediate = layers.front().gate->outputSize;
  if (hidden % kQuantGroup) return "CoreAI FFN split needs a hidden size of whole quantization groups";
  if (intermediate % kChannelUnit || intermediate < 2 * kChannelUnit)
    return "CoreAI FFN split needs two or more whole channel units of intermediate channels";
  for (const SwiGluProjections &layer : layers) {
    for (const Projection *projection : {layer.gate, layer.up, layer.down})
      if (!projection->takesPlaneViews())
        return "CoreAI FFN split needs affine Q4 projections or unrotated quantized GGUF tensors";
    if (layer.gate->outputSize != intermediate || layer.gate->inputSize != hidden ||
        layer.up->outputSize != intermediate || layer.up->inputSize != hidden || layer.down->outputSize != hidden ||
        layer.down->inputSize != intermediate)
      return "CoreAI FFN split layers differ in shape";
  }
  return nullptr;
}

uint32_t CoreAiFfn::units(std::span<const SwiGluProjections> layers) {
  if (const char *reason = unsupported(layers)) throw std::invalid_argument(reason);
  return layers.front().gate->outputSize / kChannelUnit;
}

std::optional<std::string> CoreAiFfn::unavailable() {
  if (!shim()) return "the CoreAI shim librichengine_coreai.dylib does not load";
  if (!assetPath()) return "RICHENGINE_COREAI_ASSET is not set";
  return std::nullopt;
}

CoreAiFfn::Planes::Planes(const Projection &projection) {
  if (projection.layout() == WeightLayout::Affine64) {
    requireAffineProjection(projection, {projection.outputSize, projection.inputSize});
    const AffineWeights &weights = projection.affine();
    groups = projection.inputSize / kQuantGroup;
    buffers = {weights.weights, weights.scales, weights.biases};
    return;
  }
  const QuantizedSegment &segment = projection.blocks().segments.front();
  requireSegmentPlanes(segment, "CoreAI FFN projection");
  groups = projection.inputSize / 32;
  format = segment.formatId;
  suffix = "_gguf";
  buffers = {segment.plane0, segment.plane1Slot(), segment.meta};
}

CoreAiFfn::Shape CoreAiFfn::shapeOf(std::span<const SwiGluProjections> layers, uint32_t aneUnits) {
  if (const char *reason = unsupported(layers)) throw std::invalid_argument(reason);
  Shape shape;
  shape.hidden = layers.front().gate->inputSize;
  shape.intermediate = layers.front().gate->outputSize;
  shape.layers = static_cast<uint32_t>(layers.size());
  shape.gpu = gpuChannels(shape.intermediate, aneUnits);
  shape.ane = shape.intermediate - shape.gpu;
  return shape;
}

template <class MakeBuffer>
CoreAiFfn::Memory CoreAiFfn::allocate(const Shape &shape, MakeBuffer &&buffer) {
  Memory memory;
  memory.rowScales = buffer(uint64_t{shape.layers} * (2 * shape.ane + shape.hidden) * sizeof(_Float16),
                            "coreai ffn row scales");
  memory.x = buffer(uint64_t{kMaximumRows} * shape.hidden * sizeof(_Float16), "coreai ffn x");
  memory.y = buffer(uint64_t{kMaximumRows} * shape.hidden * sizeof(_Float16), "coreai ffn y");
  memory.status = buffer(sizeof(uint32_t), "coreai ffn status");
  for (Weights &set : memory.sets) {
    set.gateQ = buffer(uint64_t{shape.ane} * shape.hidden, "coreai ffn gate weights");
    set.upQ = buffer(uint64_t{shape.ane} * shape.hidden, "coreai ffn up weights");
    set.downQ = buffer(uint64_t{shape.hidden} * shape.ane, "coreai ffn down weights");
    set.gateS = buffer(uint64_t{shape.ane} * sizeof(_Float16), "coreai ffn gate scales");
    set.upS = buffer(uint64_t{shape.ane} * sizeof(_Float16), "coreai ffn up scales");
    set.downS = buffer(uint64_t{shape.hidden} * sizeof(_Float16), "coreai ffn down scales");
  }
  return memory;
}

uint64_t CoreAiFfn::plannedBytes(std::span<const SwiGluProjections> layers, uint32_t aneUnits) {
  uint64_t bytes = 0;
  static_cast<void>(allocate(shapeOf(layers, aneUnits), [&](uint64_t size, const char *) {
                               bytes += alignUp(size);
                               return metal::MetalBuffer{};
                             }));
  return bytes;
}

void *CoreAiFfn::open() {
  const char *path = assetPath(), *error = nullptr;
  if (!path) throw std::runtime_error("RICHENGINE_COREAI_ASSET is not set");
  const std::string specific =
      "ffn_" + std::to_string(shape_.hidden) + "_" + std::to_string(shape_.ane);
  std::string failures;
  for (const char *name : {specific.c_str(), "ffn"}) {
    error = nullptr;
    if (void *function = shim()->open(path, name, backend_.nativeDevice(), handoff_.event().nativeHandle(),
                                      shape_.hidden, shape_.ane, &error))
      return function;
    if (!failures.empty()) failures += "; ";
    failures += std::string(name) + ": " + (error ? error : "an unknown error");
  }
  throw std::runtime_error("the CoreAI FFN function did not load: " + failures);
}

CoreAiFfn::CoreAiFfn(metal::MetalBackend &backend, std::span<const SwiGluProjections> layers, uint32_t aneUnits)
    : backend_(backend), linear_(backend.capabilities()), shape_(shapeOf(layers, aneUnits)),
      handoff_(backend, kHandoffBound) {
  if (!shim()) throw std::runtime_error("the CoreAI shim librichengine_coreai.dylib does not load");
  memory_ = allocate(shape_, [&](uint64_t bytes, const char *label) {
    metal::MetalBuffer buffer = backend_.allocateBuffer(bytes, metal::BufferStorage::Shared, label);
    allocatedBytes_ += buffer.allocatedBytes();
    return buffer;
  });
  *static_cast<uint32_t *>(memory_.status.contents()) = 0;
  for (const SwiGluProjections &source : layers)
    layers_.push_back({{Planes(*source.gate), Planes(*source.up), Planes(*source.down)},
                       source.gate->leadingRows(backend_, shape_.gpu),
                       source.up->leadingRows(backend_, shape_.gpu),
                       source.down->leadingInputs(shape_.gpu)});
  function_ = open();

  // Each row's shared int8 scale over the ANE's share of its inputs.
  metal::CommandGraph graph;
  for (uint32_t layer = 0; layer < layers_.size(); ++layer) {
    const auto add = [&](Matrix matrix, uint32_t row, uint32_t input, uint32_t width, uint32_t rows,
                         Rotation rotation) {
      const Planes &planes = layers_[layer].planes[static_cast<size_t>(matrix)];
      graph.add(kernel(kCoreAiFfnRowScale.data(), planes.suffix, rotation),
                {planes.buffers[0], planes.buffers[1], planes.buffers[2], rowScales(layer, matrix),
                 memory_.rowScales},
                AneFfnWeightParams{planes.groups, row, input, width, 0, 0, planes.format},
                {rows / ANE_FFN_WEIGHT_ROWS, 1, 1}, {ANE_FFN_WEIGHT_THREADS, 1, 1});
    };
    add(Matrix::Gate, shape_.gpu, 0, shape_.hidden, shape_.ane, Rotation::Inputs);
    add(Matrix::Up, shape_.gpu, 0, shape_.hidden, shape_.ane, Rotation::Inputs);
    add(Matrix::Down, 0, shape_.gpu, shape_.ane, shape_.hidden, Rotation::Intermediate);
  }
  static_cast<void>(backend_.submitCommandAsync(graph.dispatches()).wait());

  // A cold function specializes on first encode — over 2 s here, past the
  // handoff bound a verify job gets. Burn one job on the split's own buffers
  // now, while a slow open is expected: a wait of 0 is already satisfied
  // and a signal of 0 changes nothing on the shared event.
  const Weights &set = memory_.sets[0];
  const auto bind = [](const metal::MetalBuffer &buffer) {
    return coreai_ffn_buf{buffer.nativeHandle(), buffer.offsetBytes()};
  };
  const coreai_ffn_job warm{bind(memory_.x), bind(memory_.y), bind(set.gateQ), bind(set.gateS),
                            bind(set.upQ),   bind(set.upS),   bind(set.downQ), bind(set.downS),
                            0,               0,               kMinimumRows};
  std::binary_semaphore reported{0};
  int ok = 0;
  std::pair state{&reported, &ok};
  if (!shim()->start(function_, &warm,
                     [](void *ctx, int good) {
                       auto *state = static_cast<std::pair<std::binary_semaphore *, int *> *>(ctx);
                       *state->second = good;
                       state->first->release();
                     },
                     &state))
    throw std::runtime_error("the CoreAI function did not take the warm-up job");
  reported.acquire();
  if (!ok) throw std::runtime_error("the CoreAI function's warm-up evaluation failed");
}

CoreAiFfn::~CoreAiFfn() {
  // handoff_ (declared last) has waited for every report by the time its
  // destructor runs; the function the jobs encoded on goes here.
  if (function_) shim()->close(function_);
}

double CoreAiFfn::share() const { return static_cast<double>(shape_.ane) / shape_.intermediate; }

void CoreAiFfn::setMinimumRows(uint32_t rows) {
  if (rows < kMinimumRows || rows > kMaximumRows)
    throw std::invalid_argument("CoreAI FFN split takes chunks of " + std::to_string(kMinimumRows) + " rows or more");
  minimumRows_ = rows;
}

bool CoreAiFfn::splits(uint32_t rows) const {
  return rows >= minimumRows_ && rows <= kMaximumRows && !handoff_.retired();
}

metal::MetalBuffer CoreAiFfn::rowScales(uint32_t layer, Matrix matrix) const {
  const uint64_t perLayer = 2 * shape_.ane + shape_.hidden, part = static_cast<uint64_t>(matrix);
  return backend_.view(memory_.rowScales, (layer * perLayer + part * shape_.ane) * sizeof(_Float16),
                       uint64_t{matrix == Matrix::Down ? shape_.hidden : shape_.ane} * sizeof(_Float16));
}

// Layer `layer`'s int8 weights and value scales into staging set `set`,
// unrotated: gate's and up's ANE rows over the whole hidden, down's rows
// over its ANE inputs.
void CoreAiFfn::addWeights(metal::CommandGraph &graph, uint32_t layer, uint32_t set) const {
  const Layer &source = layers_.at(layer);
  const Weights &target = memory_.sets[set];
  const auto add = [&](Matrix matrix, const metal::MetalBuffer &output, const metal::MetalBuffer &scale,
                       uint32_t row, uint32_t input, uint32_t width, uint32_t rows, Rotation rotation) {
    const Planes &planes = source.planes[static_cast<size_t>(matrix)];
    requireBytes(output, uint64_t{rows} * width, "CoreAI FFN weight buffer");
    requireBytes(scale, uint64_t{rows} * sizeof(_Float16), "CoreAI FFN scale buffer");
    graph.add(kernel(kCoreAiFfnWeights.data(), planes.suffix, rotation),
              {planes.buffers[0], planes.buffers[1], planes.buffers[2], rowScales(layer, matrix), output, scale,
               memory_.rowScales},
              AneFfnWeightParams{planes.groups, row, input, width, width, 1, planes.format},
              {rows / ANE_FFN_WEIGHT_ROWS, width / blockOf(rotation), 1}, {ANE_FFN_WEIGHT_THREADS, 1, 1});
  };
  add(Matrix::Gate, target.gateQ, target.gateS, shape_.gpu, 0, shape_.hidden, shape_.ane, Rotation::Inputs);
  add(Matrix::Up, target.upQ, target.upS, shape_.gpu, 0, shape_.hidden, shape_.ane, Rotation::Inputs);
  add(Matrix::Down, target.downQ, target.downS, 0, shape_.gpu, shape_.ane, shape_.hidden, Rotation::Intermediate);
}

bool CoreAiFfn::begin() {
  if (std::exchange(unfinished_, false)) {
    handoff_.cancel("a command it split did not finish");
    warnStopped();
  }
  jobs_.clear();
  nextLayer_ = 0;
  return !handoff_.retired();
}

void CoreAiFfn::add(metal::CommandGraph &graph, uint32_t layer, const PrefillFfnBuffers &ffn,
                    metal::MetalBuffer residual, metal::MetalBuffer output, uint32_t rows) {
  encode(graph, layer, ffn, residual, output, rows, Parts::Both);
}

void CoreAiFfn::encode(metal::CommandGraph &graph, uint32_t layer, const PrefillFfnBuffers &ffn,
                       metal::MetalBuffer residual, metal::MetalBuffer output, uint32_t rows, Parts parts) {
  if (rows < 1 || rows > kMaximumRows)
    throw std::invalid_argument("CoreAI FFN split does not take a chunk of these rows");
  if (layer != nextLayer_ || layer >= layers_.size())
    throw std::logic_error("CoreAI FFN split adds layer " + std::to_string(layer) + ", not its next layer " +
                           std::to_string(nextLayer_) + " (begin() starts each command at layer 0)");
  const uint64_t hiddenBytes = uint64_t{rows} * shape_.hidden * sizeof(uint16_t);
  requireBytes(ffn.normalized, hiddenBytes, "CoreAI FFN input");
  requireBytes(output, hiddenBytes, "CoreAI FFN output");
  const Layer &current = layers_[layer];
  const uint32_t set = layer & 1;
  const bool ane = parts == Parts::Both;
  const uint64_t elements = uint64_t{rows} * shape_.hidden, groups = (elements + 255) / 256;
  // Each command stages layer 0's weights, then each layer the next one's.
  if (!layer) addWeights(graph, 0, 0);
  graph.add(std::string(kCoreAiFfnCast), {ffn.normalized, memory_.x}, CoreAiFfnParams{shape_.hidden, rows},
            {groups, 1, 1}, {256, 1, 1});
  const uint64_t ready = ane ? handoff_.next() : 0;
  if (ane) graph.signal(handoff_.event(), ready);
  linear_.addPrefillSwiGlu(graph, {&current.gate, &current.up, &current.down}, ffn, residual, output, rows);
  if (layer + 1 < layers_.size()) addWeights(graph, layer + 1, set ^ 1);
  const uint64_t done = ane ? handoff_.next() : 0;
  if (ane) graph.wait(handoff_.event(), done);
  graph.add(std::string(kCoreAiFfnJoin), {output, memory_.y, memory_.status},
            CoreAiFfnParams{shape_.hidden, rows}, {groups, 1, 1}, {256, 1, 1});
  if (ane) jobs_.push_back({set, rows, ready, done});
  ++nextLayer_;
}

metal::CommandTicket CoreAiFfn::commit(const metal::CommandGraph &graph, metal::CommandCompletion completion) {
  const std::vector<Job> jobs = std::exchange(jobs_, {});
  for (const Job &job : jobs) queue(job);
  unfinished_ = !jobs.empty();
  committedEvaluations_ = static_cast<uint32_t>(jobs.size());
  committedRows_ = jobs.empty() ? 0 : jobs.front().rows;
  const auto cancel = [&](const std::string &reason) {
    if (!std::exchange(unfinished_, false)) return;
    handoff_.cancel(reason);
    warnStopped();
  };
  try {
    return backend_.submitCommandAsync(graph.command(), std::move(completion));
  } catch (const std::exception &error) {
    cancel(std::string("its Metal command was not submitted: ") + error.what());
    throw;
  } catch (...) {
    cancel("its Metal command was not submitted");
    throw;
  }
}

bool CoreAiFfn::finish() {
  const std::optional<AwakeClock::duration> ran = completed();
  if (!ran) {
    warnStopped();
    return false;
  }
  if (!committedEvaluations_) return true;
  const double milliseconds = std::chrono::duration<double, std::milli>(*ran).count();
  ++served_.commands;
  served_.evaluations += committedEvaluations_;
  served_.milliseconds += milliseconds;
  if (std::string losing = breaker_.add(committedRows_, committedEvaluations_, milliseconds); !losing.empty()) {
    handoff_.retire(std::move(losing));
    warnStopped();
    if (lost_) std::exchange(lost_, {})();
  }
  return true;
}

std::optional<AwakeClock::duration> CoreAiFfn::completed() {
  unfinished_ = false;
  std::optional<AwakeClock::duration> ran = handoff_.finish();
  if (std::exchange(*static_cast<uint32_t *>(memory_.status.contents()), 0u) && ran) {
    handoff_.retire("the CoreAI function's output was not finite");
    ran.reset();
  }
  return ran;
}

bool CoreAiFfn::release() {
  if (unfinished_ || !jobs_.empty())
    throw std::logic_error("CoreAI FFN split released while a command it encoded is unfinished");
  if (released_ || (handoff_.retired() && !handoff_.idle())) return false;
  if (function_) {
    shim()->close(function_);
    function_ = nullptr;
  }
  released_ = true;
  logLine(dim("CoreAI FFN function unloaded while idle"));
  return true;
}

void CoreAiFfn::restore() noexcept {
  if (!released_ || handoff_.retired()) return;
  released_ = false;
  const auto start = AwakeClock::now();
  std::string failure = "an unknown exception";
  try {
    function_ = open();
    logLine(dim("CoreAI FFN function reloaded in "), std::fixed, std::setprecision(2),
            millisecondsSince(start) / 1000.0, dim(" s"));
    return;
  } catch (const std::exception &error) {
    failure = error.what();
  } catch (...) {
  }
  handoff_.retire("its function did not load again: " + failure);
  warnStopped();
}

void CoreAiFfn::warnStopped() {
  if (std::exchange(warned_, true)) return;
  logWarning("CoreAI FFN split stopped (", handoff_.reason(),
             "); the GPU runs the prefill FFN alone until the engine restarts.");
}

void CoreAiFfn::queue(const Job &job) {
  const Weights &set = memory_.sets[job.set];
  const auto bind = [](const metal::MetalBuffer &buffer) {
    return coreai_ffn_buf{buffer.nativeHandle(), buffer.offsetBytes()};
  };
  const coreai_ffn_job raw{bind(memory_.x), bind(memory_.y), bind(set.gateQ), bind(set.gateS), bind(set.upQ),
                           bind(set.upS), bind(set.downQ), bind(set.downS), 0, 0, job.rows};
  handoff_.queue(job.ready, job.done,
                 [this, raw](const metal::SharedEvent &, uint64_t wait, uint64_t signal,
                             ane::Handoff::Report report) {
                   coreai_ffn_job copy = raw;
                   copy.wait = wait;
                   copy.signal = signal;
                   auto *result = new ane::Handoff::Report(std::move(report));
                   if (!shim()->start(function_, &copy,
                                      [](void *ctx, int ok) {
                                        auto *done = static_cast<ane::Handoff::Report *>(ctx);
                                        (*done)(ok);
                                        delete done;
                                      },
                                      result)) {
                     delete result;
                     throw std::runtime_error("the CoreAI function did not take the job");
                   }
                 });
}

void CoreAiFfn::fillNormalized(const metal::MetalBuffer &normalized) {
  auto *bits = static_cast<uint16_t *>(normalized.contents());
  if (!bits) throw std::invalid_argument("CoreAI FFN check writes rows the CPU cannot");
  std::mt19937 random(20261001);
  std::uniform_real_distribution<float> uniform(-2.0f, 2.0f);
  for (uint64_t i = 0; i < normalized.sizeBytes() / sizeof(uint16_t); ++i)
    bits[i] = static_cast<uint16_t>(std::bit_cast<uint32_t>(uniform(random)) >> 16);
}

double CoreAiFfn::verify(std::span<const SwiGluProjections> layers, const PrefillFfnBuffers &ffn,
                         const std::array<metal::MetalBuffer, 2> &hidden) {
  if (layers.size() != layers_.size() || layers.front().gate->outputSize != shape_.intermediate)
    throw std::invalid_argument("CoreAI FFN split verified on other layers than its own");
  if (handoff_.retired()) throw std::runtime_error("CoreAI FFN split stopped (" + handoff_.reason() + ")");
  fillNormalized(ffn.normalized);
  // As AneFfn::verify: GPU alone, the split's GPU part over a y of zeros,
  // and the split, over two layers, with y filled with NaN before the
  // split's runs so that rows the function leaves unwritten show.
  enum class Way : uint8_t { Gpu, GpuPart, Split };
  const auto forward = [&](Way way, uint32_t count, uint32_t rows) {
    std::memset(hidden[0].contents(), 0, hidden[0].sizeBytes());
    if (way != Way::Gpu) {
      std::fill_n(static_cast<uint16_t *>(memory_.y.contents()), memory_.y.sizeBytes() / sizeof(uint16_t),
                  way == Way::Split ? kHalfNaN : 0);
      static_cast<void>(begin());
    }
    metal::CommandGraph graph;
    if (layers.front().gate->layout() == WeightLayout::Affine64)
      linear_.addPrefillSums(graph, ffn.normalized, ffn.sums, *layers.front().gate, rows);
    for (uint32_t layer = 0; layer < count; ++layer) {
      const metal::MetalBuffer &residual = hidden[layer & 1],
                               &output = hidden[(layer & 1) ^ 1];
      if (way == Way::Gpu)
        linear_.addPrefillSwiGlu(graph, layers[layer], ffn, residual, output, rows);
      else
        encode(graph, layer, ffn, residual, output, rows, way == Way::Split ? Parts::Both : Parts::Gpu);
    }
    if (way == Way::Split) {
      static_cast<void>(commit(graph, {}).wait());
      if (!completed())
        throw std::runtime_error("CoreAI FFN split of " + std::to_string(rows) + " rows failed (" + reason() +
                                 ")");
    } else {
      static_cast<void>(backend_.submitCommandAsync(graph.command()).wait());
    }
    return bf16Values(hidden[count & 1], uint64_t{rows} * shape_.hidden);
  };
  const uint32_t count = std::min<uint32_t>(2, shape_.layers);
  const std::vector<float> gpu = forward(Way::Gpu, count, kMaximumRows),
                           part = forward(Way::GpuPart, count, kMaximumRows);
  double worst = 0.0;
  for (const uint32_t rows : {kMinimumRows, kMaximumRows}) {
    const std::vector<float> split = forward(Way::Split, count, rows);
    const auto fail = [&](const std::string &why) {
      handoff_.retire("its output of " + std::to_string(rows) + " rows " + why);
      throw std::runtime_error("CoreAI FFN split's output of " + std::to_string(rows) + " rows " + why);
    };
    double difference = 0.0, contribution = 0.0;
    for (size_t i = 0; i < split.size(); ++i) {
      if (!std::isfinite(split[i])) fail("is not finite");
      difference += (double(split[i]) - gpu[i]) * (double(split[i]) - gpu[i]);
      contribution += (double(gpu[i]) - part[i]) * (double(gpu[i]) - part[i]);
    }
    const double error = std::sqrt(difference / contribution);
    if (!(error <= kMaximumAneError))
      fail("differs from the GPU alone's by " + std::to_string(100.0 * error) +
           "% of the Neural Engine's part");
    worst = std::max(worst, error);
  }
  return worst;
}

} // namespace richengine::ops
