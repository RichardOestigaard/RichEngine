#pragma once

#include "ane/Handoff.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "ops/FfnSplit.hpp"
#include "ops/Linear.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace richengine::ops {

// The dense-FFN split of ops::AneFfn over the public CoreAI framework
// instead of the private MIL program (docs/ANE_EXPERIMENTS.md, deep dives
// 6-8): W8A16 — the agent's int8 weights dequantize in the model against
// fp16 activations, which the GPU casts off the chunk's normalized bf16
// rows (coreai_ffn_cast), so it runs no rotation, packing or token scales.
// The same channel split, GPU staging a layer ahead (coreai_ffn_weights,
// unrotated) and shared-event handoff (ane::Handoff) apply; encode() binds
// the staging buffers per layer. A function of a multi-function .aimodel
// (dev/coreai/gen_ffn_asset.py, RICHENGINE_COREAI_ASSET) with a dynamic
// row dimension covers every chunk; its first call at a row count
// specializes (~10 ms) inside verify(). Selection is RICHENGINE_COREAI_FFN.
class CoreAiFfn final : public FfnSplit {
public:
  // The ANE takes `aneUnits` of the units() of intermediate channels. The
  // shim and the model's function load here. Throws when the library, the
  // asset (RICHENGINE_COREAI_ASSET) or the function is missing, or the
  // loaded function's contract is not the split's.
  CoreAiFfn(metal::MetalBackend &backend, std::span<const SwiGluProjections> layers, uint32_t aneUnits);
  CoreAiFfn(const CoreAiFfn &) = delete;
  CoreAiFfn &operator=(const CoreAiFfn &) = delete;
  ~CoreAiFfn() override;

  // Why the split does not take `layers`, or null when it does: the
  // projection and shape requirements of the GPU's side, without the
  // W8A8 path's rotation blocks (AneFfn.cpp).
  [[nodiscard]] static const char *unsupported(std::span<const SwiGluProjections> layers);
  // The units the split moves intermediate channels in, as ops::AneFfn's.
  [[nodiscard]] static uint32_t units(std::span<const SwiGluProjections> layers);
  [[nodiscard]] static uint64_t plannedBytes(std::span<const SwiGluProjections> layers, uint32_t aneUnits);
  // Why this Mac cannot serve the split: the shim library
  // (librichengine_coreai.dylib, RICHENGINE_COREAI_SHIM) does not load or
  // the asset path (RICHENGINE_COREAI_ASSET) is not set.
  [[nodiscard]] static std::optional<std::string> unavailable();

  [[nodiscard]] double share() const override;
  [[nodiscard]] uint64_t allocatedBytes() const override { return allocatedBytes_; }
  [[nodiscard]] uint32_t minimumRows() const override { return minimumRows_; }
  void setMinimumRows(uint32_t rows) override;
  void setBreaker(ane_ffn::Breaker breaker, std::function<void()> lost) override {
    breaker_ = std::move(breaker);
    lost_ = std::move(lost);
  }

  [[nodiscard]] const Served &served() const override { return served_; }

  // As AneFfn::verify, with the function exercised at the chunk rows it
  // covers and its output compared with the GPU alone's over two layers.
  [[nodiscard]] double verify(std::span<const SwiGluProjections> layers, const PrefillFfnBuffers &ffn,
                              const std::array<metal::MetalBuffer, 2> &hidden) override;

  [[nodiscard]] bool splits(uint32_t rows) const override;
  [[nodiscard]] bool retired() const override { return handoff_.retired(); }
  [[nodiscard]] std::string reason() const override { return handoff_.reason(); }

  bool begin() override;
  void add(metal::CommandGraph &graph, uint32_t layer, const PrefillFfnBuffers &ffn, metal::MetalBuffer residual,
           metal::MetalBuffer output, uint32_t rows) override;
  [[nodiscard]] metal::CommandTicket commit(const metal::CommandGraph &graph,
                                            metal::CommandCompletion completion) override;
  [[nodiscard]] bool finish() override;

  // The idle release: drops the loaded function and model, which restore()
  // opens again. Same contract as AneFfn::release.
  [[nodiscard]] bool release() override;
  void restore() noexcept override;

private:
  struct Shape final {
    uint32_t hidden = 0, intermediate = 0, layers = 0, gpu = 0, ane = 0;
  };
  // The weight contract the staging serves: Pc is int8 rows under a
  // per-row scale; B32 is int8 under each 32-input block's own scale; Si4
  // packs the codes to nibbles under the same block scales
  // (RICHENGINE_COREAI_QUANT=pc|b32|si4, dev/tools/gen_coreai_ffn.py's
  // --quant must match).
  enum class Quant : uint8_t { Pc, B32, Si4 };
  // A staging set of the layer's quantized weights and fp16 value scales
  // (the dequantize's scale argument, per row or per 32-input block).
  struct Weights final {
    metal::MetalBuffer gateQ, upQ, downQ, gateS, upS, downS;
  };
  // The split's memory: the fp16 activation rows x the model reads and the
  // y it writes, the per-layer row scales as AneFfn lays them out, the
  // join's status word and the two weight staging sets.
  struct Memory final {
    metal::MetalBuffer rowScales, x, y, status;
    std::array<Weights, 2> sets;
  };
  // A projection's weight planes as the coreai_ffn kernels bind them:
  // identical to AneFfn's.
  struct Planes final {
    explicit Planes(const Projection &projection);
    std::array<metal::MetalBuffer, 3> buffers;
    uint32_t groups = 0, format = 0;
    const char *suffix = "";
  };
  enum class Matrix : uint8_t { Gate, Up, Down };
  struct Layer final {
    std::array<Planes, 3> planes;
    Projection gate, up, down;
  };
  enum class Parts : uint8_t { Gpu, Both };
  // An evaluation of weight set `set` at `rows` rows: it starts once the
  // event reaches `ready`, and raises it to `done`.
  struct Job final {
    uint32_t set, rows;
    uint64_t ready, done;
  };

  [[nodiscard]] static Quant quantMode();
  [[nodiscard]] static Shape shapeOf(std::span<const SwiGluProjections> layers, uint32_t aneUnits);
  template <class MakeBuffer>
  [[nodiscard]] static Memory allocate(const Shape &shape, Quant quant, MakeBuffer &&buffer);
  static void fillNormalized(const metal::MetalBuffer &normalized);
  void encode(metal::CommandGraph &graph, uint32_t layer, const PrefillFfnBuffers &ffn, metal::MetalBuffer residual,
              metal::MetalBuffer output, uint32_t rows, Parts parts);
  void addWeights(metal::CommandGraph &graph, uint32_t layer, uint32_t set) const;
  [[nodiscard]] metal::MetalBuffer rowScales(uint32_t layer, Matrix matrix) const;
  void queue(const Job &job);
  [[nodiscard]] std::optional<AwakeClock::duration> completed();
  void warnStopped();
  // Opens the function: the shim handle or null when it could not.
  [[nodiscard]] void *open();

  metal::MetalBackend &backend_;
  const Linear linear_;
  const Quant quant_;
  Shape shape_;
  uint32_t minimumRows_ = kMinimumRows;
  std::vector<Layer> layers_;
  Memory memory_;
  void *function_ = nullptr;
  std::vector<Job> jobs_;
  uint32_t nextLayer_ = 0;
  uint32_t committedEvaluations_ = 0, committedRows_ = 0;
  ane_ffn::Breaker breaker_;
  std::function<void()> lost_;
  Served served_;
  bool unfinished_ = false;
  bool released_ = false;
  bool warned_ = false;
  uint64_t allocatedBytes_ = 0;
  ane::Handoff handoff_;
};

} // namespace richengine::ops
