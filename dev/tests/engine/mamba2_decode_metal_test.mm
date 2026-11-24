// Mamba-2 (SSD) verify decode, commit and prefill kernels against a direct
// CPU reference: the four-tap causal convolution with SiLU over the x|B|C
// channels, dt' = clamp(softplus(dt + bias), dt_min, dt_max) and
// dA = exp(dt' * A), the [HeadDim][StateSize] fp32 recurrence
// S = dA*S + (dt'*x_d) B_n' emitting y_d = sum_n C_n S_dn + D x_d, the
// grouped gated RMSNorm (gate before norm, NemotronHRMSNormGated), the
// convolution carry, a two-layer state cell (so the layer offsets are
// exercised), every retained count of the commit, the tape fold, and the
// out-projection tables. The recurrent state is checked at fp32 accuracy
// from the kernel's own post-conv rows and dt', after those were checked
// against the reference; the hidden rows from the recurrent rows read from
// that state.
#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "model/StateLayout.hpp"
#include "ops/Mamba2.hpp"
#include "tuning/LinearNumerics.hpp"

#include "LinearInputReference.hpp"
#include "NormReference.hpp"

#import <Foundation/Foundation.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using richengine::metal::BufferStorage;
using richengine::metal::CommandGraph;
using richengine::metal::MetalBackend;
using richengine::metal::MetalBuffer;
using namespace richengine::ops;
using namespace richengine::test;
using richengine::ops::tuning::bf16ToFloat;
using richengine::ops::tuning::floatToBf16;

constexpr uint32_t kRows = RICHENGINE_TARGET_VERIFY_ROWS;
constexpr uint32_t kMaxLanes = RICHENGINE_MAXIMUM_BATCH_WIDTH;
constexpr uint32_t kLayers = 2;
constexpr uint32_t kHeads = 64;
constexpr uint32_t kHeadDim = 64;
constexpr uint32_t kGroups = 8;
constexpr uint32_t kState = 128;
constexpr uint32_t kHeadsPerGroup = kHeads / kGroups;
constexpr uint32_t kInner = kHeads * kHeadDim;
constexpr uint32_t kConv = 6144;
constexpr uint32_t kGroupWidth = kInner / kGroups;
constexpr float kDtMin = 0.001F, kDtMax = 0.1F;
// The packed projection's storage rows pad the 10304 logical columns to a
// 256-element tile.
constexpr uint32_t kPackedRow =
    (kInner + kConv + kHeads + 255) / 256 * 256;
constexpr Mamba2Shape kShape{kHeads, kHeadDim, kGroups, kState, 4,
                             kInner, kConv, kPackedRow,
                             kDtMin, kDtMax, Mamba2Core::Nemotron64};

double roundBfloat(double value) {
  return bf16ToFloat(floatToBf16(static_cast<float>(value)));
}

// One bf16 unit in the last place at the reference's magnitude.
double bfloatUlp(double reference) {
  int exponent = 0;
  std::frexp(std::fabs(reference), &exponent);
  return std::ldexp(1.0, exponent - 8);
}

bool closeBfloat(uint16_t got, double reference, double ulps, double floor) {
  return std::fabs(double(bf16ToFloat(got)) - reference) <=
         ulps * bfloatUlp(reference) + floor;
}

bool closeFloat(float got, double reference, double tolerance) {
  return std::fabs(double(got) - reference) <=
         tolerance * (1.0 + std::fabs(reference));
}

class Random final {
public:
  explicit Random(uint64_t seed) : state_(seed) {}
  float unit() {
    state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<float>((state_ >> 40) & 0xFFFFFF) / 8388608.0F - 1.0F;
  }

private:
  uint64_t state_;
};

double sigmoid(double value) { return 1.0 / (1.0 + std::exp(-value)); }
double silu(double value) { return value * sigmoid(value); }

// The production state cell layout: every layer's conv rows, then every
// layer's recurrent state, both padded to 16 KiB — the same GdnStateLayout
// accounting, with the recurrent matrix as Heads * HeadDim * StateSize fp32.
struct Cell final {
  richengine::model::GdnStateLayout layout;
  uint64_t convLayerBytes = 0;
  uint64_t recurrentLayerBytes = 0;
  uint64_t convBytes = 0;
  uint64_t bytes = 0;

  Cell()
      : layout{kLayers, 3, kConv, kHeads, kHeadDim, kState},
        convLayerBytes(layout.convolutionLayerBytes()),
        recurrentLayerBytes(layout.recurrentLayerBytes()),
        convBytes(layout.convolutionBytes()), bytes(layout.cellBytes()) {}

  Mamba2StateStrides strides() const {
    return {convLayerBytes, recurrentLayerBytes, convBytes};
  }
  const uint16_t *conv(const uint8_t *cell, uint32_t layer) const {
    return reinterpret_cast<const uint16_t *>(cell + layer * convLayerBytes);
  }
  uint16_t *conv(uint8_t *cell, uint32_t layer) {
    return reinterpret_cast<uint16_t *>(cell + layer * convLayerBytes);
  }
  const float *recurrent(const uint8_t *cell, uint32_t layer) const {
    return reinterpret_cast<const float *>(cell + convBytes +
                                           layer * recurrentLayerBytes);
  }
  float *recurrent(uint8_t *cell, uint32_t layer) {
    return reinterpret_cast<float *>(cell + convBytes +
                                     layer * recurrentLayerBytes);
  }
};

// The mixer norm is bf16, or F32 as a GGUF stores it.
struct Fixture final {
  Cell cell;
  uint32_t lanes;
  MetalBuffer packed, convWeights, convBias, mixed, aWeights, timeBias,
      stepSizes, skipWeights, hidden, retained, tape;
  NormWeights mixerNorm;
  std::array<MetalBuffer, kMaxLanes> current, next;
  std::vector<MetalBuffer> packedLayer, mixedLayer, stepLayer, aWeightsLayer;
  uint64_t packedStride, mixedStride, stepStride;

  Fixture(MetalBackend &backend, uint32_t laneCount, bool float32 = false)
      : lanes(laneCount), packedStride(uint64_t{kRows} * kPackedRow),
        mixedStride(uint64_t{kRows} * kConv),
        stepStride(uint64_t{kRows} * kHeads) {
    Random random(0x4D414D42ULL + laneCount + (float32 ? 0x100 : 0));
    auto alloc = [&](uint64_t bytes, const char *label) {
      return backend.allocateBuffer(bytes, BufferStorage::Shared, label);
    };
    auto fill = [&](MetalBuffer &buffer, float scale) {
      auto *values = static_cast<uint16_t *>(buffer.contents());
      for (uint64_t index = 0; index < buffer.sizeBytes() / 2; ++index)
        values[index] = floatToBf16(random.unit() * scale);
    };
    packed = alloc(kLayers * kMaxLanes * packedStride * 2, "mamba2 packed");
    fill(packed, 1.0F);
    convWeights = alloc(uint64_t{kConv} * 4 * 2, "mamba2 conv weights");
    fill(convWeights, 0.5F);
    convBias = alloc(uint64_t{kConv} * 2, "mamba2 conv bias");
    fill(convBias, 0.25F);
    mixed = alloc(kLayers * kMaxLanes * mixedStride * 2, "mamba2 mixed");
    // One buffer holds every layer's ssm_a: a decode binds the layer's
    // [heads] slice, the batched commit strides the layers itself.
    aWeights = alloc(uint64_t{kLayers} * kHeads * 4, "mamba2 a weights");
    for (uint32_t head = 0; head < kLayers * kHeads; ++head)
      static_cast<float *>(aWeights.contents())[head] =
          -std::exp(random.unit() * 1.5F);
    timeBias = alloc(uint64_t{kHeads} * 2, "mamba2 time bias");
    fill(timeBias, 0.5F);
    stepSizes =
        alloc(kLayers * kMaxLanes * stepStride * 4, "mamba2 step sizes");
    skipWeights = alloc(uint64_t{kHeads} * 4, "mamba2 skip weights");
    for (uint32_t head = 0; head < kHeads; ++head)
      static_cast<float *>(skipWeights.contents())[head] = random.unit();
    hidden = alloc(uint64_t{kMaxLanes} * kRows * kInner * 2, "mamba2 hidden");
    mixerNorm = makeNormWeights(backend, kInner, float32,
                                [&](uint32_t) { return random.unit(); });
    // Nemotron's grouped norm epsilon.
    mixerNorm.rmsEpsilon = 1e-5F;
    retained = alloc(kMaxLanes * 4, "mamba2 retained");
    tape = alloc(uint64_t{kMaxLanes} * 2 * kLayers * tapeLayerBytes(),
                 "mamba2 tape");
    for (uint32_t lane = 0; lane < kMaxLanes; ++lane) {
      current[lane] = alloc(cell.bytes, "mamba2 current");
      next[lane] = alloc(cell.bytes, "mamba2 next");
      auto *bytes = static_cast<uint8_t *>(current[lane].contents());
      auto *conv = reinterpret_cast<uint16_t *>(bytes);
      for (uint64_t index = 0; index < cell.convBytes / 2; ++index)
        conv[index] = floatToBf16(random.unit());
      auto *state = reinterpret_cast<float *>(bytes + cell.convBytes);
      for (uint64_t index = 0; index < (cell.bytes - cell.convBytes) / 4;
           ++index)
        state[index] = random.unit() * 0.5F;
    }
    for (uint32_t layer = 0; layer < kLayers; ++layer) {
      packedLayer.push_back(backend.view(packed,
                                         layer * kMaxLanes * packedStride * 2,
                                         kMaxLanes * packedStride * 2));
      mixedLayer.push_back(backend.view(mixed,
                                        layer * kMaxLanes * mixedStride * 2,
                                        kMaxLanes * mixedStride * 2));
      stepLayer.push_back(backend.view(stepSizes,
                                       layer * kMaxLanes * stepStride * 4,
                                       kMaxLanes * stepStride * 4));
      aWeightsLayer.push_back(
          backend.view(aWeights, layer * uint64_t{kHeads} * 4,
                       uint64_t{kHeads} * 4));
    }
    clear();
  }

  void clear() {
    for (uint32_t lane = 0; lane < kMaxLanes; ++lane)
      std::memset(next[lane].contents(), 0, cell.bytes);
    for (MetalBuffer *buffer : {&mixed, &stepSizes, &hidden})
      std::memset(buffer->contents(), 0, buffer->sizeBytes());
  }

  uint64_t tapeLayerBytes() const {
    return (Mamba2::tapeLayerBytes(kShape, kRows) + 255) / 256 * 256;
  }
  Mamba2DecodeBuffers decodeBuffers(uint32_t layer) const {
    return {packedLayer[layer], convWeights,       convBias,
            current,            next,
            mixedLayer[layer],  aWeightsLayer[layer],
            timeBias,           stepLayer[layer],  skipWeights,
            mixerNorm,          hidden,            {},
            tape};
  }
  Mamba2CommitBuffers commitBuffers() const {
    return {packed, mixed, stepSizes, aWeights, current, next, retained};
  }

  // Row `token` of lane `lane` in layer `layer` of the packed projection.
  const uint16_t *packedRow(uint32_t layer, uint32_t lane,
                            uint32_t token) const {
    return static_cast<const uint16_t *>(packed.contents()) +
           (uint64_t{layer} * kMaxLanes + lane) * packedStride +
           uint64_t{token} * kPackedRow;
  }
  const uint16_t *mixedRow(uint32_t layer, uint32_t lane,
                           uint32_t token) const {
    return static_cast<const uint16_t *>(mixed.contents()) +
           (uint64_t{layer} * kMaxLanes + lane) * mixedStride +
           uint64_t{token} * kConv;
  }
  const float *stepRow(uint32_t layer, uint32_t lane, uint32_t token) const {
    return static_cast<const float *>(stepSizes.contents()) +
           (uint64_t{layer} * kMaxLanes + lane) * stepStride +
           uint64_t{token} * kHeads;
  }
  const uint16_t *hiddenRow(uint32_t lane, uint32_t token) const {
    return static_cast<const uint16_t *>(hidden.contents()) +
           (uint64_t{lane} * kRows + token) * kInner;
  }
  const uint8_t *cellBytes(const MetalBuffer &buffer) const {
    return static_cast<const uint8_t *>(buffer.contents());
  }
};

// Four-tap causal convolution of one channel at one token, three carried
// rows then the command's conv rows, rounded to bf16 and gated by SiLU — the
// same helper shape as the GDN test's, the conv region leading the packed
// row's tail at kInner.
double convolutionSilu(const Fixture &fixture, uint32_t layer, uint32_t lane,
                       uint32_t token, uint32_t channel) {
  const uint16_t *weights =
      static_cast<const uint16_t *>(fixture.convWeights.contents()) +
      uint64_t{channel} * 4;
  const uint16_t *carried =
      fixture.cell.conv(fixture.cellBytes(fixture.current[lane]), layer);
  const uint16_t *bias =
      static_cast<const uint16_t *>(fixture.convBias.contents());
  double value = 0.0;
  for (uint32_t tap = 0; tap < 4; ++tap) {
    const uint32_t position = token + tap;
    const uint16_t input =
        position < 3 ? carried[position * kConv + channel]
                     : fixture.packedRow(layer, lane, position - 3)[kInner + channel];
    value += double(bf16ToFloat(input)) * bf16ToFloat(weights[tap]);
  }
  value += bf16ToFloat(bias[channel]);
  value = roundBfloat(value);
  return roundBfloat(value * sigmoid(value));
}

uint16_t convolutionCarry(const Fixture &fixture, uint32_t layer,
                          uint32_t lane, uint32_t consumed, uint32_t row,
                          uint32_t channel) {
  const uint32_t source = consumed + row;
  const uint16_t *carried =
      fixture.cell.conv(fixture.cellBytes(fixture.current[lane]), layer);
  return source < 3 ? carried[source * kConv + channel]
                    : fixture.packedRow(layer, lane, source - 3)[kInner + channel];
}

// dt' = clamp(softplus(dt + bias), limits), the bias add and the softplus
// rounded to bf16 as the kernel's; the kernel's fast transcendentals may land
// the softplus one bf16 step away near a boundary, so every neighbouring
// rounded value is a candidate.
std::array<double, 3> stepSizeCandidates(double dtRaw, double bias) {
  const double x = roundBfloat(dtRaw + bias);
  const double softplus =
      std::max(x, 0.0) + std::log1p(std::exp(-std::fabs(x)));
  const uint16_t rounded = floatToBf16(static_cast<float>(softplus));
  std::array<double, 3> candidates;
  for (int step = -1; step <= 1; ++step)
    candidates[step + 1] = std::clamp(
        double(bf16ToFloat(static_cast<uint16_t>(rounded + step))),
        double(kDtMin), double(kDtMax));
  return candidates;
}

void checkConvolution(const Fixture &fixture, uint32_t layer, uint32_t lane,
                      const std::string &where) {
  for (uint32_t token = 0; token < kRows; ++token) {
    const uint16_t *mixed = fixture.mixedRow(layer, lane, token);
    for (uint32_t channel = 0; channel < kConv; ++channel)
      require(closeBfloat(mixed[channel],
                          convolutionSilu(fixture, layer, lane, token,
                                          channel),
                          3.0, 1e-6),
              where + ": mixed row mismatch");
  }
}

void checkSteps(const Fixture &fixture, uint32_t layer, uint32_t lane,
                const std::string &where) {
  const auto *bias =
      static_cast<const uint16_t *>(fixture.timeBias.contents());
  for (uint32_t token = 0; token < kRows; ++token) {
    const uint16_t *packed = fixture.packedRow(layer, lane, token);
    const float *steps = fixture.stepRow(layer, lane, token);
    for (uint32_t head = 0; head < kHeads; ++head) {
      const std::array<double, 3> candidates = stepSizeCandidates(
          bf16ToFloat(packed[kInner + kConv + head]), bf16ToFloat(bias[head]));
      bool matched = false;
      for (const double candidate : candidates)
        matched = matched || closeFloat(steps[head], candidate, 1e-5);
      if (!matched) {
        std::cerr << where << " token=" << token << " head=" << head
                  << " actual=" << steps[head] << " expected "
                  << candidates[0] << "/" << candidates[1] << "/"
                  << candidates[2] << "\n";
        require(false, where + ": step size mismatch");
      }
    }
  }
}

// The SSD recurrence over `tokens` rows of one head from the lane's incoming
// state, driven by the kernel's own post-conv rows and dt'. `mixed` and
// `steps` point at row 0 of the lane's tensors ([token][kConv] bf16 and
// [token][kHeads] fp32); the recurrent output rows go to `rows` when
// requested, in the per-head layout [token][kHeadDim].
std::vector<double> recurrence(const uint16_t *mixed, const float *steps,
                               const float *aWeights, float dSkip,
                               const float *stateIn, uint32_t head,
                               uint32_t tokens, std::vector<double> *rows) {
  const uint32_t group = head / kHeadsPerGroup;
  std::vector<double> state(stateIn + uint64_t{head} * kHeadDim * kState,
                            stateIn + (uint64_t{head} + 1) * kHeadDim * kState);
  if (rows)
    rows->assign(uint64_t{tokens} * kHeadDim, 0.0);
  for (uint32_t token = 0; token < tokens; ++token) {
    const uint16_t *conv = mixed + uint64_t{token} * kConv;
    const double dtv = steps[uint64_t{token} * kHeads + head];
    const double dA = std::exp(dtv * double(aWeights[head]));
    const uint16_t *xRow = conv + head * kHeadDim;
    const uint16_t *bRow = conv + kInner + group * kState;
    const uint16_t *cRow = conv + kInner + kGroups * kState + group * kState;
    for (uint32_t dim = 0; dim < kHeadDim; ++dim) {
      double *row = state.data() + uint64_t{dim} * kState;
      const double scale = dtv * bf16ToFloat(xRow[dim]);
      double output = 0.0;
      for (uint32_t n = 0; n < kState; ++n) {
        row[n] = std::fma(scale, bf16ToFloat(bRow[n]), row[n] * dA);
        output += row[n] * bf16ToFloat(cRow[n]);
      }
      if (rows)
        (*rows)[uint64_t{token} * kHeadDim + dim] =
            output + dSkip * bf16ToFloat(xRow[dim]);
    }
  }
  return state;
}

// The fixture's recurrence entry: one head of lane `lane`, layer `layer`.
std::vector<double> recurrence(const Fixture &fixture, uint32_t layer,
                               uint32_t lane, uint32_t head, uint32_t tokens,
                               std::vector<double> *rows) {
  const auto *aWeights =
      static_cast<const float *>(fixture.aWeights.contents()) +
      uint64_t{layer} * kHeads;
  const auto *skips =
      static_cast<const float *>(fixture.skipWeights.contents());
  return recurrence(
      fixture.mixedRow(layer, lane, 0), fixture.stepRow(layer, lane, 0),
      aWeights, skips[head],
      fixture.cell.recurrent(fixture.cellBytes(fixture.current[lane]), layer),
      head, tokens, rows);
}

void checkState(const Fixture &fixture, uint32_t layer, uint32_t lane,
                uint32_t head, const std::vector<double> &expected,
                const std::string &where) {
  const float *state =
      fixture.cell.recurrent(fixture.cellBytes(fixture.next[lane]), layer) +
      uint64_t{head} * kHeadDim * kState;
  for (uint32_t index = 0; index < kHeadDim * kState; ++index)
    require(closeFloat(state[index], expected[index], 1e-4),
            where + ": recurrent state mismatch");
}

void checkCarry(const Fixture &fixture, uint32_t layer, uint32_t lane,
                uint32_t consumed, const std::string &where) {
  const uint16_t *carry =
      fixture.cell.conv(fixture.cellBytes(fixture.next[lane]), layer);
  for (uint32_t row = 0; row < 3; ++row)
    for (uint32_t channel = 0; channel < kConv; ++channel)
      require(carry[row * kConv + channel] ==
                  convolutionCarry(fixture, layer, lane, consumed, row,
                                   channel),
              where + ": convolution carry mismatch");
}

// The grouped gated RMSNorm of one group's 512 recurrent row values at one
// token: v = y * silu(z), normalized, scaled by the norm weight — the
// reference rows computed from `y` (the scan's fp64 outputs rounded to bf16
// as the kernel's own hidden rows are) and the gate columns of the packed
// row. Where the fp32 row cancels, the kernel's value can sit a bf16 step or
// two off the double-precision reference, so a small inexact fraction is
// allowed like the GDN gate check's.
uint64_t checkGate(const uint16_t *hiddenRow, const uint16_t *packedRow,
                   const std::vector<double> &y, uint32_t token,
                   uint32_t group, const NormWeights &mixerNorm,
                   const std::string &where) {
  const uint32_t first = group * kGroupWidth;
  std::vector<double> gated(kGroupWidth);
  double squares = 0.0;
  for (uint32_t e = 0; e < kGroupWidth; ++e) {
    const double value =
        roundBfloat(y[uint64_t{token} * kInner + first + e]) *
        silu(bf16ToFloat(packedRow[first + e]));
    gated[e] = value;
    squares += value * value;
  }
  const double inverse =
      1.0 / std::sqrt(squares / kGroupWidth + mixerNorm.rmsEpsilon);
  uint64_t inexact = 0;
  for (uint32_t e = 0; e < kGroupWidth; ++e) {
    const double reference =
        gated[e] * inverse * normWeight(mixerNorm, first + e);
    if (!closeBfloat(hiddenRow[first + e], reference, 3.0, 1e-6)) {
      static int shown = 0;
      if (shown++ < 20)
        std::cerr << where << " token=" << token << " group=" << group
                  << " dim=" << e
                  << " actual=" << bf16ToFloat(hiddenRow[first + e])
                  << " expected=" << reference << "\n";
      ++inexact;
    }
  }
  return inexact;
}

// The scan's per-token outputs of every head, collected into the hidden
// layout [token][kInner] the grouped gate reads.
std::vector<double> scanRows(const Fixture &fixture, uint32_t layer,
                             uint32_t lane, uint32_t tokens) {
  std::vector<double> y(uint64_t{tokens} * kInner);
  std::vector<double> rows;
  for (uint32_t head = 0; head < kHeads; ++head) {
    static_cast<void>(recurrence(fixture, layer, lane, head, tokens, &rows));
    for (uint32_t token = 0; token < tokens; ++token)
      for (uint32_t dim = 0; dim < kHeadDim; ++dim)
        y[uint64_t{token} * kInner + head * kHeadDim + dim] =
            rows[uint64_t{token} * kHeadDim + dim];
  }
  return y;
}

// The hidden rows' inexact count must stay under 1% of the row values.
void checkHidden(const std::vector<double> &y, const uint16_t *hiddenBase,
                 const uint16_t *packedBase, uint64_t packedStride,
                 uint32_t tokens, const NormWeights &mixerNorm,
                 const std::string &where) {
  uint64_t inexact = 0;
  for (uint32_t token = 0; token < tokens; ++token)
    for (uint32_t group = 0; group < kGroups; ++group)
      inexact += checkGate(hiddenBase + uint64_t{token} * kInner,
                           packedBase + uint64_t{token} * packedStride, y,
                           token, group, mixerNorm, where);
  require(inexact <= uint64_t{tokens} * kInner / 100,
          where + ": hidden differs from the reference in more than 1% of values");
}

void checkDecode(const Fixture &fixture, uint32_t layer, uint32_t lane) {
  const std::string where = "decode layer " + std::to_string(layer) +
                            " lane " + std::to_string(lane);
  checkConvolution(fixture, layer, lane, where);
  checkSteps(fixture, layer, lane, where);
  checkCarry(fixture, layer, lane, kRows, where);
  // Every layer writes the hidden rows into the same scratch, as the model
  // graph does, so those hold the last layer's gated values.
  const bool lastLayer = layer + 1 == kLayers;
  for (uint32_t head = 0; head < kHeads; ++head) {
    const std::vector<double> state = recurrence(
        fixture, layer, lane, head, kRows, nullptr);
    checkState(fixture, layer, lane, head, state, where);
  }
  if (!lastLayer)
    return;
  const std::vector<double> y = scanRows(fixture, layer, lane, kRows);
  checkHidden(y, fixture.hiddenRow(lane, 0),
              fixture.packedRow(layer, lane, 0), kPackedRow, kRows,
              fixture.mixerNorm, where);
}

void requireUntouched(const Fixture &fixture, uint32_t lane,
                      const std::string &where) {
  const uint8_t *bytes = fixture.cellBytes(fixture.next[lane]);
  for (uint64_t index = 0; index < fixture.cell.bytes; ++index)
    require(bytes[index] == 0, where + ": idle lane cell was written");
}

void runDecode(MetalBackend &backend, uint32_t lanes, bool float32) {
  Fixture fixture(backend, lanes, float32);
  CommandGraph graph;
  const std::string where =
      "lanes " + std::to_string(lanes) + (float32 ? " f32 norm" : "");
  for (uint32_t layer = 0; layer < kLayers; ++layer)
    require(Mamba2::addDecode(graph, fixture.decodeBuffers(layer), kShape,
                              lanes, layer, fixture.cell.strides(),
                              LinearInput::Plain)
                    .layout == LinearInput::Plain,
            where + ": plain Mamba2 claimed a table");
  static_cast<void>(backend.submitCommand(graph.dispatches()));
  for (uint32_t lane = 0; lane < kMaxLanes; ++lane) {
    if (lane >= lanes) {
      requireUntouched(fixture, lane, where);
      continue;
    }
    for (uint32_t layer = 0; layer < kLayers; ++layer)
      checkDecode(fixture, layer, lane);
  }
  // The commit reads no norm weights; the bf16 pass covers it.
  if (float32)
    return;

  // The commit replays the retained rows from the incoming cell over the
  // decoded post-conv rows and dt'; eight retained rows leave the decoded
  // cell.
  std::vector<std::vector<uint8_t>> decoded;
  for (uint32_t lane = 0; lane < lanes; ++lane)
    decoded.emplace_back(fixture.cellBytes(fixture.next[lane]),
                         fixture.cellBytes(fixture.next[lane]) +
                             fixture.cell.bytes);
  CommandGraph commit;
  Mamba2::addCommit(commit, fixture.commitBuffers(), kShape, kLayers, lanes,
                    fixture.cell.strides());
  for (uint32_t base = 1; base <= kRows; ++base) {
    auto *retained = static_cast<uint32_t *>(fixture.retained.contents());
    for (uint32_t lane = 0; lane < kMaxLanes; ++lane)
      retained[lane] = 1 + (base + lane - 1) % kRows;
    for (uint32_t lane = 0; lane < lanes; ++lane)
      std::memcpy(fixture.next[lane].contents(), decoded[lane].data(),
                  fixture.cell.bytes);
    static_cast<void>(backend.submitCommand(commit.dispatches()));
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      const uint32_t count = retained[lane];
      const std::string commitWhere =
          where + " commit retained " + std::to_string(count) + " lane " +
          std::to_string(lane);
      if (count == kRows) {
        require(std::memcmp(fixture.next[lane].contents(),
                            decoded[lane].data(), fixture.cell.bytes) == 0,
                commitWhere + ": full acceptance rewrote the cell");
        continue;
      }
      // Every head at the shortest, a middle and the longest partial replay;
      // a spread of heads for the other counts keeps the double-precision
      // replay short under shader validation.
      std::vector<uint32_t> heads{0, kHeads / 2, kHeads - 1};
      if (count == 1 || count == 4 || count == 7) {
        heads.clear();
        for (uint32_t head = 0; head < kHeads; ++head)
          heads.push_back(head);
      }
      for (uint32_t layer = 0; layer < kLayers; ++layer) {
        checkCarry(fixture, layer, lane, count, commitWhere);
        for (uint32_t head : heads)
          checkState(fixture, layer, lane, head,
                     recurrence(fixture, layer, lane, head, count, nullptr),
                     commitWhere);
      }
    }
  }
}

// The folded commit: step one's decoded rows land on the lane's tape; step
// two's scan folds the retained ones into the recurrent state as its pending
// prefix — its committed cell must equal the standalone commit's.
void runFold(MetalBackend &backend, uint32_t lanes) {
  Fixture fixture(backend, lanes, false);
  const std::string where = "fold lanes " + std::to_string(lanes);
  const uint64_t slotBytes = uint64_t{kLayers} * fixture.tapeLayerBytes();
  CommandGraph step1;
  Mamba2TapeLanes tape1;
  tape1.layerBytes = fixture.tapeLayerBytes();
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    tape1.step[lane] = uint64_t{lane} * 2 * slotBytes;
    tape1.pending[lane] = tape1.step[lane] + slotBytes;
    tape1.rows[lane] = 0;
  }
  for (uint32_t layer = 0; layer < kLayers; ++layer)
    Mamba2::addDecode(step1, fixture.decodeBuffers(layer), kShape, lanes,
                      layer, fixture.cell.strides(), LinearInput::Plain, {},
                      kRows, &tape1);
  static_cast<void>(backend.submitCommand(step1.dispatches()));
  // Step two folds the retained prefix of step one's rows: each lane's
  // committed cell becomes state_in + retained rows — the standalone
  // commit's output for the same retained count.
  std::array<uint32_t, kMaxLanes> retained{};
  auto *counts = static_cast<uint32_t *>(fixture.retained.contents());
  for (uint32_t lane = 0; lane < kMaxLanes; ++lane)
    counts[lane] = retained[lane] = 1 + lane % (kRows - 1);
  CommandGraph step2;
  Mamba2TapeLanes tape2;
  tape2.layerBytes = fixture.tapeLayerBytes();
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    tape2.pending[lane] = uint64_t{lane} * 2 * slotBytes;
    tape2.step[lane] = tape2.pending[lane] + slotBytes;
    tape2.rows[lane] = retained[lane];
  }
  // The committed cell lands in `next` — step one's output was already
  // verified; reset it so the fold's write is the only signal. The reference
  // reads the emitted rows — step two overwrites them, so compute the
  // expected states before it runs.
  std::vector<std::vector<std::vector<double>>> expected(lanes);
  for (uint32_t lane = 0; lane < lanes; ++lane)
    for (uint32_t layer = 0; layer < kLayers; ++layer)
      for (uint32_t head : {0u, kHeads / 2, kHeads - 1})
        expected[lane].push_back(
            recurrence(fixture, layer, lane, head, retained[lane], nullptr));
  for (uint32_t lane = 0; lane < lanes; ++lane)
    std::memset(fixture.next[lane].contents(), 0, fixture.cell.bytes);
  for (uint32_t layer = 0; layer < kLayers; ++layer)
    Mamba2::addDecode(step2, fixture.decodeBuffers(layer), kShape, lanes,
                      layer, fixture.cell.strides(), LinearInput::Plain, {},
                      kRows, &tape2);
  static_cast<void>(backend.submitCommand(step2.dispatches()));
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    const std::string laneWhere =
        where + " retained " + std::to_string(retained[lane]) + " lane " +
        std::to_string(lane);
    size_t index = 0;
    for (uint32_t layer = 0; layer < kLayers; ++layer) {
      // The folded carry holds the last three of the pending prefix's raw
      // conv inputs — step one's packed rows, which the tape retained.
      checkCarry(fixture, layer, lane, retained[lane], laneWhere);
      for (uint32_t head : {0u, kHeads / 2, kHeads - 1})
        checkState(fixture, layer, lane, head, expected[lane][index++],
                   laneWhere);
    }
  }
}

// The out-projection table a fused decode writes into its scratch and the one
// the projection's own preparation writes from the decoded hidden rows.
struct PreparedTables final {
  LinearInput layout;
  uint32_t width, lanes;
  uint64_t tableSize, sumsSize;
  MetalBuffer table, sums, referenceTable, referenceSums;

  PreparedTables(MetalBackend &backend, LinearInput tableLayout,
                 uint32_t hiddenWidth, uint32_t laneCount)
      : layout(tableLayout), width(hiddenWidth), lanes(laneCount),
        tableSize(tableBytes(width, lanes * kRows)),
        sumsSize(tableSumsBytes(layout, width, lanes * kRows)),
        table(sharedBuffer(backend, tableSize)),
        sums(sharedBuffer(backend, sumsSize)),
        referenceTable(sharedBuffer(backend, tableSize)),
        referenceSums(sharedBuffer(backend, sumsSize)) {}

  LinearScratch scratch() const { return {table, sums, {}, {}}; }
  void addReference(CommandGraph &graph, const MetalBuffer &hidden) const {
    addReferencePreparation(graph, layout, hidden, referenceTable,
                            referenceSums, width, lanes * kRows / 8);
  }
  // The decode reported the table it wrote, and wrote the reference's bytes.
  void requireWritten(const PreparedInput &reported, const MetalBuffer &hidden,
                      const std::string &what) const {
    require(reported.layout == layout && reported.source.sameView(hidden),
            what + " did not report the table it wrote");
    require(!std::memcmp(table.contents(), referenceTable.contents(),
                         tableSize),
            what + " table mismatch");
    require(!std::memcmp(sums.contents(), referenceSums.contents(), sumsSize),
            what + " sums mismatch");
  }
};

std::string caseName(const char *test, uint32_t lanes, LinearInput layout) {
  return std::string(test) + " lanes " + std::to_string(lanes) +
         (layout == LinearInput::Plain     ? " plain"
          : layout == LinearInput::Table64 ? " Table64"
          : layout == LinearInput::Table16 ? " Table16"
                                           : " Packed");
}

void fusedPreparation(MetalBackend &backend, uint32_t lanes,
                      LinearInput layout, bool float32) {
  const std::string what = caseName("fused Mamba2", lanes, layout);
  Fixture fixture(backend, lanes, float32);
  const PreparedTables tables(backend, layout, kInner, lanes);
  CommandGraph reference;
  // Without scratch the table cannot be written: Mamba2 refuses it before it
  // encodes anything.
  rejects([&] {
    Mamba2::addDecode(reference, fixture.decodeBuffers(0), kShape, lanes, 0,
                      fixture.cell.strides(), layout);
  }, "scratch", what + " accepted a table without scratch");
  require(Mamba2::addDecode(reference, fixture.decodeBuffers(0), kShape,
                            lanes, 0, fixture.cell.strides(),
                            LinearInput::Plain)
                  .layout == LinearInput::Plain,
          what + " plain kernel claimed a table");
  tables.addReference(reference, fixture.hidden);
  (void)backend.submitCommand(reference.dispatches());
  // The active lanes' bf16 hidden rows.
  std::vector<uint8_t> expected(uint64_t{lanes} * kRows * kInner * 2);
  std::memcpy(expected.data(), fixture.hidden.contents(), expected.size());
  fixture.clear();
  auto buffers = fixture.decodeBuffers(0);
  buffers.linearScratch = tables.scratch();
  // A plain consumer gets no table from the scratch it shares.
  CommandGraph unprepared;
  require(Mamba2::addDecode(unprepared, buffers, kShape, lanes, 0,
                            fixture.cell.strides(), LinearInput::Plain)
                  .layout == LinearInput::Plain,
          what + " claimed a table for a plain consumer");
  CommandGraph fused;
  const PreparedInput prepared =
      Mamba2::addDecode(fused, buffers, kShape, lanes, 0,
                        fixture.cell.strides(), layout);
  (void)backend.submitCommand(fused.dispatches());
  require(!std::memcmp(expected.data(), fixture.hidden.contents(),
                       expected.size()),
          what + " changed output");
  tables.requireWritten(prepared, fixture.hidden, what);
  for (uint32_t lane = 0; lane < lanes; ++lane)
    checkDecode(fixture, 0, lane);
}

// The prefill window: `tokens` rows of the packed projection through the
// sequential scan — prepare's conv and dt', the scan's recurrent rows and
// state, the gate's hidden — against the same CPU reference the decode
// checks use.
void runPrefill(MetalBackend &backend, uint32_t tokens, bool sumsKernel) {
  const std::string where = "prefill tokens " + std::to_string(tokens) +
                            (sumsKernel ? " sums" : "");
  Random random(0x505245ULL + tokens);
  auto alloc = [&](uint64_t bytes, const char *label) {
    return backend.allocateBuffer(bytes, BufferStorage::Shared, label);
  };
  auto fill = [&](MetalBuffer &buffer, float scale) {
    auto *values = static_cast<uint16_t *>(buffer.contents());
    for (uint64_t index = 0; index < buffer.sizeBytes() / 2; ++index)
      values[index] = floatToBf16(random.unit() * scale);
  };
  MetalBuffer packed = alloc(uint64_t{tokens} * kPackedRow * 2,
                             "prefill packed");
  fill(packed, 1.0F);
  MetalBuffer convWeights = alloc(uint64_t{kConv} * 4 * 2, "prefill conv");
  fill(convWeights, 0.5F);
  MetalBuffer convBias = alloc(uint64_t{kConv} * 2, "prefill conv bias");
  fill(convBias, 0.25F);
  MetalBuffer convIn = alloc(uint64_t{3} * kConv * 2, "prefill conv in");
  fill(convIn, 1.0F);
  MetalBuffer convOut = alloc(uint64_t{3} * kConv * 2, "prefill conv out");
  MetalBuffer postConv =
      alloc(uint64_t{tokens} * kConv * 2, "prefill post conv");
  MetalBuffer steps =
      alloc(uint64_t{tokens} * kHeads * 4, "prefill step sizes");
  MetalBuffer aWeights = alloc(uint64_t{kHeads} * 4, "prefill a");
  for (uint32_t head = 0; head < kHeads; ++head)
    static_cast<float *>(aWeights.contents())[head] =
        -std::exp(random.unit() * 1.5F);
  MetalBuffer skipWeights = alloc(uint64_t{kHeads} * 4, "prefill skip");
  for (uint32_t head = 0; head < kHeads; ++head)
    static_cast<float *>(skipWeights.contents())[head] = random.unit();
  MetalBuffer timeBias = alloc(uint64_t{kHeads} * 2, "prefill bias");
  fill(timeBias, 0.5F);
  const uint64_t stateFloats = uint64_t{kHeads} * kHeadDim * kState;
  MetalBuffer stateIn = alloc(stateFloats * 4, "prefill state in");
  for (uint64_t index = 0; index < stateFloats; ++index)
    static_cast<float *>(stateIn.contents())[index] = random.unit() * 0.5F;
  MetalBuffer stateOut = alloc(stateFloats * 4, "prefill state out");
  MetalBuffer recurrent =
      alloc(uint64_t{tokens} * kInner * 2, "prefill recurrent");
  NormWeights mixerNorm = makeNormWeights(
      backend, kInner, false, [&](uint32_t) { return random.unit(); });
  mixerNorm.rmsEpsilon = 1e-5F;
  MetalBuffer hidden = alloc(uint64_t{tokens} * kInner * 2, "prefill hidden");
  MetalBuffer sums =
      sumsKernel ? alloc(uint64_t{tokens} * (kInner / 64) * 4, "prefill sums")
                 : MetalBuffer{};
  CommandGraph graph;
  Mamba2::addPrefill(graph,
                     {packed, convWeights, convBias, convIn, convOut,
                      postConv, steps, aWeights, skipWeights, timeBias,
                      stateIn, stateOut, recurrent, mixerNorm, hidden},
                     kShape, tokens, sums);
  static_cast<void>(backend.submitCommand(graph.dispatches()));

  const auto *packedRows = static_cast<const uint16_t *>(packed.contents());
  const auto *convRows = static_cast<const uint16_t *>(convIn.contents());
  const auto *convWeights16 =
      static_cast<const uint16_t *>(convWeights.contents());
  const auto *convBiasRow =
      static_cast<const uint16_t *>(convBias.contents());
  const auto *bias = static_cast<const uint16_t *>(timeBias.contents());
  const auto *a = static_cast<const float *>(aWeights.contents());
  const auto *skips = static_cast<const float *>(skipWeights.contents());
  const auto *post = static_cast<const uint16_t *>(postConv.contents());
  const auto *stepRows = static_cast<const float *>(steps.contents());
  const auto *stateOutF = static_cast<const float *>(stateOut.contents());
  const auto *hiddenRows = static_cast<const uint16_t *>(hidden.contents());

  // The prepare pass: every post-conv channel, dt' per (token, head) and the
  // window's carry.
  for (uint32_t token = 0; token < tokens; ++token)
    for (uint32_t channel = 0; channel < kConv; ++channel) {
      double value = 0.0;
      for (uint32_t tap = 0; tap < 4; ++tap) {
        const uint32_t position = token + tap;
        const uint16_t input =
            position < 3
                ? convRows[position * kConv + channel]
                : packedRows[uint64_t{position - 3} * kPackedRow + kInner +
                             channel];
        value += double(bf16ToFloat(input)) *
                 bf16ToFloat(convWeights16[uint64_t{channel} * 4 + tap]);
      }
      value += bf16ToFloat(convBiasRow[channel]);
      value = roundBfloat(value);
      require(closeBfloat(post[uint64_t{token} * kConv + channel],
                          roundBfloat(value * sigmoid(value)), 3.0, 1e-6),
              where + ": post-conv row mismatch");
    }
  for (uint32_t token = 0; token < tokens; ++token)
    for (uint32_t head = 0; head < kHeads; ++head) {
      const std::array<double, 3> candidates = stepSizeCandidates(
          bf16ToFloat(packedRows[uint64_t{token} * kPackedRow + kInner +
                                 kConv + head]),
          bf16ToFloat(bias[head]));
      bool matched = false;
      for (const double candidate : candidates)
        matched =
            matched || closeFloat(stepRows[uint64_t{token} * kHeads + head],
                                  candidate, 1e-5);
      require(matched, where + ": step size mismatch");
    }
  for (uint32_t row = 0; row < 3; ++row)
    for (uint32_t channel = 0; channel < kConv; ++channel) {
      const uint32_t source = tokens + row;
      const uint16_t expected =
          source < 3 ? convRows[source * kConv + channel]
                     : packedRows[uint64_t{source - 3} * kPackedRow + kInner +
                                  channel];
      require(static_cast<const uint16_t *>(
                  convOut.contents())[row * kConv + channel] == expected,
              where + ": convolution carry mismatch");
    }

  // The scan: every head's state and the recurrent rows.
  std::vector<double> y(uint64_t{tokens} * kInner);
  std::vector<double> headRows;
  for (uint32_t head = 0; head < kHeads; ++head) {
    const std::vector<double> state =
        recurrence(post, stepRows, a, skips[head],
                   static_cast<const float *>(stateIn.contents()), head,
                   tokens, &headRows);
    const float *got = stateOutF + uint64_t{head} * kHeadDim * kState;
    for (uint32_t index = 0; index < kHeadDim * kState; ++index)
      require(closeFloat(got[index], state[index], 1e-4),
              where + ": recurrent state mismatch");
    for (uint32_t token = 0; token < tokens; ++token)
      for (uint32_t dim = 0; dim < kHeadDim; ++dim)
        y[uint64_t{token} * kInner + head * kHeadDim + dim] =
            headRows[uint64_t{token} * kHeadDim + dim];
  }
  // The scan's ungated rows land bf16-rounded in `recurrent`; the fp64
  // reference near a rounding boundary can sit a step or two off the fp32
  // row, so the inexact count is capped like the gate's.
  const auto *recurrentRows = static_cast<const uint16_t *>(recurrent.contents());
  uint64_t inexactRows = 0;
  for (uint32_t token = 0; token < tokens; ++token)
    for (uint32_t dim = 0; dim < kInner; ++dim)
      inexactRows += !closeBfloat(recurrentRows[uint64_t{token} * kInner + dim],
                                  y[uint64_t{token} * kInner + dim], 3.0,
                                  1e-6);
  require(inexactRows <= uint64_t{tokens} * kInner / 100,
          where + ": recurrent rows differ from the reference in more than 1% of values");
  checkHidden(y, hiddenRows, packedRows, kPackedRow, tokens, mixerNorm,
              where);
  // The sums rows the out-projection's prefill tile reads.
  if (sumsKernel) {
    const auto *sumRows = static_cast<const float *>(sums.contents());
    for (uint32_t token = 0; token < tokens; ++token)
      for (uint32_t block = 0; block < kInner / 64; ++block) {
        double expected = 0.0;
        for (uint32_t e = 0; e < 64; ++e)
          expected += bf16ToFloat(
              hiddenRows[uint64_t{token} * kInner + block * 64 + e]);
        require(
            closeFloat(sumRows[uint64_t{token} * (kInner / 64) + block],
                       expected, 1e-5),
            where + ": sums mismatch");
      }
  }
}

void rejectsInvalid(MetalBackend &backend) {
  Fixture fixture(backend, 1);
  CommandGraph graph;
  rejects([&] {
    Mamba2::addDecode(graph, fixture.decodeBuffers(0), kShape, 0, 0,
                      fixture.cell.strides(), LinearInput::Plain);
  }, "invalid", "a zero-lane Mamba2 request was accepted");
  rejects([&] {
    Mamba2::addDecode(graph, fixture.decodeBuffers(0), kShape, kMaxLanes + 1,
                      0, fixture.cell.strides(), LinearInput::Plain);
  }, "invalid", "an over-wide Mamba2 request was accepted");
  rejects([&] {
    Mamba2::addDecode(graph, fixture.decodeBuffers(0),
                      Mamba2Shape{16, 64, 8, 128, 4, 1024, 3072, 4160,
                                  kDtMin, kDtMax, Mamba2Core::Nemotron64},
                      1, 0, fixture.cell.strides(), LinearInput::Plain);
  }, "Mamba2", "a non-Nemotron Mamba2 shape was accepted");
  rejects([&] {
    Mamba2::addCommit(graph, fixture.commitBuffers(), kShape, 0, 1,
                      fixture.cell.strides());
  }, "invalid", "a zero-layer Mamba2 commit was accepted");
  rejects([&] {
    auto buffers = fixture.decodeBuffers(0);
    buffers.linearScratch.input = sharedBuffer(backend, 16);
    buffers.linearScratch.sums = sharedBuffer(backend, 4);
    Mamba2::addDecode(graph, buffers, kShape, 1, 0, fixture.cell.strides(),
                      LinearInput::Table64);
  }, "scratch", "an undersized Mamba2 table scratch was accepted");
  rejects([&] {
    // F32 weights need twice the bytes of bf16 ones.
    auto buffers = fixture.decodeBuffers(0);
    buffers.mixerNorm.float32 = true;
    Mamba2::addDecode(graph, buffers, kShape, 1, 0, fixture.cell.strides(),
                      LinearInput::Plain);
  }, "norm weights", "short F32 Mamba2 norm weights were accepted");
  rejects([&] {
    Mamba2::addPrefill(graph,
                       {fixture.packedLayer[0], fixture.convWeights,
                        fixture.convBias,
                        sharedBuffer(backend, 3 * kConv * 2),
                        sharedBuffer(backend, 3 * kConv * 2),
                        sharedBuffer(backend, 64), sharedBuffer(backend, 64),
                        fixture.aWeights, fixture.skipWeights,
                        fixture.timeBias, sharedBuffer(backend, 64),
                        sharedBuffer(backend, 64), sharedBuffer(backend, 64),
                        fixture.mixerNorm, fixture.hidden},
                       kShape, 0);
  }, "invalid", "a zero-token Mamba2 prefill was accepted");
  require(graph.empty(), "invalid Mamba2 request partially encoded a graph");
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::invalid_argument("usage: mamba2-decode METALLIB");
    MetalBackend backend(argv[1]);
    rejectsInvalid(backend);
    // Table64 feeds the affine models, whose norms are bf16; Table16 and
    // Packed a GGUF's, whose norms are F32.
    for (LinearInput layout :
         {LinearInput::Table64, LinearInput::Table16, LinearInput::Packed})
      for (uint32_t lanes = 1; lanes <= kMaxLanes; ++lanes)
        fusedPreparation(backend, lanes, layout,
                         layout != LinearInput::Table64);
    for (bool float32 : {false, true})
      for (uint32_t lanes = 1; lanes <= kMaxLanes; ++lanes)
        runDecode(backend, lanes, float32);
    for (uint32_t lanes = 1; lanes <= kMaxLanes; ++lanes)
      runFold(backend, lanes);
    // Sequential-window prefill, crossing a scan-block boundary.
    for (uint32_t tokens : {1u, 9u, 17u, 40u})
      for (bool sums : {false, true})
        runPrefill(backend, tokens, sums);
    std::cout << "mamba2_decode_metal_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "mamba2_decode_metal_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
