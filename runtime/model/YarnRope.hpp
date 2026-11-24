#pragma once

// YaRN RoPE scaling for a draft's rotary tables (Su et al., "YaRN:
// Efficient Context Window Extension of Large Language Models"), the
// convention of Hugging Face transformers' _compute_yarn_parameters:
// every rotary pair whose base frequency's wavelength exceeds the
// original window's beta_slow (1) rotations divides by the declared
// factor, every pair under beta_fast (32) rotations keeps its base
// frequency, and the pairs between blend linearly across the ramp. The
// attention factor mscale = 0.1 ln(scale) + 1 scales the rotated queries
// and keys; the table builder folds it into the draft's cosine and sine
// rows (PrefillArena fills the inverse frequencies, rope_build_tables
// the rows). NVIDIA's Nemotron DFlash v1 draft is the format's first
// YaRN rope_parameters (factor 128 over an 8192-position window).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <span>
#include <stdexcept>

namespace richengine::model {

// The inverse rotary frequencies of `pairs` pairs of base `theta`,
// theta^(-pair / pairs), into `frequencies`. A `factor` past 1 with a
// declared `originalMaxPositions` ramp-blends each pair between its base
// value and its interpolation over the factor; otherwise the table is
// the unscaled base progression and the draft's tables come out
// identical to an unscaled ("default") rope's.
inline void yarnInverseRotaryFrequencies(float theta, uint32_t pairs,
                                         float factor,
                                         uint32_t originalMaxPositions,
                                         std::span<float> frequencies) {
  if (frequencies.size() < pairs)
    throw std::invalid_argument("invalid rotary inverse frequency table");
  for (uint32_t pair = 0; pair < pairs; ++pair)
    frequencies[pair] =
        std::pow(theta, -static_cast<float>(pair) / pairs);
  if (!(factor > 1.0F) || !originalMaxPositions || !pairs)
    return;
  // The correction range over the rotated dimensions: the dimensions
  // where beta_fast (32) and beta_slow (1) rotations fit the original
  // window, and the linear ramp across it over the pairs.
  const double dimensions = 2.0 * pairs;
  const double logBase = std::log(static_cast<double>(theta));
  const auto correction = [&](double rotations) {
    return dimensions *
           std::log(static_cast<double>(originalMaxPositions) /
                    (rotations * 2.0 * std::numbers::pi)) /
           (2.0 * logBase);
  };
  const double low = std::max(std::floor(correction(32.0)), 0.0);
  double high = std::min(std::ceil(correction(1.0)), dimensions - 1.0);
  if (low == high)
    high += 0.001;
  for (uint32_t pair = 0; pair < pairs; ++pair) {
    const double ramp = std::clamp(
        (static_cast<double>(pair) - low) / (high - low), 0.0, 1.0);
    const float base = frequencies[pair];
    frequencies[pair] = static_cast<float>(
        static_cast<double>(base / factor) * ramp + base * (1.0 - ramp));
  }
}

// YaRN's attention factor mscale, the HF convention's get_mscale over
// the effective scale: the draft's declared max_position_embeddings over
// the original window it stretches — the factor alone when the draft
// states none (the two agree on NVIDIA's draft: 1048576 / 8192 = 128).
// 1 for an unscaled draft, so a "default" rope's tables are untouched.
inline float yarnAttentionFactor(float factor, uint32_t originalMaxPositions,
                                 uint32_t maxPositions) noexcept {
  if (!(factor > 1.0F) || !originalMaxPositions)
    return 1.0F;
  const double scale =
      maxPositions ? static_cast<double>(maxPositions) / originalMaxPositions
                   : static_cast<double>(factor);
  return scale > 1.0 ? static_cast<float>(0.1 * std::log(scale) + 1.0)
                     : 1.0F;
}

} // namespace richengine::model
