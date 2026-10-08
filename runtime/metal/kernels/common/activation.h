#pragma once
#include <metal_stdlib>
using namespace metal;

// SiLU and the logistic sigmoid of every kernel, e^-x taken as
// fast::exp2(-x * log2(e)). A gated product multiplies richengine_silu(gate)
// into its other factor, so silu(g) * up evaluates g / (1 + e^-g) * up.
inline float richengine_silu(float x) {
  return x / (1.0f + fast::exp2(-1.44269504089f * x));
}
inline float2 richengine_silu(float2 x) {
  return x / (1.0f + fast::exp2(-1.44269504089f * x));
}
inline float richengine_sigmoid(float x) {
  return 1.0f / (1.0f + fast::exp2(-1.44269504089f * x));
}

// gelu_pytorch_tanh (Gemma 4's GeGLU activation):
// 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))). A gated GeGLU product
// multiplies richengine_gelu_tanh(gate) into its up factor.
// tanh itself saturates to ±1, but the exp-based implementations the
// compilers emit overflow for |x| ≳ 44 (exp(2x) passes fp32's maximum and
// the inf/inf quotient NaNs): clamp to the saturation region first.
inline float richengine_tanh(float x) {
  return x > 10.0f ? 1.0f : x < -10.0f ? -1.0f : tanh(x);
}

inline float richengine_gelu_tanh(float x) {
  const float inner =
      0.79788456080286536f * (x + 0.044715f * x * x * x);
  return 0.5f * x * (1.0f + richengine_tanh(inner));
}
