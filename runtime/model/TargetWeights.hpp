#pragma once

#include "model/TargetModel.hpp"
#include "model/DiffusionGemma.hpp"
#include "model/Qwen4Exp.hpp"

#include <variant>

namespace richengine::model {

// The package-level capability split — an ordinary trunk's weights,
// DiffusionGemma's diffusion package or the qwen4exp package — not a
// per-model dispatch: every registered arch loader returns one of these
// alternatives.
using TargetWeights =
    std::variant<TargetModelWeights, DiffusionGemmaWeights, Qwen4ExpWeights>;

} // namespace richengine::model
