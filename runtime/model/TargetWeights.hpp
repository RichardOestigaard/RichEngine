#pragma once

#include "model/TargetModel.hpp"
#include "model/DiffusionGemma.hpp"
#include "model/Qwen4Exp.hpp"
#include "model/MaxwellMoe.hpp"

#include <variant>

namespace richengine::model {

// The package-level capability split — an ordinary trunk's weights,
// DiffusionGemma's diffusion package, the qwen4exp package or
// maxwell_moe's — not a per-model dispatch: every registered arch loader
// returns one of these alternatives.
using TargetWeights =
    std::variant<TargetModelWeights, DiffusionGemmaWeights, Qwen4ExpWeights,
                 MaxwellMoeWeights>;

} // namespace richengine::model
