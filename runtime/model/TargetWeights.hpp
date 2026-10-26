#pragma once

#include "model/TargetModel.hpp"
#include "model/DiffusionGemma.hpp"

#include <variant>

namespace richengine::model {

// The package-level capability split — an ordinary trunk's weights or
// DiffusionGemma's diffusion package — not a per-model dispatch: every
// registered arch loader returns one of these two alternatives.
using TargetWeights = std::variant<TargetModelWeights, DiffusionGemmaWeights>;

} // namespace richengine::model
