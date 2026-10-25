#include "model/DFlashTwo8width.hpp"

namespace richengine::model {

DFlashTwo8width::DFlashTwo8width(const DFlashDraftWeights &weights,
                                 metal::MetalBackend &backend,
                                 const ops::ExecutionPlans &operators)
    : DFlashDraft(weights, backend, operators) {}

} // namespace richengine::model
