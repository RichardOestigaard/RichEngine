#include "model/DFlashTwo16width.hpp"

namespace richengine::model {

DFlashTwo16width::DFlashTwo16width(const DFlashDraftWeights &weights,
                                   metal::MetalBackend &backend,
                                   const ops::ExecutionPlans &operators)
    : DFlashDraft(weights, backend, operators) {}

} // namespace richengine::model
