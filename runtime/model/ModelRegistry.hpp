#pragma once

// The registries that replace the per-model enumeration: each target
// architecture registers its loader under its TargetLayout::arch string, and
// each model family registers the descriptor maker and draft layouts an
// assembly's model.json names (what kModelFamilies[] was). The family
// makers' static-init registration (Phase B) fills them; the factory and the
// descriptor pass look entries up instead of dispatching on the layout type.
//
// Compile note for Phase B: the registration signatures are written against
// the erased types — TargetWeights (TargetWeights.hpp) and the non-template
// TargetFiles — which the family-header conversion makes valid. Until then
// this header intentionally does not parse: ModelDescriptor.hpp still
// aliases TargetLayout to the per-model variant and TargetFiles is still an
// alias template.
#include "model/TargetLayout.hpp"
#include "model/TargetWeights.hpp"
#include "model/TargetFiles.hpp"
#include "model/DFlashDraft.hpp"
#include "model/ModelDescriptor.hpp"

#include <span>
#include <string>
#include <string_view>

namespace richengine::model {

struct ModelDescriptor;
struct DFlashDraftLayout;
struct AffineArchHooks;

// One target architecture: its discriminator (TargetLayout::arch) and the
// loader that reads its weights from the target's files. The loader is the
// family's load*Weights entry; returning the TargetWeights variant lets
// DiffusionGemma's derived weights travel with the rest.
struct TargetArchRegistration {
  std::string_view arch;
  TargetWeights (*loadTarget)(metal::MetalBackend &backend,
                              const TargetLayout &layout,
                              const TargetFiles &files);
  // The family's affine source hooks (model/AffineTarget.hpp): its image
  // planner and config.json validator, null where the default GDN schema
  // and no validation apply.
  const AffineArchHooks *affineHooks = nullptr;
};

// One kModelFamilies row: the name an assembly's model.json states, the
// model_type its upstream config declares (legacyType an early release's
// alternate, empty for none), the descriptor maker and the family's draft
// layouts — draftLayout the DFlash2 draft, dflashV1Layout the plain draft,
// either null where the family has none.
struct ModelFamilyRegistration {
  std::string_view name;
  std::string_view modelType;
  std::string_view legacyType;
  ModelDescriptor (*makeDescriptor)(std::string name);
  DFlashDraftLayout (*draftLayout)();
  DFlashDraftLayout (*dflashV1Layout)();
};

// Both return true so a family's translation unit can register at static
// init (`const bool registered = register...`). Register before lookup: the
// lookups expose the registrations' storage.
bool registerTargetArch(TargetArchRegistration registration);
bool registerModelFamily(ModelFamilyRegistration registration);

[[nodiscard]] const TargetArchRegistration *
findTargetArch(std::string_view arch);
// The affine hooks the layout's arch registered, null for an unregistered
// arch or one that registered none.
[[nodiscard]] const AffineArchHooks *affineHooksFor(const TargetLayout &layout);
[[nodiscard]] const ModelFamilyRegistration *
findModelFamily(std::string_view name);
// The family whose config states `modelType` (or whose legacyType does).
[[nodiscard]] const ModelFamilyRegistration *
findModelFamilyByType(std::string_view modelType);
[[nodiscard]] std::span<const ModelFamilyRegistration> modelFamilies();

} // namespace richengine::model
