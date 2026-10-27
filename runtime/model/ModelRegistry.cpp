// The registry storage: function-local vectors so the families' static-init
// registration never races an init order. Registrations all land before the
// first lookup — the family translation units register in their headers'
// statics, and loading begins after main() starts.
#include "model/ModelRegistry.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace richengine::model {
namespace {

std::vector<TargetArchRegistration> &targetArchs() {
  static std::vector<TargetArchRegistration> registrations;
  return registrations;
}

std::vector<ModelFamilyRegistration> &families() {
  static std::vector<ModelFamilyRegistration> registrations;
  return registrations;
}

} // namespace

bool registerTargetArch(TargetArchRegistration registration) {
  // Idempotent on the pair: several family files may register the arch they
  // share, but a second loader under one name is a preset bug.
  for (const TargetArchRegistration &existing : targetArchs())
    if (existing.arch == registration.arch) {
      if (existing.loadTarget == registration.loadTarget &&
          existing.affineHooks == registration.affineHooks &&
          existing.makeEncoder == registration.makeEncoder)
        return true;
      throw std::invalid_argument("duplicate target arch " +
                                  std::string(registration.arch));
    }
  targetArchs().push_back(registration);
  return true;
}

bool registerModelFamily(ModelFamilyRegistration registration) {
  // Family names are the manifest's lookup key; model types repeat
  // legitimately across families.
  if (findModelFamily(registration.name))
    throw std::invalid_argument("duplicate model family " +
                                std::string(registration.name));
  families().push_back(registration);
  return true;
}

const TargetArchRegistration *findTargetArch(std::string_view arch) {
  for (const TargetArchRegistration &registration : targetArchs())
    if (registration.arch == arch) return &registration;
  return nullptr;
}

const AffineArchHooks *affineHooksFor(const TargetLayout &layout) {
  const TargetArchRegistration *registration = findTargetArch(layout.arch);
  return registration ? registration->affineHooks : nullptr;
}

const ModelFamilyRegistration *findModelFamily(std::string_view name) {
  for (const ModelFamilyRegistration &registration : families())
    if (registration.name == name) return &registration;
  return nullptr;
}

const ModelFamilyRegistration *
findModelFamilyByType(std::string_view modelType) {
  for (const ModelFamilyRegistration &registration : families())
    if (registration.modelType == modelType) return &registration;
  for (const ModelFamilyRegistration &registration : families())
    if (!registration.legacyType.empty() && registration.legacyType == modelType)
      return &registration;
  return nullptr;
}

std::span<const ModelFamilyRegistration> modelFamilies() {
  return families();
}

} // namespace richengine::model
