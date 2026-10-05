#pragma once

// RICHENGINE_* environment flag and value parsing. Two flag conventions exist in
// the tree and both are preserved: envFlag treats any set value as on,
// envFlagOn requires exactly "1".

#include <cstdlib>
#include <string>

namespace richengine {

// Set to anything: presence alone enables (RICHENGINE_ICB_OFF, RICHENGINE_MTL4,
// RICHENGINE_NO_FUSED_GATE, the *_PACKED_* switches).
[[nodiscard]] inline bool envFlag(const char *name) {
  return std::getenv(name) != nullptr;
}

// Set exactly to "1" (RICHENGINE_VERIFY_TREE, RICHENGINE_ADAPTIVE_PROPOSALS,
// RICHENGINE_NGRAM_PREDRAFT).
[[nodiscard]] inline bool envFlagOn(const char *name) {
  const char *value = std::getenv(name);
  return value && std::string(value) == "1";
}

[[nodiscard]] inline uint32_t envUint(const char *name, uint32_t fallback) {
  const char *value = std::getenv(name);
  return value ? static_cast<uint32_t>(std::atoi(value)) : fallback;
}

[[nodiscard]] inline double envDouble(const char *name, double fallback) {
  const char *value = std::getenv(name);
  return value ? std::atof(value) : fallback;
}

} // namespace richengine
