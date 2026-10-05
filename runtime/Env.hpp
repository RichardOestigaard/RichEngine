#pragma once

// SPLASH_* environment flag and value parsing. Two flag conventions exist in
// the tree and both are preserved: envFlag treats any set value as on,
// envFlagOn requires exactly "1".

#include <cstdlib>
#include <string>

namespace splash {

// Set to anything: presence alone enables (SPLASH_ICB_OFF, SPLASH_MTL4,
// SPLASH_NO_FUSED_GATE, the *_PACKED_* switches).
[[nodiscard]] inline bool envFlag(const char *name) {
  return std::getenv(name) != nullptr;
}

// Set exactly to "1" (SPLASH_VERIFY_TREE, SPLASH_ADAPTIVE_PROPOSALS,
// SPLASH_NGRAM_PREDRAFT).
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

} // namespace splash
