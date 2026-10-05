#pragma once

#include "engine/wire/Protocol.hpp"

#include <string>

namespace richengine::test {

// What a ready runtime's status provider returns, at the current schema.
inline std::string readyStatusJson() {
  return "{\"schema_version\":" +
         std::to_string(protocol::kStatusSchemaVersion) + ",\"ready\":true}";
}

} // namespace richengine::test
