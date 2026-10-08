#pragma once

#include "StderrLine.hpp"

#include <sstream>
#include <string>

namespace richengine::engine {

// A notice while the runtime starts, as the server's print_status writes
// its own (server/diagnostics.py) and logLine writes the later ones: the
// local time, then the parts on one line, bounded, with control characters
// — such as those of a caught exception's message — as spaces.
template <typename... Parts>
void logStartup(const Parts &...parts) noexcept {
  try {
    std::ostringstream text;
    (text << ... << parts);
    writeLogLine("", text.str());
  } catch (...) {
    // Optional diagnostics must not affect startup or serving.
  }
}

} // namespace richengine::engine
