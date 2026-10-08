#pragma once

#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <ctime>
#include <sstream>
#include <string>
#include <string_view>

namespace richengine {

// The server and this runtime write to the same stderr. Each line goes out
// in one write, newline included, so that lines written at once stay whole.
inline void writeStderrLine(std::string_view text) noexcept {
  try {
    std::string line(text);
    line += '\n';
    for (std::string_view rest = line; !rest.empty();) {
      const ssize_t written = ::write(STDERR_FILENO, rest.data(), rest.size());
      if (written < 0 && errno == EINTR)
        continue;
      if (written <= 0)
        return;
      rest.remove_prefix(static_cast<size_t>(written));
    }
  } catch (...) {
    // Diagnostics must not affect startup or serving.
  }
}

// Whether stderr colors: the launcher's gate (install/launcher.py _ansi), a
// terminal with NO_COLOR unset, so piped output and logs stay byte-plain.
inline bool stderrAnsi() noexcept {
  static const bool ansi = ::isatty(STDERR_FILENO) && !std::getenv("NO_COLOR");
  return ansi;
}

// The bytes a styled part writes around its text: kStyleMark, the SGR
// parameters and kStyleEnd. writeLogLine writes them as SGR on a terminal;
// a control byte a message otherwise carries, such as one in a caught
// exception's message, is still a space.
inline constexpr char kStyleMark = '\x01';
inline constexpr char kStyleEnd = '\x02';

// A part of a log line in the help palette (install/launcher.py): dim is its
// 'Usage' gray for detail the operator can skip, accent its cyan for names
// and flags they may reuse. Off a terminal the text alone is written.
template <typename Part> struct Styled {
  const Part &part;
  const char *sgr;
};

template <typename Part> Styled<Part> dim(const Part &part) noexcept {
  return {part, "2"};
}

template <typename Part> Styled<Part> accent(const Part &part) noexcept {
  return {part, "36"};
}

template <typename Part>
std::ostream &operator<<(std::ostream &out, const Styled<Part> &styled) {
  if (stderrAnsi()) out << kStyleMark << styled.sgr << kStyleEnd;
  out << styled.part;
  if (stderrAnsi()) out << kStyleMark << '0' << kStyleEnd;
  return out;
}

// The parts joined, sanitized and timestamped. prefix is written verbatim
// before them: it carries a status word's color, which the sanitizing pass
// would strip as a control byte.
inline void writeLogLine(std::string_view prefix, std::string_view message) noexcept {
  try {
    const std::time_t now = std::time(nullptr);
    std::tm local{};
    char timestamp[9] = "--:--:--";
    if (localtime_r(&now, &local))
      std::strftime(timestamp, sizeof(timestamp), "%H:%M:%S", &local);
    std::ostringstream line;
    if (stderrAnsi()) line << "\x1b[2m" << timestamp << "\x1b[0m";
    else line << timestamp;
    line << ' ' << prefix;
    const std::string_view bounded = message.substr(0, 768);
    for (size_t at = 0; at < bounded.size(); ++at) {
      const unsigned char character = bounded[at];
      if (character == static_cast<unsigned char>(kStyleMark)) {
        // An inserted style mark becomes SGR; a stray or truncated one is a
        // space like every other control byte.
        const size_t end = bounded.find(kStyleEnd, at + 1);
        const std::string_view sgr =
            end == std::string_view::npos ? "" : bounded.substr(at + 1, end - at - 1);
        if (!sgr.empty() && sgr.size() <= 11 &&
            sgr.find_first_not_of("0123456789;") == std::string_view::npos) {
          line << "\x1b[" << sgr << 'm';
          at = end;
          continue;
        }
      }
      line << (character < 32 || character == 127 ? ' ' : char(character));
    }
    if (message.size() > 768) line << "...";
    writeStderrLine(line.str());
  } catch (...) {
    // Diagnostics must not affect startup or serving.
  }
}

// A notice the runtime gives while it starts, serves or stops, as the server
// prints its own (server/diagnostics.py print_status): the local time, then
// the parts on one line, bounded, with control characters, such as those of
// a caught exception's message, as spaces. An error that ends the process
// is written as an "error: ..." line instead (main.mm).
template <typename... Parts> void logLine(const Parts &...parts) noexcept {
  try {
    std::ostringstream text;
    (text << ... << parts);
    writeLogLine("", text.str());
  } catch (...) {
    // Diagnostics must not affect startup or serving.
  }
}

// A notice of a fault the runtime goes on serving past, in the form of the
// server's warnings (server/server.py): "Warning · " and the parts, the word
// yellow on a terminal like the server's.
template <typename... Parts> void logWarning(const Parts &...parts) noexcept {
  try {
    std::ostringstream text;
    (text << ... << parts);
    writeLogLine(stderrAnsi() ? "\x1b[33mWarning\x1b[0m · " : "Warning · ",
                 text.str());
  } catch (...) {
    // Diagnostics must not affect startup or serving.
  }
}

// A fatal line, "error: ...", the keyword red on a terminal like the
// server's Error lines.
inline void writeErrorLine(std::string_view text) noexcept {
  if (stderrAnsi()) {
    std::string line("\x1b[31merror:\x1b[0m ");
    line += text;
    writeStderrLine(line);
  } else {
    std::string line("error: ");
    line += text;
    writeStderrLine(line);
  }
}

} // namespace richengine
