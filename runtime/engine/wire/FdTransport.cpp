#include "engine/wire/FdTransport.hpp"
#include "AwakeClock.hpp"
#include "TestConfig.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <mutex>
#include <optional>
#include <poll.h>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <unistd.h>
#include <unordered_set>
#include <utility>
#include <vector>

namespace richengine::engine {
namespace {

class RestoreFdFlags final {
public:
  RestoreFdFlags(int fd, int flags) : fd_(fd), flags_(flags) {}
  ~RestoreFdFlags() { static_cast<void>(fcntl(fd_, F_SETFL, flags_)); }

  RestoreFdFlags(const RestoreFdFlags &) = delete;
  RestoreFdFlags &operator=(const RestoreFdFlags &) = delete;

private:
  int fd_;
  int flags_;
};

[[noreturn]] void throwIo(const char *operation) {
  throw std::system_error(errno, std::generic_category(), operation);
}

int pollTimeout(const NativeRuntime &loop) {
  auto delay = loop.millisecondsUntilNextWakeup();
  int timeout = -1;
  if (delay) {
    if (*delay <= 0.0)
      timeout = 0;
    else if (*delay >= double(INT_MAX))
      timeout = INT_MAX;
    else
      timeout = static_cast<int>(std::ceil(*delay));
  }
  return timeout;
}

NativeProcessExit loopFailure(const NativeRuntime &loop) {
  return loop.engineHealthy() ? NativeProcessExit::ProtocolFailure
                              : NativeProcessExit::EngineFailure;
}

uint16_t loadU16(const uint8_t *bytes) {
  return static_cast<uint16_t>(bytes[0]) |
         (static_cast<uint16_t>(bytes[1]) << 8);
}

uint32_t loadU32(const uint8_t *bytes) {
  return static_cast<uint32_t>(bytes[0]) |
         (static_cast<uint32_t>(bytes[1]) << 8) |
         (static_cast<uint32_t>(bytes[2]) << 16) |
         (static_cast<uint32_t>(bytes[3]) << 24);
}

uint64_t loadU64(const uint8_t *bytes) {
  uint64_t result = 0;
  for (uint32_t index = 0; index < 8; ++index) {
    result |= static_cast<uint64_t>(bytes[index]) << (index * 8);
  }
  return result;
}

bool clientFrameType(uint16_t raw) {
  switch (static_cast<protocol::FrameType>(raw)) {
  case protocol::FrameType::Request:
  case protocol::FrameType::Cancel:
  case protocol::FrameType::MaskResponse:
  case protocol::FrameType::StatusRequest:
  case protocol::FrameType::ForcedTokens:
  case protocol::FrameType::ConstraintDfa:
  case protocol::FrameType::Vocabulary:
  case protocol::FrameType::ScoreBatch:
  case protocol::FrameType::CorpusAppend:
    return true;
  default:
    return false;
  }
}

// A client frame whose payload opens with the request id it concerns. A
// mask or cancel may not move ahead of an earlier frame about the same
// request — a cancel must not pass the request it ends, nor a mask the
// constraint traffic that produced it — while frames about anything else
// may be passed safely.
bool scopedFrameType(protocol::FrameType type) noexcept {
  switch (type) {
  case protocol::FrameType::Request:
  case protocol::FrameType::Cancel:
  case protocol::FrameType::MaskResponse:
  case protocol::FrameType::ForcedTokens:
  case protocol::FrameType::ConstraintDfa:
    return true;
  default:
    return false;
  }
}

// Mirrors the client stream's framing a take ahead of NativeRuntime's
// parser, reading only the 24-byte headers — plus a scoped frame's request
// id, its first payload word — so the drain can serve complete MaskResponse
// and Cancel frames ahead of the request frames queued beside them. A take
// that opens mid-frame is passed through untouched: the parser is owed that
// frame's continuation first. A header that fails the wire's own checks
// breaks the mirror for good; the parser is about to fail the stream the
// same way, and order inside a dying stream no longer matters.
class FrameScan final {
public:
  // One complete frame in the take being scanned, its offsets relative to
  // the take's chunks concatenated.
  struct Located final {
    protocol::FrameType type;
    uint64_t requestId;
    bool scoped;
    size_t begin;
    size_t end;
  };

  // Starts a take. True while the stream rests on a frame boundary — the
  // only moment the take's frames can be served out of order.
  [[nodiscard]] bool beginTake() const noexcept {
    return !broken_ && headerBytes_ == 0 && !readingPayload_;
  }

  // Consumes one chunk of the take; base is the chunk's offset inside it.
  // Complete frames land in located in wire order.
  void feed(std::span<const uint8_t> bytes, size_t base,
            std::vector<Located> &located) {
    if (broken_)
      return;
    size_t pos = 0;
    while (pos < bytes.size()) {
      if (!readingPayload_) {
        if (headerBytes_ == 0)
          frameBegin_ = base + pos;
        const size_t count =
            std::min(protocol::kFrameHeaderBytes - headerBytes_,
                     bytes.size() - pos);
        std::memcpy(header_.data() + headerBytes_, bytes.data() + pos, count);
        headerBytes_ += count;
        pos += count;
        if (headerBytes_ < protocol::kFrameHeaderBytes)
          continue;
        if (!parseHeader()) {
          broken_ = true;
          return;
        }
      }
      if (idBytesRead_ < idBytesNeeded_) {
        const size_t count =
            std::min(idBytesNeeded_ - idBytesRead_, bytes.size() - pos);
        std::memcpy(idBytes_.data() + idBytesRead_, bytes.data() + pos,
                    count);
        idBytesRead_ += count;
        skip_ -= count;
        pos += count;
        if (idBytesRead_ == idBytesNeeded_)
          requestId_ = loadU64(idBytes_.data());
      }
      // count may be zero at a chunk boundary: a frame whose payload just
      // ran out is still emitted now, with its end on the boundary.
      const size_t count =
          static_cast<size_t>(std::min<uint64_t>(skip_, bytes.size() - pos));
      skip_ -= count;
      pos += count;
      if (skip_ == 0) {
        located.push_back(
            {type_, requestId_, scoped_, frameBegin_, base + pos});
        resetFrame();
      }
    }
  }

private:
  bool parseHeader() noexcept {
    static constexpr std::array<uint8_t, 4> kMagic{'R', 'I', 'C', 'H'};
    if (!std::equal(kMagic.begin(), kMagic.end(), header_.begin()))
      return false;
    if (loadU16(header_.data() + 4) != protocol::kProtocolVersion)
      return false;
    if (loadU16(header_.data() + 6) != protocol::kFrameHeaderBytes)
      return false;
    const uint16_t raw = loadU16(header_.data() + 8);
    if (!clientFrameType(raw) || loadU16(header_.data() + 10) ||
        loadU32(header_.data() + 20))
      return false;
    type_ = static_cast<protocol::FrameType>(raw);
    skip_ = loadU64(header_.data() + 12);
    if (skip_ > protocol::kAbsoluteMaxFramePayloadBytes)
      return false;
    scoped_ = scopedFrameType(type_);
    // Every request-scoped payload opens with its u64 request id.
    if (scoped_ && skip_ < 8)
      return false;
    idBytesNeeded_ = scoped_ ? 8 : 0;
    idBytesRead_ = 0;
    requestId_ = 0;
    // A zero-length payload fails the parser's per-type bound, but the
    // frame boundary still lands here, so the mirror stays aligned.
    readingPayload_ = skip_ != 0;
    return true;
  }

  void resetFrame() noexcept {
    headerBytes_ = 0;
    readingPayload_ = false;
    skip_ = 0;
    idBytesRead_ = 0;
    idBytesNeeded_ = 0;
    requestId_ = 0;
    scoped_ = false;
  }

  std::array<uint8_t, protocol::kFrameHeaderBytes> header_{};
  size_t headerBytes_ = 0;
  bool readingPayload_ = false;
  protocol::FrameType type_ = protocol::FrameType::Request;
  uint64_t skip_ = 0;
  std::array<uint8_t, 8> idBytes_{};
  size_t idBytesRead_ = 0;
  size_t idBytesNeeded_ = 0;
  uint64_t requestId_ = 0;
  bool scoped_ = false;
  size_t frameBegin_ = 0;
  bool broken_ = false;
};

// Feeds one take to the loop. A burst of request frames fully parsed and
// admitted before control and tick would delay the mask responses and
// cancels queued behind it — mask waits run five seconds, the first two
// minutes — so complete MaskResponse and Cancel frames serve first, then
// the rest. The move is a stable partition: each class keeps wire order,
// and a priority frame never passes an earlier frame scoped to the same
// request (a cancel must not beat the request it ends) nor a ScoreBatch,
// whose item ids stay opaque to the scan.
bool deliverInput(NativeRuntime &loop, FrameScan &scan,
                  const std::deque<std::vector<uint8_t>> &chunks) {
  const bool reorder = scan.beginTake();
  std::vector<FrameScan::Located> located;
  size_t base = 0;
  for (const std::vector<uint8_t> &chunk : chunks) {
    scan.feed(chunk, base, located);
    base += chunk.size();
  }
  auto inOrder = [&] {
    for (const std::vector<uint8_t> &chunk : chunks)
      if (!loop.receive(chunk))
        return false;
    return true;
  };
  if (!reorder || located.size() < 2)
    return inOrder();
  std::unordered_set<uint64_t> seenIds;
  bool opaque = false;
  std::vector<size_t> first;
  std::vector<size_t> rest;
  for (size_t index = 0; index < located.size(); ++index) {
    const FrameScan::Located &frame = located[index];
    const bool priority =
        (frame.type == protocol::FrameType::MaskResponse ||
         frame.type == protocol::FrameType::Cancel) &&
        !opaque && !(frame.scoped && seenIds.contains(frame.requestId));
    (priority ? first : rest).push_back(index);
    if (frame.scoped)
      seenIds.insert(frame.requestId);
    if (frame.type == protocol::FrameType::ScoreBatch)
      opaque = true;
  }
  // An already-partitioned take — the priority frames are its leading
  // frames — is not worth a copy of its bytes.
  if (first.empty() || first.back() + 1 == first.size())
    return inOrder();
  std::vector<uint8_t> all;
  all.reserve(base);
  for (const std::vector<uint8_t> &chunk : chunks)
    all.insert(all.end(), chunk.begin(), chunk.end());
  std::vector<uint8_t> ahead;
  std::vector<uint8_t> behind;
  ahead.reserve(base);
  behind.reserve(base);
  for (size_t index : first)
    ahead.insert(ahead.end(), all.begin() + located[index].begin,
                 all.begin() + located[index].end);
  for (size_t index : rest)
    behind.insert(behind.end(), all.begin() + located[index].begin,
                  all.begin() + located[index].end);
  // The take's tail — a frame still arriving — keeps its place at the end.
  if (!located.empty())
    behind.insert(behind.end(), all.begin() + located.back().end, all.end());
  return loop.receive(ahead) && loop.receive(behind);
}

} // namespace

struct FdTransport::LoopWake {
  int readFd;
  int writeFd;
  std::atomic<bool> controlPending{false};
  std::atomic<bool> shutdownRequested{false};

  LoopWake() {
    int descriptors[2];
    if (pipe(descriptors) < 0)
      throwIo("pipe(loop wake)");
    readFd = descriptors[0];
    writeFd = descriptors[1];
    auto makeNonBlocking = [&](int fd) {
      int flags = fcntl(fd, F_GETFL);
      if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) {
        int saved = errno;
        close(readFd);
        close(writeFd);
        errno = saved;
        throwIo("fcntl(loop wake)");
      }
    };
    makeNonBlocking(readFd);
    makeNonBlocking(writeFd);
  }

  ~LoopWake() {
    close(readFd);
    close(writeFd);
  }

  void notify() const noexcept {
    constexpr uint8_t byte = 1;
    while (true) {
      ssize_t written = write(writeFd, &byte, sizeof(byte));
      if (written > 0 ||
          (written < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))) {
        return;
      }
      if (written < 0 && errno == EINTR)
        continue;
      return;
    }
  }

  void notifyControl() noexcept {
    controlPending.store(true, std::memory_order_release);
    notify();
  }

  bool takeControl() noexcept {
    return controlPending.exchange(false, std::memory_order_acq_rel);
  }

  void drain() const noexcept {
    std::array<uint8_t, 64> bytes{};
    while (true) {
      ssize_t count = read(readFd, bytes.data(), bytes.size());
      if (count > 0)
        continue;
      if (count < 0 && errno == EINTR)
        continue;
      return;
    }
  }
};

// Reads the input on a thread of its own and queues it for the loop, which
// takes it between ticks. The bytes, then the input's end or a read error,
// reach the loop in the order they were read. Destruction stops the thread,
// whether it waits for input or for room in the queue, and joins it.
class FdTransport::InputReader final {
public:
  struct Input final {
    // What each read returned, in order.
    std::deque<std::vector<uint8_t>> chunks;
    // Set once the input has ended after these chunks: 0 at its end,
    // otherwise the error that ended reading.
    std::optional<int> end;
  };

  InputReader(int fd, size_t queueBytes, std::shared_ptr<LoopWake> wake)
      : fd_(fd), queueBytes_(queueBytes), wake_(std::move(wake)) {
    int descriptors[2];
    if (pipe(descriptors) < 0)
      throwIo("pipe(input reader stop)");
    stopRead_ = descriptors[0];
    stopWrite_ = descriptors[1];
    try {
      thread_ = std::thread([this] { read(); });
    } catch (...) {
      close(stopRead_);
      close(stopWrite_);
      throw;
    }
  }

  ~InputReader() {
    {
      std::lock_guard lock(mutex_);
      stopping_ = true;
    }
    room_.notify_all();
    constexpr uint8_t byte = 1;
    while (write(stopWrite_, &byte, sizeof(byte)) < 0 && errno == EINTR) {
    }
    thread_.join();
    close(stopRead_);
    close(stopWrite_);
  }

  InputReader(const InputReader &) = delete;
  InputReader &operator=(const InputReader &) = delete;

  // Everything read since the last call, with the input's end once nothing
  // read before it is left.
  [[nodiscard]] Input take() {
    Input input;
    {
      std::lock_guard lock(mutex_);
      input.chunks.swap(chunks_);
      queuedBytes_ = 0;
      input.end = end_;
    }
    room_.notify_all();
    return input;
  }

private:
  void read() noexcept {
    try {
      std::vector<uint8_t> buffer(64 * 1024);
      while (true) {
        size_t room = 0;
        {
          std::unique_lock lock(mutex_);
          room_.wait(lock, [&] { return stopping_ || queuedBytes_ < queueBytes_; });
          if (stopping_)
            return;
          room = queueBytes_ - queuedBytes_;
        }
        std::array<pollfd, 2> descriptors{pollfd{fd_, POLLIN, 0},
                                          pollfd{stopRead_, POLLIN, 0}};
        if (poll(descriptors.data(), descriptors.size(), -1) < 0) {
          if (errno == EINTR)
            continue;
          return finish(errno);
        }
        if (descriptors[1].revents)
          return;
        if (descriptors[0].revents & POLLNVAL)
          return finish(EBADF);
        if (!descriptors[0].revents)
          continue;
        // Only take() changes the room meanwhile, and only to more, so the
        // queue never passes its bound.
        const ssize_t count =
            ::read(fd_, buffer.data(), std::min(buffer.size(), room));
        if (count > 0) {
          std::vector<uint8_t> chunk(buffer.begin(), buffer.begin() + count);
          {
            std::lock_guard lock(mutex_);
            queuedBytes_ += chunk.size();
            chunks_.push_back(std::move(chunk));
          }
          wake_->notify();
          continue;
        }
        if (count == 0)
          return finish(0);
        if (errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK)
          return finish(errno);
      }
    } catch (const std::bad_alloc &) {
      finish(ENOMEM);
    }
  }

  void finish(int error) noexcept {
    {
      std::lock_guard lock(mutex_);
      end_ = error;
    }
    wake_->notify();
  }

  int fd_ = -1;
  size_t queueBytes_ = 0;
  std::shared_ptr<LoopWake> wake_;
  int stopRead_ = -1;
  int stopWrite_ = -1;
  std::mutex mutex_;
  std::condition_variable room_;
  std::deque<std::vector<uint8_t>> chunks_;
  size_t queuedBytes_ = 0;
  std::optional<int> end_;
  bool stopping_ = false;
  std::thread thread_;
};

FdTransport::FdTransport(int inputFd, int outputFd)
    : inputFd_(inputFd), outputFd_(outputFd),
      inputQueueBytes_(
          testConfig().transportInputQueueBytes.value_or(kInputQueueBytes)),
      wake_(std::make_shared<LoopWake>()) {
  if (inputFd_ < 0 || outputFd_ < 0) {
    throw std::invalid_argument("native transport requires valid fds");
  }
}

NativeRuntime::ByteSink FdTransport::outputSink() {
  return [this](std::span<const uint8_t> bytes) { writeAll(bytes); };
}

std::function<void()> FdTransport::controlNotifier() {
  std::shared_ptr<LoopWake> wake = wake_;
  return [wake] { wake->notifyControl(); };
}

void FdTransport::setControlHandler(ControlHandler handler) {
  controlHandler_ = std::move(handler);
}

NativeProcessExit FdTransport::run(NativeRuntime &loop) {
  std::shared_ptr<LoopWake> wake = wake_;
  loop.setCompletionNotifier([wake] { wake->notify(); });
  // Nonblocking so that a read after a spurious readiness returns EAGAIN and
  // the reader goes back to poll, where its stop pipe can end it; without it
  // ~InputReader could wait forever in read().
  int originalFlags = fcntl(inputFd_, F_GETFL);
  if (originalFlags < 0)
    throwIo("fcntl(F_GETFL)");
  if (fcntl(inputFd_, F_SETFL, originalFlags | O_NONBLOCK) < 0) {
    throwIo("fcntl(F_SETFL)");
  }
  RestoreFdFlags restore(inputFd_, originalFlags);
  // Stopped and joined on every way out of run(), before the flags return.
  InputReader reader(inputFd_, inputQueueBytes_, wake);
  FrameScan scan;
  bool deferredControl = false;

  while (!loop.connectionMustClose()) {
    if (shutdownRequested())
      return NativeProcessExit::CleanEof;
    {
      InputReader::Input input = reader.take();
      if (!deliverInput(loop, scan, input.chunks))
        return loopFailure(loop);
      if (input.end) {
        if (*input.end == ENOMEM) {
          failure_ = "the native input reader ran out of memory";
          return NativeProcessExit::EngineFailure;
        }
        if (*input.end) {
          failure_ = "read(native input): " +
                     std::string(std::strerror(*input.end));
          return NativeProcessExit::IoFailure;
        }
        return loop.finishInput() ? NativeProcessExit::CleanEof
                                  : loopFailure(loop);
      }
    } // What was taken is freed before control, tick and poll.

    const auto stepStarted = AwakeClock::now();
    deferredControl = wake->takeControl() || deferredControl;
    if (deferredControl && !loop.commandInFlight()) {
      deferredControl = loop.runControl(controlHandler_);
    }

    const bool progressed = loop.tick();
    const double tickMilliseconds = millisecondsSince(stepStarted);
    if (tickMilliseconds > maxTickMilliseconds_)
      maxTickMilliseconds_ = tickMilliseconds;
    if (progressed)
      continue;
    if (loop.connectionMustClose())
      return loopFailure(loop);

    // Input, completions, control notifications and shutdown all wake the
    // loop through this pipe.
    pollfd descriptor{wake->readFd, POLLIN, 0};
    int result;
    do {
      result = poll(&descriptor, 1, pollTimeout(loop));
    } while (result < 0 && errno == EINTR);
    if (result < 0) {
      failure_ = "poll(loop wake): " + std::string(std::strerror(errno));
      return NativeProcessExit::IoFailure;
    }
    if (descriptor.revents & POLLIN)
      wake->drain();
    // A zero result is a deadline, health-check or admission-retry wake; the
    // next iteration takes input, runs control and ticks.
  }
  return loopFailure(loop);
}

bool FdTransport::runFlush(NativeRuntime &loop, std::chrono::milliseconds budget) {
  const auto deadline = AwakeClock::now() + budget;
  while (!loop.flushRestorePoints()) {
    const auto left = std::chrono::ceil<std::chrono::milliseconds>(
        deadline - AwakeClock::now());
    if (!loop.engineHealthy() || left.count() <= 0)
      return false;
    pollfd descriptor{wake_->readFd, POLLIN, 0};
    const int result = poll(&descriptor, 1, static_cast<int>(left.count()));
    if (result < 0 && errno != EINTR) {
      failure_ = "poll(loop wake): " + std::string(std::strerror(errno));
      return false;
    }
    if (result > 0 && (descriptor.revents & POLLIN))
      wake_->drain();
  }
  return true;
}

bool FdTransport::shutdownRequested() const noexcept {
  return wake_->shutdownRequested.load(std::memory_order_acquire);
}

double FdTransport::maxTickMilliseconds() const noexcept {
  return maxTickMilliseconds_;
}

const std::string &FdTransport::failure() const noexcept { return failure_; }

void FdTransport::requestShutdown() noexcept {
  wake_->shutdownRequested.store(true, std::memory_order_release);
  wake_->notify();
}

void FdTransport::writeAll(std::span<const uint8_t> bytes) const {
  size_t offset = 0;
  while (offset < bytes.size()) {
    ssize_t count =
        write(outputFd_, bytes.data() + offset, bytes.size() - offset);
    if (count > 0) {
      offset += static_cast<size_t>(count);
      continue;
    }
    // The signal that requests a shutdown also ends a write the server has
    // stopped draining.
    if (count < 0 && errno == EINTR && !shutdownRequested())
      continue;
    if (count == 0) {
      throw std::runtime_error("native output accepted zero bytes");
    }
    throwIo("write(native output)");
  }
}

} // namespace richengine::engine
