#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace splash::model {

// A raw row-major tensor handed to or received from a CoreML model.
enum class AneDType : uint8_t { Float16, Int32, Float32 };

struct AneTensor final {
  std::string name;
  AneDType type;
  std::vector<int64_t> shape;
  void *data;
};

// Wraps one CoreML model configured for the Neural Engine. Prediction runs
// synchronously inside predict(); submit() queues work on a serial queue so
// callers can overlap ANE execution with GPU encoding.
class AnePredictor final {
public:
  // Loads a .mlpackage, .mlmodelc or .mlmodel path. Returns nullptr and fills
  // `error` when the model cannot be compiled or loaded.
  static std::unique_ptr<AnePredictor> load(std::string_view path,
                                            std::string &error);
  ~AnePredictor();
  AnePredictor(const AnePredictor &) = delete;
  AnePredictor &operator=(const AnePredictor &) = delete;

  // Copies each input tensor into the matching named model input, runs the
  // model, and copies each named output back into the caller's buffers.
  // Returns false on shape/type mismatch or model failure.
  bool predict(std::span<const AneTensor> inputs, std::span<AneTensor> outputs,
               std::string &error);

  // Queues `job` on the predictor's serial queue; at most one job runs.
  void submit(std::function<void()> job);

private:
  struct Impl;
  explicit AnePredictor(Impl *impl);
  Impl *impl_;
};

} // namespace splash::model
