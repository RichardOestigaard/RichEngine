#include "model/GgufTrellis.hpp"

#include "Checked.hpp"

#include <cmath>
#include <limits>
#include <optional>
#include <span>

namespace richengine::model {
namespace {

constexpr bool powerOfTwo(int32_t value) noexcept {
  return value > 0 && (value & (value - 1)) == 0;
}

int32_t integerValue(double value, std::string_view key) {
  if (!std::isfinite(value) || value != std::floor(value) ||
      value < std::numeric_limits<int32_t>::min() || value > std::numeric_limits<int32_t>::max())
    throw GgufError("trellis descriptor is not an int32 array: " + std::string(key));
  return static_cast<int32_t>(value);
}

void validateSide(int32_t length, int32_t kind, int32_t block, std::string_view name) {
  if (kind == 0) return;
  if (kind != 1 || !powerOfTwo(block) || block < 2 || length % block)
    throw GgufError("unsupported trellis transform for " + std::string(name));
}

void validateVector(const GgufTensor *tensor, uint32_t type, uint64_t elements, std::string_view name) {
  if (!tensor) throw GgufError("missing trellis side tensor: " + std::string(name));
  if (tensor->type != type || tensor->rows() != 1 || tensor->columns() != elements)
    throw GgufError("invalid trellis side tensor: " + std::string(name));
}

void validateGather(const GgufTensor *tensor, uint64_t elements, std::string_view name) {
  if (tensor && (tensor->type != ggml::kI32 || tensor->rows() != 1 || tensor->columns() != elements))
    throw GgufError("invalid trellis gather tensor: " + std::string(name));
}

} // namespace

uint32_t TrellisTensorInfo::bits() const noexcept {
  return weight ? ggmlTrellisBits(weight->type) : 0;
}
uint32_t TrellisTensorInfo::inputSize() const noexcept {
  return weight ? static_cast<uint32_t>(weight->columns()) : 0;
}
uint32_t TrellisTensorInfo::outputSize() const noexcept {
  return weight ? static_cast<uint32_t>(weight->rows()) : 0;
}

bool ggufHasTrellis(const GgufFile &file) {
  return file.unsignedValue("trellis.version").has_value();
}

uint32_t ggufTrellisBandCount(const GgufFile &file, std::string_view key) {
  const std::optional<uint64_t> count = file.unsignedValue(key);
  if (!count) return 0;
  if (!*count || *count > 256) throw GgufError("invalid trellis band count: " + std::string(key));
  return static_cast<uint32_t>(*count);
}

TrellisDescriptor readTrellisDescriptor(const GgufFile &file, const GgufTensor &weight) {
  if (file.unsignedValue("trellis.version") != 1)
    throw GgufError("unsupported or missing trellis.version");
  std::string base = weight.name;
  if (base.ends_with(".weight")) base.resize(base.size() - 7);
  const std::string key = "trellis.desc." + base;
  const std::optional<std::span<const double>> values = file.numericArray(key);
  if (!values) throw GgufError("missing trellis descriptor: " + weight.name);
  if (values->size() != 8) throw GgufError("unsupported trellis descriptor length: " + weight.name);
  TrellisDescriptor d;
  d.rowsKind = integerValue((*values)[0], key);
  d.rowsBlock = integerValue((*values)[1], key);
  d.rowsShards = integerValue((*values)[2], key);
  d.colsKind = integerValue((*values)[3], key);
  d.colsBlock = integerValue((*values)[4], key);
  d.colsShards = integerValue((*values)[5], key);
  d.parts = integerValue((*values)[6], key);
  d.part0 = integerValue((*values)[7], key);
  if (weight.dims.size() != 2 || weight.columns() > std::numeric_limits<uint32_t>::max() ||
      weight.rows() > std::numeric_limits<uint32_t>::max() || weight.columns() % 16 || weight.rows() % 16)
    throw GgufError("invalid trellis tensor shape: " + weight.name);
  const auto bits = ggmlTrellisBits(weight.type);
  if (bits < 2) throw GgufError("unsupported trellis type for " + weight.name + ": " + ggmlTypeName(weight.type));
  if (d.rowsShards != 1 || d.colsShards != 1 || d.parts != 1 || d.part0 != static_cast<int32_t>(weight.rows()))
    throw GgufError("unsupported trellis transform partitioning: " + weight.name);
  validateSide(static_cast<int32_t>(weight.rows()), d.rowsKind, d.rowsBlock, weight.name);
  validateSide(static_cast<int32_t>(weight.columns()), d.colsKind, d.colsBlock, weight.name);
  return d;
}

TrellisTensorInfo readTrellisTensor(const GgufFile &file, std::string_view name) {
  const GgufTensor &weight = file.require(name);
  if (!ggmlTrellisBits(weight.type)) throw GgufError("not a trellis tensor: " + std::string(name));
  TrellisTensorInfo info;
  info.weight = &weight;
  info.descriptor = readTrellisDescriptor(file, weight);
  std::string base(name);
  if (base.ends_with(".weight")) base.resize(base.size() - 7);
  info.s16 = file.find(base + ".trellis_s16");
  info.pre = file.find(base + ".trellis_pre");
  info.post = file.find(base + ".trellis_post");
  info.pin = file.find(base + ".trellis_pin");
  info.pout = file.find(base + ".trellis_pout");
  validateVector(info.s16, 1, weight.columns() / 16, base + ".trellis_s16");
  validateVector(info.pre, ggml::kF32, weight.columns(), base + ".trellis_pre");
  validateVector(info.post, ggml::kF32, weight.rows(), base + ".trellis_post");
  validateGather(info.pin, weight.columns(), base + ".trellis_pin");
  validateGather(info.pout, weight.rows(), base + ".trellis_pout");
  return info;
}

} // namespace richengine::model
