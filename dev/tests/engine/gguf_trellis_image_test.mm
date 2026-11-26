// gguf-trellis-image <richengine.metallib>: a synthetic NariQ qwen35 GGUF
// through image planning, preparation and the production target loader.
#include "GgufFixtures.hpp"
#include "model/GgufTarget.hpp"
#include "model/Qwen3_8.hpp"
#include "model/TargetLoader.hpp"
#include "model/WeightImages.hpp"

#include <cstring>
#include <random>

using namespace gguf_fixtures;
using richengine::metal::MetalBackend;
namespace model = richengine::model;
namespace ops = richengine::ops;
namespace test_gguf = richengine::test::gguf;

namespace {

constexpr uint32_t kTRL2 = 45, kTRL3 = 46;

model::gguf::TargetGeometry geometry() {
  model::gguf::TargetGeometry g;
  g.layers = 2;
  g.hiddenSize = 512;
  g.vocabularySize = 256;
  g.intermediateSize = 256;
  g.gdnKeyHeads = 4;
  g.gdnValueHeads = 16;
  g.gdnHeadDimension = 128;
  g.convolutionDimension = 3072; // q and k of 4 heads, v of 16
  g.attentionWidth = 2048;      // sixteen query heads of 128
  g.attentionKvHeads = 2;
  g.attentionHeadDimension = 128;
  g.rotaryPairs = 32;
  g.rotaryTheta = 1e7F;
  g.fullAttentionPeriod = 2;
  return g;
}

test_gguf::Bytes bytes(uint64_t count, uint32_t seed) {
  test_gguf::Bytes out(count);
  std::mt19937 rng(seed);
  for (uint8_t &byte : out) byte = uint8_t(rng());
  return out;
}

test_gguf::Bytes floats(uint64_t count, uint32_t seed) {
  test_gguf::Bytes out(count * 4);
  for (uint64_t i = 0; i < count; ++i) {
    const float value = 0.5f + float((i + seed) % 17) / 32.0f;
    std::memcpy(out.data() + 4 * i, &value, 4);
  }
  return out;
}

test_gguf::Bytes halves(uint64_t count, uint32_t seed) {
  test_gguf::Bytes out(count * 2);
  for (uint64_t i = 0; i < count; ++i) {
    const __fp16 value = __fp16(0.5f + float((i + seed) % 7) / 16.0f);
    std::memcpy(out.data() + 2 * i, &value, 2);
  }
  return out;
}

test_gguf::Bytes permutation(uint64_t count, uint32_t seed, bool identity = false) {
  std::vector<int32_t> values(count);
  for (uint64_t i = 0; i < count; ++i) values[i] = int32_t(i);
  if (!identity) {
    std::mt19937 rng(seed);
    for (uint64_t i = count; i-- > 1;) std::swap(values[i], values[rng() % (i + 1)]);
  }
  test_gguf::Bytes out(count * 4);
  std::memcpy(out.data(), values.data(), out.size());
  return out;
}

std::string baseName(std::string name) {
  if (name.ends_with(".weight")) name.resize(name.size() - 7);
  return name;
}

void addSide(std::vector<Tensor> &tensors, const std::string &base, const char *side, uint32_t type,
             uint64_t elements, test_gguf::Bytes data) {
  tensors.push_back({base + ".trellis_" + side, {elements}, type, std::move(data)});
}

void addTrellisSides(std::vector<Tensor> &tensors, std::vector<test_gguf::Key> &keys,
                     const Tensor &weight, bool pin, bool pout, uint32_t seed) {
  const std::string base = baseName(weight.name);
  const uint64_t columns = weight.dims[0], rows = weight.dims[1];
  keys.push_back(test_gguf::int32ArrayKey("trellis.desc." + base,
                                          {1, 16, 1, 1, 32, 1, 1, int32_t(rows)}));
  addSide(tensors, base, "s16", model::ggml::kF16, columns / 16, halves(columns / 16, seed + 1));
  addSide(tensors, base, "pre", model::ggml::kF32, columns, floats(columns, seed + 2));
  addSide(tensors, base, "post", model::ggml::kF32, rows, floats(rows, seed + 3));
  if (pin) addSide(tensors, base, "pin", model::ggml::kI32, columns, permutation(columns, seed + 4));
  if (pout) addSide(tensors, base, "pout", model::ggml::kI32, rows, permutation(rows, seed + 5));
}

struct TrellisTarget {
  model::gguf::TargetGeometry g = geometry();
  std::vector<Tensor> tensors;
  std::vector<test_gguf::Key> keys;
};

TrellisTarget makeTarget() {
  TrellisTarget target;
  const TensorTypes types{{"attn_q.weight", kTRL2},       {"attn_k.weight", kTRL2},
                          {"attn_v.weight", kTRL2},       {"attn_output.weight", kTRL2},
                          {"attn_qkv.weight", kTRL2},     {"attn_gate.weight", kTRL2},
                          {"ssm_beta.weight", kTRL3},     {"ssm_alpha.weight", kTRL3},
                          {"ssm_out.weight", kTRL2},      {"ffn_gate.weight", kTRL2},
                          {"ffn_up.weight", kTRL2},       {"ffn_down.weight", kTRL2},
                          {"token_embd.weight", kTRL2}};
  target.tensors = targetTensors(target.g, types);
  target.keys = metadata(target.g);
  target.keys.push_back(test_gguf::uint32Key("trellis.version", 1));
  target.keys.push_back(test_gguf::uint32Key("trellis.lm_head_bands", 2));

  std::vector<Tensor> coded;
  for (Tensor &tensor : target.tensors) {
    if (tensor.name == "output.weight") continue;
    if (model::ggmlTrellisBits(tensor.type)) {
      tensor.data = bytes(tensor.data.size(), uint32_t(3000 + coded.size()));
      coded.push_back(tensor);
    } else if (tensor.type == model::ggml::kF32) {
      tensor.data = floatValues(tensor.dims[0] * (tensor.dims.size() > 1 ? tensor.dims[1] : 1),
                                uint32_t(4000 + coded.size()), readAsBfloat16(tensor));
    }
  }
  target.tensors.erase(std::remove_if(target.tensors.begin(), target.tensors.end(),
                                      [](const Tensor &t) { return t.name == "output.weight"; }),
                       target.tensors.end());
  for (const Tensor &weight : coded) {
    const bool embedding = weight.name == "token_embd.weight";
    const bool pin = !embedding && (weight.name.find("ssm_out.weight") != std::string::npos ||
                                     weight.name.find("attn_qkv.weight") != std::string::npos);
    const bool pout = !embedding && weight.name.find("attn_qkv.weight") != std::string::npos;
    addTrellisSides(target.tensors, target.keys, weight, pin, pout,
                    uint32_t(5000 + target.tensors.size()));
  }
  // Two 128-row bands cover the 256-row padded vocabulary.
  for (uint32_t band = 0; band < 2; ++band) {
    Tensor weight{std::string("output.band0") + char('0' + band) + ".weight",
                  {target.g.hiddenSize, 128}, kTRL2,
                  bytes(uint64_t{target.g.hiddenSize} * 128 * 2 / 8, 6100 + band)};
    addTrellisSides(target.tensors, target.keys, weight, false, true, 6200 + 8 * band);
    target.tensors.push_back(std::move(weight));
  }
  target.tensors.push_back({"output.vocab_gather", {256}, model::ggml::kI32, permutation(256, 6300)});
  return target;
}

void writeTarget(const std::filesystem::path &directory, const TrellisTarget &target) {
  std::filesystem::create_directory(directory);
  richengine::test::writeFile(directory / "target.gguf", test_gguf::file(target.keys, target.tensors));
}

uint64_t groupedSource(uint64_t row, uint64_t from, uint32_t headRows, uint32_t keyHeads, uint32_t valueHeadsPerKey) {
  if (row < from) return row;
  const uint64_t head = (row - from) / headRows, within = (row - from) % headRows;
  return from + ((head % valueHeadsPerKey) * keyHeads + head / valueHeadsPerKey) * headRows + within;
}

std::vector<int32_t> ints(const test_gguf::Bytes &data) {
  std::vector<int32_t> values(data.size() / 4);
  std::memcpy(values.data(), data.data(), data.size());
  return values;
}

std::vector<int32_t> ints(const richengine::metal::MetalBuffer &buffer, uint64_t count) {
  std::vector<int32_t> values(count);
  std::memcpy(values.data(), buffer.contents(), count * 4);
  return values;
}

bool bytesEqual(const richengine::metal::MetalBuffer &buffer, const test_gguf::Bytes &expected) {
  return buffer.sizeBytes() == expected.size() && std::memcmp(buffer.contents(), expected.data(), expected.size()) == 0;
}

const Tensor &named(const std::vector<Tensor> &tensors, const std::string &name) {
  return *std::find_if(tensors.begin(), tensors.end(), [&](const Tensor &t) { return t.name == name; });
}

void checkPrepared(MetalBackend &backend, const std::filesystem::path &directory) {
  const TrellisTarget target = makeTarget();
  writeTarget(directory / "target", target);
  model::WeightImages images(backend);
  model::GgufTargetLoader loader(backend, images, model::findTargetGguf(directory / "target"), target.g);
  static_cast<void>(loader.layer(0));
  static_cast<void>(loader.layer(1));
  static_cast<void>(loader.head());
  static_cast<void>(loader.embedding());

  model::TargetLayout layout = model::qwen38TargetLayout();
  layout.layers = target.g.layers;
  layout.hiddenSize = target.g.hiddenSize;
  layout.vocabularySize = target.g.vocabularySize;
  layout.gdnKeyHeads = target.g.gdnKeyHeads;
  layout.gdnValueHeads = target.g.gdnValueHeads;
  layout.gdnHeadDimension = target.g.gdnHeadDimension;
  layout.convolutionDimension = target.g.convolutionDimension;
  layout.packedGdnWidth = 5376; // qkv 3072 | z 2048 | one alpha-beta tile
  layout.attentionWidth = target.g.attentionWidth;
  layout.attentionQueryHeads = 16;
  layout.attentionKvHeads = target.g.attentionKvHeads;
  layout.attentionHeadDimension = target.g.attentionHeadDimension;
  layout.packedFullWidth = 4608; // q and gate | k | v
  layout.intermediateSize = target.g.intermediateSize;
  layout.hiddenCaptureLayers.fill(layout.layers - 1);
  layout.fullAttentionPeriod = target.g.fullAttentionPeriod;
  model::GgufTargetLoader files(backend, images, model::findTargetGguf(directory / "target"), target.g);
  const model::Qwen3_8Weights weights = model::loadQwen3_8Weights(backend, layout, files);

  check(weights.logitsProjection.blocks().segments.size() == 2 &&
            weights.logitsProjection.blocks().segments[0].isTrellis() &&
            weights.logitsProjection.blocks().segments[0].trellisData->hasScatter,
        "trellis target: two banded-head segments with vocabulary scatter");
  check(weights.tokenEmbedding.blocks().trellis.has_value(), "trellis target: coded token embedding");

  const auto &gdn = std::get<model::GdnMixerWeights>(weights.layers[0].mixer);
  const auto &segments = gdn.inputProjection.blocks().segments;
  check(segments.size() == 4 && segments[0].outputSize == 3072 && segments[1].outputSize == 2048 &&
            segments[2].outputSize == 16 && segments[3].outputSize == 16,
        "trellis target: GDN input segments qkv | z | beta | alpha");
  check(segments[0].columnOffset == 0 && segments[1].columnOffset == 3072 &&
            segments[2].columnOffset == 5120 && segments[3].columnOffset == 5136,
        "trellis target: GDN segment column offsets");

  const Tensor &qkv = named(target.tensors, "blk.0.attn_qkv.weight");
  check(bytesEqual(segments[0].trellisData->codes, qkv.data), "trellis target: qkv coded payload copied unchanged");
  check(bytesEqual(segments[0].trellisData->pre, named(target.tensors, "blk.0.attn_qkv.trellis_pre").data),
        "trellis target: qkv pre vector copied unchanged");

  const std::vector<int32_t> original = ints(named(target.tensors, "blk.0.attn_qkv.trellis_pout").data);
  const std::vector<int32_t> composed = ints(segments[0].trellisData->pout, qkv.dims[1]);
  bool correct = composed.size() == original.size();
  for (uint64_t row = 0; correct && row < composed.size(); ++row) {
    const uint64_t source = groupedSource(row, 1024, 128, target.g.gdnKeyHeads,
                                          target.g.gdnValueHeads / target.g.gdnKeyHeads);
    correct = composed[row] == original[source];
  }
  check(correct, "trellis target: qkv output gather composed into grouped head order");

  const std::vector<int32_t> scatter = ints(segments[0].trellisData->pin, qkv.dims[0]);
  check(scatter == ints(named(target.tensors, "blk.0.attn_qkv.trellis_pin").data),
        "trellis target: qkv input gather copied unchanged");

  const std::vector<int32_t> inputGather = ints(gdn.outputProjection.blocks().segments[0].trellisData->pin,
                                                target.g.attentionWidth);
  check(inputGather == ints(named(target.tensors, "blk.0.ssm_out.trellis_pin").data),
        "trellis target: ssm_out input gather copied unchanged");

  const auto &head0 = weights.logitsProjection.blocks().segments.front();
  const std::vector<int32_t> vocab = ints(head0.trellisData->scatter, 256);
  const std::vector<int32_t> vocabGather = ints(named(target.tensors, "output.vocab_gather").data);
  std::vector<int32_t> inverseGather(256, -1);
  for (uint32_t row = 0; row < 256; ++row) inverseGather[static_cast<uint32_t>(vocabGather[row])] = int32_t(row);
  check(vocab == inverseGather, "trellis target: vocabulary gather inverted for scatter output");
}

void checkReal(MetalBackend &backend, const std::filesystem::path &path) {
  model::WeightImages images(backend);
  const model::TargetLayout layout = model::qwen38TargetLayout();
  const model::gguf::TargetGeometry geometry = model::ggufTargetGeometry(layout);
  model::GgufTargetLoader files(backend, images, path, geometry);
  const model::Qwen3_8Weights weights = model::loadQwen3_8Weights(backend, layout, files);
  check(weights.layers.size() == layout.layers, "real NariQ target: every layer loaded");
  check(weights.logitsProjection.blocks().segments.size() == 16,
        "real NariQ target: sixteen banded-head segments");
  uint64_t rows = 0;
  for (const ops::QuantizedSegment &segment : weights.logitsProjection.blocks().segments) {
    check(segment.isTrellis() && segment.trellisData->hasScatter,
          "real NariQ target: every head band is trellis with vocabulary scatter");
    rows += segment.outputSize;
  }
  check(rows == layout.vocabularySize, "real NariQ target: head bands cover the vocabulary");
  check(weights.tokenEmbedding.blocks().trellis.has_value(),
        "real NariQ target: coded token embedding");
  uint32_t gdnLayers = 0, fullLayers = 0;
  for (const model::TargetLayerWeights &layer : weights.layers) {
    if (std::holds_alternative<model::GdnMixerWeights>(layer.mixer)) ++gdnLayers;
    if (std::holds_alternative<model::AttentionMixerWeights>(layer.mixer)) ++fullLayers;
  }
  check(gdnLayers == 48 && fullLayers == 16, "real NariQ target: hybrid mixer counts");
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc != 2 && argc != 3) {
      std::fprintf(stderr, "usage: gguf-trellis-image METALLIB [REAL-NARIQ.gguf]\n");
      return 2;
    }
    const richengine::test::TemporaryDirectory directory("richengine-gguf-trellis-image");
    MetalBackend backend(argv[1]);
    guarded("NariQ GGUF image preparation and loading", [&] { checkPrepared(backend, directory.path()); });
    if (argc == 3) guarded("real NariQ GGUF preparation and loading", [&] { checkReal(backend, argv[2]); });
    std::printf("%s (%d failures)\n", failures ? "GGUF trellis image tests FAILED" : "GGUF trellis image tests passed",
                failures);
    return failures ? 1 : 0;
  }
}
