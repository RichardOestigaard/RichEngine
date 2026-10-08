#include "model/Ornith9B.hpp"
#include "model/TargetLoader.hpp"

#include <cstring>
#include <map>

namespace richengine::model {
namespace {

// Attaches a Prism ML GGUF's input rotation, which gguf::planImages checked
// names every quantized tensor of the target and its token table: every block
// projection multiplies H (D x) (ops::InputRotation), the token table gathers
// D (H r), and the GDN writes its value heads grouped, the order of the
// rotated output projection's inputs. One signs buffer per input width; the
// manifest fingerprint covers the signs.
void rotateInputs(metal::MetalBackend &backend, Ornith9BWeights &weights, const GgufRotation &rotation) {
  const uint64_t allocationBaseline = backend.memoryStats().allocatedBytes;
  std::map<uint32_t, metal::MetalBuffer> signs;
  std::string identity = weights.manifestFingerprintSha256 + " input rotation";
  for (const auto &[width, values] : rotation.signs) {
    metal::MetalBuffer buffer = backend.allocateBuffer(width, metal::BufferStorage::Shared, "rotation-signs");
    std::memcpy(buffer.contents(), values.data(), width);
    identity += " " + std::to_string(width) + ":" +
                weightDigest(std::span(reinterpret_cast<const uint8_t *>(values.data()), values.size()));
    signs.emplace(width, std::move(buffer));
  }
  const auto signsOf = [&](uint32_t width) {
    const auto found = signs.find(width);
    if (found == signs.end()) throw WeightStoreError("the rotation has no signs of width " + std::to_string(width));
    return found->second;
  };
  const auto rotate = [&](ops::Projection &projection) { projection.rotation.signs = signsOf(projection.inputSize); };
  for (auto &layer : weights.layers) {
    std::visit([&](auto &mixer) {
      rotate(mixer.inputProjection);
      rotate(mixer.outputProjection);
    }, layer.mixer);
    if (auto *gdn = std::get_if<GdnMixerWeights>(&layer.mixer)) gdn->outputHeadOrder = ops::GdnHeadOrder::Grouped;
    rotate(layer.gateProjection);
    rotate(layer.upProjection);
    rotate(layer.downProjection);
  }
  rotate(weights.logitsProjection);
  weights.tokenEmbedding.rotation.signs = signsOf(weights.tokenEmbedding.inputSize);
  weights.manifestFingerprintSha256 = weightDigest(identity);
  weights.actualAllocatedBytes += metal::allocationDelta(allocationBaseline, backend.memoryStats().allocatedBytes);
}

} // namespace

Ornith9BWeights loadOrnith9BWeights(metal::MetalBackend &backend, Ornith9BLayout layout,
                                   const TargetFiles<Ornith9BLayout> &files) {
  // The dense FFN reads the same projections from either format.
  const auto readFfn = [&](WeightFile &file, DenseLayerWeights &layer, const auto &format) {
    layer.gateProjection =
        format.projection(file, layout.intermediateSize, layout.hiddenSize, "mlp-gate");
    layer.upProjection =
        format.projection(file, layout.intermediateSize, layout.hiddenSize, "mlp-up");
    layer.downProjection =
        format.projection(file, layout.hiddenSize, layout.intermediateSize, "mlp-down");
  };
  Ornith9BWeights weights = loadTargetWeights<Ornith9BWeights>(backend, layout, files, readFfn);
  if (const auto *gguf = std::get_if<std::reference_wrapper<GgufTargetLoader>>(&files))
    if (const std::optional<GgufRotation> &rotation = gguf->get().rotation()) rotateInputs(backend, weights, *rotation);
  return weights;
}

} // namespace richengine::model
