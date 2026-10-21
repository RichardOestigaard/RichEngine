// The oQ (oMLX) formats' kernels (metal/abi/QuantFormat.h, ids
// GGUF_FMT_AF5..GGUF_FMT_AF8G128): the staged tile's dequantization of the
// slot-order code bitstream (gguf_test_dequant_*, kernels/common/oq_formats.h)
// against the CPU reference of GgufFormatReference.hpp, and the native-row
// token gathers (ops::Embedding, kernels/shared/embedding.metal) of every AF
// format and of the fp32 rows a table keeps unquantized
// (QuantizedSegment::kFloat32). The test metallib links the production kernels
// and the dequant test kernels into one library.
#include "GgufFormatReference.hpp"
#include "TestBuffers.hpp"
#include "metal/CommandGraph.hpp"
#include "ops/KernelNames.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/Gguf.h"
#include "ops/Embedding.hpp"
#include "tuning/LinearNumerics.hpp"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace gguf_reference;
using richengine::metal::CommandGraph;
using richengine::metal::MetalBackend;
using richengine::metal::MetalBuffer;
using richengine::test::sharedBuffer;
using richengine::ops::tuning::bf16ToFloat;
using richengine::ops::tuning::floatToBf16;

namespace {

MetalBuffer upload(MetalBackend &backend, const std::vector<uint8_t> &bytes) {
  MetalBuffer buffer = sharedBuffer(backend, bytes.size());
  std::memcpy(buffer.contents(), bytes.data(), bytes.size());
  return buffer;
}
MetalBuffer upload(MetalBackend &backend, const std::vector<float> &values) {
  MetalBuffer buffer = sharedBuffer(backend, values.size() * sizeof(float));
  std::memcpy(buffer.contents(), values.data(), buffer.sizeBytes());
  return buffer;
}

// Every weight of every oQ format is the half rounding of its fp32 value
// s * code + bias (gguf-dequant's check for the GGML formats).
int dequant(MetalBackend &backend, std::mt19937 &rng) {
  constexpr uint32_t N = 256, K = 1024, kGroup = 32, kThreads = 32;
  int failures = 0;
  for (uint32_t fi = GGUF_FMT_AF5; fi <= GGUF_FMT_AF8G128; ++fi) {
    const Fmt f = Fmt(fi);
    std::vector<float> values;
    const Packed planes = repack(f, makeNative(f, N, K, rng), N, K, &values);
    const MetalBuffer w0 = upload(backend, planes.w0), meta = upload(backend, planes.meta);
    const MetalBuffer output = sharedBuffer(backend, uint64_t{N} * K * 2);
    CommandGraph graph;
    graph.add(std::string(richengine::ops::kGgufTestDequant) + fmtName(f), {w0, meta, meta, output},
              GgufDecodeParams{K, 1, N, 0}, {N * (K / kGroup) / kThreads, 1, 1}, {kThreads, 1, 1});
    static_cast<void>(backend.submitCommand(graph.dispatches()));
    const auto *got = static_cast<const uint16_t *>(output.contents());
    size_t differ = 0;
    for (size_t i = 0; i < values.size(); ++i) differ += got[i] != f2h(values[i]);
    std::cout << fmtName(f) << ": " << differ << " of " << values.size()
              << " weights differ from FP16(fp32 dequantization) " << (differ ? "FAIL" : "ok") << '\n';
    failures += differ != 0;
  }
  return failures;
}

// The token gathers: each gathered value is the bf16 rounding of the row's
// fp32 value, for tokens in any order, repeated, and the vocabulary's first
// and last.
int tokenGather(MetalBackend &backend, std::mt19937 &rng) {
  constexpr uint32_t kVocabulary = 64, kHidden = 1024;
  const std::vector<uint32_t> tokens{kVocabulary - 1, 0, 17, 17, 42, 3, kVocabulary - 1, 29, 8};
  int failures = 0;
  for (uint32_t fi = GGUF_FMT_AF4; fi <= GGUF_FMT_AF8G128; ++fi) {
    const Fmt f = Fmt(fi);
    const std::vector<uint8_t> native = makeNative(f, kVocabulary, kHidden, rng);
    const richengine::ops::EmbeddingWeights table(kVocabulary, kHidden,
                                                richengine::ops::NativeRows(upload(backend, native), f));
    const MetalBuffer ids = sharedBuffer(backend, tokens.size() * sizeof(uint32_t)),
                      output = sharedBuffer(backend, uint64_t{tokens.size()} * kHidden * 2);
    std::memcpy(ids.contents(), tokens.data(), ids.sizeBytes());
    CommandGraph graph;
    richengine::ops::Embedding::add(graph, ids, table, output, uint32_t(tokens.size()));
    static_cast<void>(backend.submitCommand(graph.dispatches()));
    const auto *got = static_cast<const uint16_t *>(output.contents());
    std::vector<float> values(kHidden);
    size_t differ = 0;
    for (size_t r = 0; r < tokens.size(); ++r) {
      rowValues(f, native.data() + size_t(tokens[r]) * rowBytes(f, kHidden), kHidden, values.data());
      for (uint32_t k = 0; k < kHidden; ++k) differ += got[r * kHidden + k] != floatToBf16(values[k]);
    }
    if (differ) {
      std::cout << fmtName(f) << " gather: " << differ << " values differ from bf16(fp32) FAIL\n";
      ++failures;
    }
  }
  return failures;
}

// gguf_embed_f32 over plain fp32 rows: out[e] = bfloat(row[e]).
int floatGather(MetalBackend &backend, std::mt19937 &rng) {
  constexpr uint32_t kVocabulary = 64, kHidden = 512;
  const std::vector<uint32_t> tokens{kVocabulary - 1, 0, 17, 17, 42, 3, kVocabulary - 1, 29, 8};
  std::uniform_real_distribution<float> range(-2.0f, 2.0f);
  std::vector<float> rows(size_t{kVocabulary} * kHidden);
  for (float &v : rows) v = range(rng);
  const richengine::ops::EmbeddingWeights table(
      kVocabulary, kHidden,
      richengine::ops::NativeRows(upload(backend, rows), richengine::ops::QuantizedSegment::kFloat32));
  const MetalBuffer ids = sharedBuffer(backend, tokens.size() * sizeof(uint32_t)),
                    output = sharedBuffer(backend, uint64_t{tokens.size()} * kHidden * 2);
  std::memcpy(ids.contents(), tokens.data(), ids.sizeBytes());
  CommandGraph graph;
  richengine::ops::Embedding::add(graph, ids, table, output, uint32_t(tokens.size()));
  static_cast<void>(backend.submitCommand(graph.dispatches()));
  const auto *got = static_cast<const uint16_t *>(output.contents());
  size_t differ = 0;
  for (size_t r = 0; r < tokens.size(); ++r)
    for (uint32_t k = 0; k < kHidden; ++k)
      differ += got[r * kHidden + k] != floatToBf16(rows[size_t(tokens[r]) * kHidden + k]);
  if (differ) {
    std::cout << "f32 gather: " << differ << " values differ from bf16(fp32) FAIL\n";
    return 1;
  }
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc != 2) {
      std::cerr << "usage: oq-formats <oq-formats.metallib>\n";
      return 2;
    }
    std::mt19937 rng(42);
    int failures = 0;
    try {
      MetalBackend backend(argv[1]);
      failures += dequant(backend, rng);
      failures += tokenGather(backend, rng);
      failures += floatGather(backend, rng);
    } catch (const std::exception &e) {
      std::cerr << "oq-formats: FAIL: " << e.what() << '\n';
      return 1;
    }
    std::cout << "oq-formats: " << (failures ? "FAIL (" + std::to_string(failures) + " failures)" : "PASS")
              << '\n';
    return failures ? 1 : 0;
  }
}
