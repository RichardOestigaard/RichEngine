#include "ModelDescriptor.hpp"
#include "families/FamilyMakers.hpp"
#include "DSparkDraft.hpp"
#include "DFlashV1Draft.hpp"
#include "QwenVision.hpp"
#include "WeightStore.hpp"
#include "metal/abi/ExecutionGeometry.h"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

namespace richengine::model {
namespace {

struct GeometryField final {
  const char *name;
  uint64_t value;
};

// A model's records and configurations are kilobytes of JSON: a file past
// this is not one and is not read, as a checkpoint's config.json is not
// (SafetensorsCheckpoint.mm).
constexpr uint64_t kMaximumJsonBytes = 1 << 20;

// The JSON object of the file at path; with sha256, the SHA-256 of its bytes
// too.
NSDictionary *readObject(const std::filesystem::path &path,
                         std::string_view label, std::string *sha256 = nullptr) {
  std::error_code sizeError;
  const uintmax_t bytes = std::filesystem::file_size(path, sizeError);
  if (!sizeError && bytes > kMaximumJsonBytes)
    throw std::invalid_argument(std::string(label) + " exceeds " +
                                std::to_string(kMaximumJsonBytes) + " bytes");
  NSString *nativePath = [NSString stringWithUTF8String:path.c_str()];
  if (!nativePath)
    throw std::invalid_argument(std::string(label) +
                                " path is not representable");
  NSError *readError = nil;
  NSData *data = [NSData dataWithContentsOfFile:nativePath
                                        options:0
                                          error:&readError];
  if (!data) {
    const char *description = readError.localizedDescription.UTF8String;
    throw std::invalid_argument("could not read " + std::string(label) +
                                ": " +
                                (description ? description
                                             : "unknown read error"));
  }
  if (sha256)
    *sha256 = weightDigest(std::span<const uint8_t>(static_cast<const uint8_t *>(data.bytes), data.length));
  NSError *parseError = nil;
  id value = [NSJSONSerialization JSONObjectWithData:data
                                             options:0
                                               error:&parseError];
  if (![value isKindOfClass:[NSDictionary class]]) {
    const char *description = parseError.localizedDescription.UTF8String;
    throw std::invalid_argument("could not parse " + std::string(label) +
                                ": " +
                                (description ? description
                                             : "expected a JSON object"));
  }
  return static_cast<NSDictionary *>(value);
}

NSDictionary *requireObject(NSDictionary *object, NSString *key,
                            std::string_view label) {
  id value = object[key];
  if (![value isKindOfClass:[NSDictionary class]])
    throw std::invalid_argument(std::string(label) + " must be an object");
  return static_cast<NSDictionary *>(value);
}

NSArray *requireArray(NSDictionary *object, NSString *key,
                      std::string_view label) {
  id value = object[key];
  if (![value isKindOfClass:[NSArray class]])
    throw std::invalid_argument(std::string(label) + " must be an array");
  return static_cast<NSArray *>(value);
}

std::string requireString(NSDictionary *object, NSString *key,
                          std::string_view label) {
  id value = object[key];
  if (![value isKindOfClass:[NSString class]])
    throw std::invalid_argument(std::string(label) + " must be a string");
  const char *text = static_cast<NSString *>(value).UTF8String;
  if (!text || !*text)
    throw std::invalid_argument(std::string(label) + " must not be empty");
  return text;
}

uint64_t requireUnsigned(NSDictionary *object, NSString *key,
                         std::string_view label) {
  id value = object[key];
  if (![value isKindOfClass:[NSNumber class]] ||
      CFGetTypeID((__bridge CFTypeRef)value) == CFBooleanGetTypeID()) {
    throw std::invalid_argument(std::string(label) +
                                " must be an unsigned integer");
  }
  NSNumber *number = static_cast<NSNumber *>(value);
  if (CFNumberIsFloatType((__bridge CFNumberRef)number)) {
    // A config saved from Python writes a float that holds a whole number,
    // rope_theta=1e7 as 10000000.0, and the installer compares it equal to
    // that integer. A double holds every integer exactly up to 2^53 - 1.
    const double real = number.doubleValue;
    if (std::isfinite(real) && real >= 1 && real <= 9007199254740991.0 &&
        std::floor(real) == real)
      return static_cast<uint64_t>(real);
  } else if (number.longLongValue > 0 &&
             static_cast<uint64_t>(number.longLongValue) ==
                 number.unsignedLongLongValue) {
    return number.unsignedLongLongValue;
  }
  throw std::invalid_argument(std::string(label) +
                              " must be a positive unsigned integer");
}

void requireEqual(uint64_t actual, uint64_t expected,
                  std::string_view label) {
  if (actual != expected) {
    throw std::invalid_argument(std::string(label) + " mismatch: package " +
                                std::to_string(actual) + ", runtime " +
                                std::to_string(expected));
  }
}

void requireEqual(std::string_view actual, std::string_view expected,
                  std::string_view label) {
  if (actual != expected) {
    throw std::invalid_argument(std::string(label) + " mismatch: package " +
                                std::string(actual) + ", runtime " +
                                std::string(expected));
  }
}

// A budget the source may declare smaller than the runtime's (an export
// trimming the trained context); larger values are rejected.
void requireAtMost(NSDictionary *object, NSString *key, uint64_t maximum,
                   std::string_view label) {
  const uint64_t actual = requireUnsigned(object, key, label);
  if (actual > maximum) {
    throw std::invalid_argument(std::string(label) + " mismatch: package " +
                                std::to_string(actual) + ", runtime " +
                                std::to_string(maximum));
  }
}

// The one rule every number of a record or configuration is checked by: a
// JSON number, never a boolean, equal to the value expected. A config saved
// from Python may write a whole number as a float, rope_theta=1e7 as
// 10000000.0, which equals that integer; every value expected is exact as a
// double.
void requireNumber(id value, double expected, std::string_view label) {
  if (![value isKindOfClass:[NSNumber class]] ||
      CFGetTypeID((__bridge CFTypeRef)value) == CFBooleanGetTypeID())
    throw std::invalid_argument(std::string(label) + " must be a number");
  if ([value doubleValue] != expected)
    throw std::invalid_argument(std::string(label) + " mismatch: package " +
                                [value description].UTF8String + ", runtime " +
                                @(expected).description.UTF8String);
}

// A key and the number it must hold.
struct ExpectedNumber final {
  template <class Number>
  ExpectedNumber(const char *key, Number value)
      : key(key), value(static_cast<double>(value)) {}
  const char *key;
  double value;
};

// The numbers object holds at the keys, which errors name after `where`.
void requireNumbers(NSDictionary *object, std::string_view where,
                    std::initializer_list<ExpectedNumber> fields) {
  for (const ExpectedNumber &field : fields)
    requireNumber(object[@(field.key)], field.value,
                  std::string(where) + " " + field.key);
}

// The numbers of an array, in order.
void requireNumbers(NSArray *values, std::span<const uint32_t> expected,
                    std::string_view label) {
  if (values.count != expected.size())
    throw std::invalid_argument(std::string(label) + " count mismatch: package " +
                                std::to_string(values.count) + ", runtime " +
                                std::to_string(expected.size()));
  for (size_t index = 0; index < expected.size(); ++index)
    requireNumber(values[index], expected[index],
                  std::string(label) + " " + std::to_string(index));
}

// The JSON booleans, never numbers, object holds at the keys.
void requireBooleans(NSDictionary *object, std::string_view where,
                     std::initializer_list<std::pair<const char *, bool>> fields) {
  for (const auto &[key, expected] : fields) {
    id value = object[@(key)];
    if (![value isKindOfClass:[NSNumber class]] ||
        CFGetTypeID((__bridge CFTypeRef)value) != CFBooleanGetTypeID() ||
        [value boolValue] != expected)
      throw std::invalid_argument(std::string(where) + " " + key + " must be " +
                                  (expected ? "true" : "false"));
  }
}

// Each layer's type in a layer_types array, as the array names a
// full-attention layer and a GDN layer.
template <class Target>
void requireLayerTypes(NSArray *types, const Target &target,
                       NSString *attention, NSString *gdn,
                       std::string_view label) {
  if (types.count != target.layers)
    throw std::invalid_argument(std::string(label) + " count mismatch: package " +
                                std::to_string(types.count) + ", runtime " +
                                std::to_string(target.layers));
  for (uint32_t layer = 0; layer < target.layers; ++layer) {
    NSString *expected = target.isFullAttentionLayer(layer) ? attention : gdn;
    if (![types[layer] isEqual:expected])
      throw std::invalid_argument(std::string(label) + " " +
                                  std::to_string(layer) + " must be " +
                                  expected.UTF8String);
  }
}

// A package records the execution geometry it was published with. Its draft
// was trained for blocks of draft_query_rows rows, the anchor and
// draft_proposal_tokens proposals, over draft_sliding_window context tokens,
// which the draft kernels are built for. The batch width, prefill budget, KV
// page and verify rows it records were that runtime's choices; this runtime
// makes its own.
void validateExecutionGeometry(NSDictionary *manifest) {
  requireNumbers(requireObject(manifest, @"execution_geometry",
                               "model execution geometry"),
                 "execution_geometry",
                 {{"draft_proposal_tokens", ExecutionLimits::draftProposalTokens},
                  {"draft_query_rows", ExecutionLimits::draftQueryRows}});
  // The recorded window is the trained context the draft was packed for;
  // anything beyond the ring's physical capacity would be silently clamped.
  const uint64_t recordedWindow = requireUnsigned(
      requireObject(manifest, @"execution_geometry", "model execution geometry"),
      @"draft_sliding_window", "execution_geometry draft_sliding_window");
  if (recordedWindow > ExecutionLimits::draftRingCapacity)
    throw std::invalid_argument(
        "draft sliding_window exceeds the runtime ring");
}

void validateCommonFormat(NSDictionary *format, std::string_view targetMagic,
                          std::string_view draftMagic) {
  requireNumbers(format, "format",
                 {{"section_alignment_bytes", kWeightFileAlignment}});
  requireEqual(requireString(format, @"target_layer_magic",
                             "target_layer_magic"),
               targetMagic, "target_layer_magic");
  requireEqual(requireString(format, @"draft_layer_magic",
                             "draft_layer_magic"),
               draftMagic, "draft_layer_magic");
  requireEqual(requireString(format, @"vision_magic", "vision_magic"),
               kVisionMagic, "vision_magic");
}

[[nodiscard]] std::string_view draftLayerMagic(const DFlashDraftLayout &draft) {
  switch (draft.kind) {
  case DraftKind::DFlashV1: return kDFlashV1DraftMagic;
  case DraftKind::DSpark: return kDSparkDraftMagic;
  default: return kDFlashLayerMagic;
  }
}

[[nodiscard]] bool isDSparkArchitecture(std::string_view architecture) {
  return architecture == "Qwen3DSparkModel" ||
         architecture == "Lfm2DSparkDraftModel" ||
         architecture == "DSparkDraftModel";
}

[[nodiscard]] std::string_view draftArchitecture(const DFlashDraftLayout &draft) {
  switch (draft.kind) {
  case DraftKind::DFlashV1: return "DFlashDraftModel";
  case DraftKind::DSpark: return "Qwen3DSparkModel";
  default: return "DFlash2DraftModel";
  }
}

// A packed manifest selects its draft format by the layer files' magic:
// MDFD0004 is a packed DFlash2 draft, MDFP0005 a packed plain-transformer
// DFlash draft. `dflashV1Layout` is null for a target with no plain draft.
void applyManifestDraftKind(NSDictionary *format, ModelDescriptor &descriptor,
                            const DFlashDraftLayout *dflashV1Layout,
                            const DFlashDraftLayout &dflash2Layout) {
  const std::string magic =
      requireString(format, @"draft_layer_magic", "draft_layer_magic");
  if (magic == kDFlashLayerMagic) {
    descriptor.draft = dflash2Layout;
  } else if (magic == kDFlashV1DraftMagic && dflashV1Layout) {
    descriptor.draft = *dflashV1Layout;
    // The DFlash drafts ship their causal sliding layers first and one full
    // layer last; a manifest may override with an explicit causal_layers mask.
    id causal = format[@"causal_layers"];
    if (causal)
      descriptor.draft.causalLayers = static_cast<uint32_t>(
          requireUnsigned(format, @"causal_layers", "causal_layers"));
  } else {
    throw std::invalid_argument("unsupported draft layer magic: " + magic);
  }
  descriptor.stateLayout.draft = descriptor.draft.stateLayout();
}

// Defined below validateNewPackedFormat, which it calls.
void validateGemma4(NSDictionary *manifest,
                    const std::filesystem::path &root,
                    const ModelDescriptor &descriptor);
void validateDiffusionGemma(NSDictionary *manifest,
                            const std::filesystem::path &root,
                            const ModelDescriptor &descriptor);

void validateTokenizer(const std::filesystem::path &root,
                       const ModelDescriptor &descriptor,
                       std::string_view expectedTextModelType) {
  NSDictionary *config = readObject(root / "tokenizer" / "config.json",
                                    "tokenizer model config");
  // Flat tokenizer configs (the dense and LFM2 targets) declare the fields
  // at the top level; the Qwen families nest them under text_config.
  NSDictionary *text = config[@"text_config"];
  if (![text isKindOfClass:[NSDictionary class]]) text = config;
  requireEqual(requireString(text, @"model_type", "text model type"),
               expectedTextModelType, "text model type");
  requireNumbers(
      text, "tokenizer text config",
      {{"hidden_size",
        std::visit([](const auto &layout) { return layout.hiddenSize; },
                   descriptor.target)},
       {"vocab_size", descriptor.capabilities.vocabularySize}});
  // Exports disagree on the trained context (131072 vs 128000).
  requireAtMost(text, @"max_position_embeddings",
                descriptor.capabilities.maximumContextTokens,
                "tokenizer max_position_embeddings");
}

// The dense qwen3_5_text package format, shared by the 27B and Ornith 9B;
// the descriptor's layout decides the family's sizes.
void validateQwen38(NSDictionary *manifest,
                    const std::filesystem::path &root,
                    const ModelDescriptor &descriptor) {
  requireNumbers(manifest, "manifest", {{"schema_version", 3}});
  NSDictionary *format =
      requireObject(manifest, @"format", "model weight format");
  requireNumbers(format, "format",
                 {{"q4_bits", 4},
                  {"q4_group_size", kQ4GroupElements},
                  {"q4_storage_n", kQ4StorageN}});
  const std::string_view layerMagic = std::visit(
      [](const auto &layout) { return std::decay_t<decltype(layout)>::layerMagic; },
      descriptor.target);
  validateCommonFormat(format, layerMagic, draftLayerMagic(descriptor.draft));
  validateTokenizer(root, descriptor, "qwen3_5_text");
}

void validateCaptureLayers(NSDictionary *draft, const Qwen3_6MoeLayout &layout) {
  NSArray *layers =
      requireArray(draft, @"target_capture_layers", "target capture layers");
  requireEqual(layers.count, layout.hiddenCaptureLayers.size(),
               "target capture layer count");
  for (uint32_t index = 0; index < layers.count; ++index) {
    id value = layers[index];
    if (![value isKindOfClass:[NSNumber class]])
      throw std::invalid_argument("target capture layer must be an integer");
    requireEqual(static_cast<NSNumber *>(value).unsignedLongLongValue,
                 layout.hiddenCaptureLayers[index],
                 "target capture layer " + std::to_string(index));
  }
}

void validateQwen36(NSDictionary *manifest,
                    const std::filesystem::path &root,
                    const ModelDescriptor &descriptor) {
  requireNumbers(manifest, "manifest", {{"schema_version", 4}});
  NSDictionary *format =
      requireObject(manifest, @"format", "model weight format");
  requireNumbers(format, "format",
                 {{"q4_bits", 4},
                  {"q8_bits", 8},
                  {"quant_group_size", kQ4GroupElements},
                  {"storage_n", kQ4StorageN}});
  validateCommonFormat(format, Qwen3_6MoeLayout::layerMagic,
                       draftLayerMagic(descriptor.draft));

  const auto &targetLayout = std::get<Qwen3_6MoeLayout>(descriptor.target);
  NSDictionary *target =
      requireObject(manifest, @"target", "target declaration");
  requireEqual(requireString(target, @"architecture", "target architecture"),
               "qwen3_5_moe", "target architecture");
  requireNumbers(target, "target",
                 {{"layers", targetLayout.layers},
                  {"hidden_size", targetLayout.hiddenSize},
                  {"vocabulary_size", targetLayout.vocabularySize},
                  {"gdn_actual_width", targetLayout.actualGdnWidth()},
                  {"gdn_packed_width", targetLayout.packedGdnWidth},
                  {"attention_packed_width", targetLayout.packedFullWidth},
                  {"experts", targetLayout.experts},
                  {"experts_per_token", targetLayout.expertsPerToken},
                  {"moe_intermediate_size", targetLayout.expertIntermediateSize},
                  {"shared_expert_intermediate_size",
                   targetLayout.expertIntermediateSize}});
  requireLayerTypes(requireArray(target, @"layer_types", "target layer_types"),
                    targetLayout, @"attention", @"gdn", "target layer_types");

  const DFlashDraftLayout &draftLayout = descriptor.draft;
  NSDictionary *draft =
      requireObject(manifest, @"draft", "draft declaration");
  requireEqual(requireString(draft, @"architecture", "draft architecture"),
               draftArchitecture(draftLayout), "draft architecture");
  for (const GeometryField &field : std::to_array<GeometryField>(
           {{"layers", draftLayout.layers},
            {"hidden_size", draftLayout.hiddenSize},
            {"intermediate_size", draftLayout.intermediateSize}})) {
    requireEqual(requireUnsigned(draft,
                                 [NSString stringWithUTF8String:field.name],
                                 field.name),
                 field.value, field.name);
  }
  if (draftLayout.kind == DraftKind::DFlashV1) {
    // The packed plain draft may declare its own geometry: the trained
    // block must cover the engine's eight rows, and the sliding window the
    // ring holds may be shallower than the model's declared one.
    for (const GeometryField &field : std::to_array<GeometryField>(
             {{"num_attention_heads", draftLayout.attentionSize / draftLayout.attentionHeadDimension},
              {"num_key_value_heads", draftLayout.kvHeads},
              {"head_dim", draftLayout.attentionHeadDimension},
              {"causal_layers", draftLayout.causalLayers}})) {
      requireEqual(requireUnsigned(draft,
                                   [NSString stringWithUTF8String:field.name],
                                   field.name),
                   field.value, field.name);
    }
    // The packed plain draft's trained block must match the family layout's;
    // the runtime may dispatch more query rows than it trains for.
    requireEqual(requireUnsigned(draft, @"block_size", "draft block_size"),
                 uint64_t{draftLayout.blockSize}, "draft block_size");
    requireEqual(requireUnsigned(draft, @"sliding_window", "draft sliding_window"),
                 uint64_t{draftLayout.slidingWindow}, "draft sliding_window");
  } else {
    for (const GeometryField &field : std::to_array<GeometryField>(
             {{"sliding_window", ExecutionLimits::draftContextTokens},
              {"block_size", draftLayout.blockSize},
              {"dynamic_conv_group_size", 16},
              {"dynamic_conv_kernel_size", 2},
              {"selector_rank", draftLayout.selectorRank},
              {"selector_top_k", 16}})) {
      requireEqual(requireUnsigned(draft,
                                   [NSString stringWithUTF8String:field.name],
                                   field.name),
                   field.value, field.name);
    }
  }
  validateCaptureLayers(draft, targetLayout);
  validateTokenizer(root, descriptor, "qwen3_5_moe_text");
}

// The dense and LFM2 packed formats' draft declaration: a plain transformer
// or DSpark draft whose geometry the manifest declares in full.
void applyDeclaredDraft(NSDictionary *manifest, NSDictionary *format,
                        ModelDescriptor &descriptor,
                        const DFlashDraftLayout &defaults) {
  descriptor.draft = defaults;
  NSDictionary *draft = requireObject(manifest, @"draft", "draft declaration");
  const std::string architecture =
      requireString(draft, @"architecture", "draft architecture");
  const bool dspark =
      defaults.kind == DraftKind::DSpark && isDSparkArchitecture(architecture);
  if (architecture == draftArchitecture(defaults) || dspark) {
    if (defaults.kind != DraftKind::DFlash2) {
      DFlashDraftLayout &d = descriptor.draft;
      const uint64_t heads =
          requireUnsigned(draft, @"num_attention_heads", "draft num_attention_heads");
      const uint64_t kvHeads =
          requireUnsigned(draft, @"num_key_value_heads", "draft num_key_value_heads");
      const uint64_t headDim = requireUnsigned(draft, @"head_dim", "draft head_dim");
      // The compiled draft attention cores cover 32x8x128 (half-split),
      // 16x2x128 and interleaved 32x8x64 head patterns.
      const bool supported =
          (heads == 32 && kvHeads == 8 && headDim == 128) ||
          (heads == 16 && kvHeads == 2 && headDim == 128) ||
          (heads == 32 && kvHeads == 8 && headDim == 64);
      if (!supported)
        throw std::invalid_argument("unsupported draft head geometry");
      d.layers = static_cast<uint32_t>(
          requireUnsigned(draft, @"layers", "draft layers"));
      d.hiddenSize = static_cast<uint32_t>(
          requireUnsigned(draft, @"hidden_size", "draft hidden_size"));
      d.intermediateSize = static_cast<uint32_t>(
          requireUnsigned(draft, @"intermediate_size", "draft intermediate_size"));
      d.kvHeads = static_cast<uint32_t>(kvHeads);
      d.attentionHeadDimension = static_cast<uint32_t>(headDim);
      d.attentionSize = static_cast<uint32_t>(heads * headDim);
      d.qkvSize = static_cast<uint32_t>((heads + 2 * kvHeads) * headDim);
      d.rotaryTheta = static_cast<float>(
          requireUnsigned(draft, @"rope_theta", "draft rope_theta"));
      const uint64_t blockSize =
          requireUnsigned(draft, @"block_size", "draft block_size");
      // The runtime dispatches draftQueryRows regardless of the trained
      // block; the cap on acceptance is the block's proposals — block_size
      // rows for a plain draft, block_size proposals for a DSpark.
      if (blockSize < (dspark ? 1 : 2))
        throw std::invalid_argument("draft block_size is below a proposal row");
      d.blockSize = static_cast<uint32_t>(blockSize);
      if (dspark) {
        d.causalLayers = 0;
        d.markovRank = static_cast<uint32_t>(
            requireUnsigned(draft, @"markov_rank", "draft markov_rank"));
        requireEqual(d.markovRank, RICHENGINE_DRAFT_SELECTOR_RANK,
                     "draft markov_rank");
        d.ropeInterleaved =
            [draft[@"rope_interleaved"] isEqual:@YES] ? 1 : 0;
        const uint64_t captures = static_cast<uint64_t>(std::visit(
            [](const auto &layout) { return layout.hiddenCaptureLayers.size(); },
            descriptor.target));
        if (d.targetHiddenSize !=
            captures * std::visit([](const auto &layout) { return layout.hiddenSize; },
                                  descriptor.target))
          throw std::invalid_argument("draft captured hidden size mismatch");
      } else {
        d.causalLayers = static_cast<uint32_t>(
            requireUnsigned(draft, @"causal_layers", "draft causal_layers"));
        const uint64_t window =
            requireUnsigned(draft, @"sliding_window", "draft sliding_window");
        d.slidingWindow = static_cast<uint32_t>(
            std::min<uint64_t>(window, ExecutionLimits::draftRingCapacity));
      }
    }
  } else {
    throw std::invalid_argument("unsupported draft architecture: " + architecture);
  }
  requireEqual(requireString(format, @"draft_layer_magic", "draft_layer_magic"),
               draftLayerMagic(descriptor.draft), "draft_layer_magic");
  descriptor.stateLayout.draft = descriptor.draft.stateLayout();
}

// The shared fields of a new-family packed format: the Q4 constants, the
// common file magics and the flat tokenizer contract.
void validateNewPackedFormat(NSDictionary *manifest,
                             const std::filesystem::path &root,
                             const ModelDescriptor &descriptor,
                             std::string_view modelType) {
  requireEqual(requireUnsigned(manifest, @"schema_version", "schema_version"),
               1, "schema_version");
  NSDictionary *format =
      requireObject(manifest, @"format", "model weight format");
  requireEqual(requireUnsigned(format, @"q4_bits", "q4_bits"), 4, "q4_bits");
  requireEqual(requireUnsigned(format, @"q4_group_size", "q4_group_size"),
               kQ4GroupElements, "q4_group_size");
  requireEqual(requireUnsigned(format, @"q4_storage_n", "q4_storage_n"),
               kQ4StorageN, "q4_storage_n");
  const std::string_view layerMagic = std::visit(
      [](const auto &layout) { return std::decay_t<decltype(layout)>::layerMagic; },
      descriptor.target);
  validateCommonFormat(format, layerMagic, draftLayerMagic(descriptor.draft));
  validateTokenizer(root, descriptor, modelType);
}

void requireNumbers(NSDictionary *object, std::initializer_list<GeometryField> fields) {
  for (const auto &field : fields)
    requireEqual(requireUnsigned(object, [NSString stringWithUTF8String:field.name], field.name), field.value, field.name);
}

// The packed gemma4 manifest: the shared format fields plus the target's
// dual attention geometry and MoE sizes.
void validateGemma4(NSDictionary *manifest,
                    const std::filesystem::path &root,
                    const ModelDescriptor &descriptor) {
  validateNewPackedFormat(manifest, root, descriptor, "gemma4_text");
  const Gemma4MoeLayout &targetLayout =
      std::get<Gemma4MoeLayout>(descriptor.target);
  NSDictionary *target =
      requireObject(manifest, @"target", "target declaration");
  requireEqual(requireString(target, @"architecture", "target architecture"),
               "gemma4_text", "target architecture");
  requireNumbers(target, "target",
                 {{"layers", targetLayout.layers},
                  {"hidden_size", targetLayout.hiddenSize},
                  {"vocabulary_size", targetLayout.vocabularySize},
                  {"num_attention_heads", targetLayout.attentionQueryHeads},
                  {"num_key_value_heads", targetLayout.attentionKvHeads},
                  {"head_dim", targetLayout.attentionHeadDimension},
                  {"global_num_key_value_heads", targetLayout.globalKvHeads},
                  {"global_head_dim", targetLayout.globalHeadDimension},
                  {"sliding_window", targetLayout.slidingWindowTokens},
                  {"experts", targetLayout.experts},
                  {"experts_per_token", targetLayout.expertsPerToken},
                  {"moe_intermediate_size",
                   targetLayout.expertIntermediateSize},
                  {"shared_expert_intermediate_size",
                   targetLayout.sharedIntermediateSize},
                  {"final_logit_softcapping", targetLayout.logitSoftcap},
                  {"global_rope_theta",
                   static_cast<uint64_t>(targetLayout.globalRotaryTheta)},
                  {"rope_theta",
                   static_cast<uint64_t>(targetLayout.rotaryTheta)}});
  requireLayerTypes(requireArray(target, @"layer_types", "target layer_types"),
                    targetLayout, @"full_attention", @"sliding_attention",
                    "target layer_types");
}

// The packed diffusiongemma manifest: the Gemma 4 target declaration plus a
// top-level `diffusion` block carrying the denoising schedule, and no draft.
void validateDiffusionGemma(NSDictionary *manifest,
                            const std::filesystem::path &root,
                            const ModelDescriptor &descriptor) {
  validateNewPackedFormat(manifest, root, descriptor,
                          "diffusion_gemma_text");
  const DiffusionGemmaLayout &targetLayout =
      std::get<DiffusionGemmaLayout>(descriptor.target);
  NSDictionary *target =
      requireObject(manifest, @"target", "target declaration");
  requireEqual(requireString(target, @"architecture", "target architecture"),
               "diffusion_gemma_text", "target architecture");
  requireNumbers(target, "target",
                 {{"layers", targetLayout.layers},
                  {"hidden_size", targetLayout.hiddenSize},
                  {"vocabulary_size", targetLayout.vocabularySize},
                  {"num_attention_heads", targetLayout.attentionQueryHeads},
                  {"num_key_value_heads", targetLayout.attentionKvHeads},
                  {"head_dim", targetLayout.attentionHeadDimension},
                  {"global_num_key_value_heads", targetLayout.globalKvHeads},
                  {"global_head_dim", targetLayout.globalHeadDimension},
                  {"sliding_window", targetLayout.slidingWindowTokens},
                  {"experts", targetLayout.experts},
                  {"experts_per_token", targetLayout.expertsPerToken},
                  {"moe_intermediate_size",
                   targetLayout.expertIntermediateSize},
                  {"shared_expert_intermediate_size",
                   targetLayout.sharedIntermediateSize},
                  {"final_logit_softcapping", targetLayout.logitSoftcap},
                  {"global_rope_theta",
                   static_cast<uint64_t>(targetLayout.globalRotaryTheta)},
                  {"rope_theta",
                   static_cast<uint64_t>(targetLayout.rotaryTheta)}});
  requireLayerTypes(requireArray(target, @"layer_types", "target layer_types"),
                    targetLayout, @"full_attention", @"sliding_attention",
                    "target layer_types");
  if ([manifest[@"draft"] isKindOfClass:[NSDictionary class]])
    throw std::invalid_argument("diffusion gemma targets take no draft");
  NSDictionary *diffusion =
      requireObject(manifest, @"diffusion", "diffusion schedule");
  const DiffusionSchedule &schedule = targetLayout.diffusion;
  // The schedule's float fields compare as floats: a JSON 0.1 has no exact
  // double equal of 0.1f.
  const auto requireFloat = [&](const char *key, float expected) {
    id value = diffusion[@(key)];
    if (![value isKindOfClass:[NSNumber class]] ||
        CFGetTypeID((__bridge CFTypeRef)value) == CFBooleanGetTypeID() ||
        static_cast<float>([value doubleValue]) != expected)
      throw std::invalid_argument(std::string("diffusion schedule ") + key +
                                  " mismatch");
  };
  requireNumbers(diffusion, "diffusion schedule",
                 {{"canvas_length", schedule.canvasLength},
                  {"max_denoising_steps", schedule.maxDenoisingSteps},
                  {"stability_threshold", schedule.stabilityThreshold}});
  requireFloat("t_min", schedule.tMin);
  requireFloat("t_max", schedule.tMax);
  requireFloat("entropy_bound", schedule.entropyBound);
  requireFloat("confidence_threshold", schedule.confidenceThreshold);
}

// The target's text configuration. Every one holds the sizes the descriptor
// shares with it. An MLX target's config.json, which is target/config.json
// too and which its images are planned from, also holds the rest of what the
// kernels compute; a GGUF's is what the installer derived from the GGUF's
// metadata, which the GGUF planner checks.
template <class Layout>
void validateTextConfig(NSDictionary *text, const Layout &target,
                        TargetSource source) {
  requireNumbers(text, "text config",
                 {{"hidden_size", target.hiddenSize},
                  {"num_hidden_layers", target.layers},
                  {"vocab_size", target.vocabularySize},
                  {"max_position_embeddings", target.maximumContextTokens},
                  {"num_attention_heads", target.attentionQueryHeads},
                  {"num_key_value_heads", target.attentionKvHeads},
                  {"head_dim", target.attentionHeadDimension}});
  if (source != TargetSource::Mlx) return;
  requireNumbers(text, "text config",
                 {{"linear_num_key_heads", target.gdnKeyHeads},
                  {"linear_num_value_heads", target.gdnValueHeads},
                  {"linear_key_head_dim", target.gdnHeadDimension},
                  {"linear_value_head_dim", target.gdnHeadDimension},
                  {"linear_conv_kernel_dim", kGdnConvolutionTaps},
                  {"full_attention_interval", target.fullAttentionPeriod},
                  {"rms_norm_eps", 1e-6}});
  if constexpr (std::decay_t<Layout>::ffnKind == FfnKind::SparseMoe)
    requireNumbers(text, "text config",
                   {{"num_experts", target.experts},
                    {"num_experts_per_tok", target.expertsPerToken},
                    {"moe_intermediate_size", target.expertIntermediateSize},
                    {"shared_expert_intermediate_size",
                     target.expertIntermediateSize}});
  else
    requireNumbers(text, "text config",
                   {{"intermediate_size", target.intermediateSize}});
  requireBooleans(text, "text config",
                  {{"attention_bias", false},
                   {"attn_output_gate", true},
                   {"tie_word_embeddings", false}});
  requireEqual(requireString(text, @"hidden_act", "text config hidden_act"),
               "silu", "text config hidden_act");
  requireLayerTypes(requireArray(text, @"layer_types", "text config layer_types"),
                    target, @"full_attention", @"linear_attention",
                    "text config layer_types");
  NSDictionary *rope =
      requireObject(text, @"rope_parameters", "text config rope_parameters");
  requireNumbers(rope, "text config rope_parameters",
                 {{"rope_theta", target.rotaryTheta},
                  {"partial_rotary_factor", 2.0 * target.rotaryPairs /
                                                target.attentionHeadDimension}});
  // Transformers also reads the rope type from the older `type` key, which
  // fine-tunes such as Ornith 1.5 still write.
  NSString *typeKey = rope[@"rope_type"] ? @"rope_type" : @"type";
  const std::string typeLabel =
      std::string("text config rope_parameters ") + typeKey.UTF8String;
  requireEqual(requireString(rope, typeKey, typeLabel), "default", typeLabel);
}

// A DFlash2 checkpoint's config: the draft's layout; the block, window,
// convolutions and selector the draft kernels are built for; and the target's
// mask token and the layers the draft reads. Returns the declared block_size:
// the runtime may dispatch more query rows than the block trains for.
uint32_t validateDraftConfig(NSDictionary *draft, const DFlashDraftLayout &layout,
                             uint32_t maskToken,
                             std::span<const uint32_t> captureLayers) {
  NSArray *architectures =
      requireArray(draft, @"architectures", "draft architectures");
  if (architectures.count != 1 ||
      ![architectures[0] isEqual:@"DFlash2DraftModel"])
    throw std::invalid_argument("draft is not a DFlash2 model");
  requireNumbers(draft, "draft config",
                 {{"num_hidden_layers", layout.layers},
                  {"hidden_size", layout.hiddenSize},
                  {"vocab_size", layout.vocabularySize},
                  {"intermediate_size", layout.intermediateSize},
                  {"num_attention_heads",
                   layout.attentionSize / layout.attentionHeadDimension},
                  {"num_key_value_heads", layout.kvHeads},
                  {"head_dim", layout.attentionHeadDimension},
                  {"sliding_window", ExecutionLimits::draftContextTokens},
                  {"rms_norm_eps", 1e-6}});
  requireBooleans(draft, "draft config",
                  {{"is_causal", false},
                   {"attention_bias", false},
                   {"tie_word_embeddings", false}});
  requireEqual(requireString(draft, @"hidden_act", "draft config hidden_act"),
               "silu", "draft config hidden_act");
  NSDictionary *rope =
      requireObject(draft, @"rope_parameters", "draft config rope_parameters");
  requireEqual(requireString(rope, @"rope_type",
                             "draft config rope_parameters rope_type"),
               "default", "draft config rope_parameters rope_type");
  requireNumbers(rope, "draft config rope_parameters",
                 {{"rope_theta", layout.rotaryTheta}});
  NSDictionary *flash =
      requireObject(draft, @"dflash_config", "draft config dflash_config");
  const uint64_t blockSize =
      requireUnsigned(flash, @"block_size", "draft config block_size");
  if (blockSize < 2)
    throw std::invalid_argument("draft block_size is below a proposal row");
  requireNumbers(flash, "draft config dflash_config",
                 {{"conv_group_size", kDraftConvolutionGroup},
                  {"conv_kernel_size", kDraftConvolutionTaps},
                  {"selector_rank", layout.selectorRank},
                  {"selector_top_k", RICHENGINE_DRAFT_CANDIDATES},
                  {"mask_token_id", maskToken}});
  requireNumbers(requireArray(flash, @"target_layer_ids",
                              "draft config target_layer_ids"),
                 captureLayers, "draft config target_layer_ids");
  return static_cast<uint32_t>(blockSize);
}

// The families an assembly's model.json may name (install/families.py's
// FAMILIES), each with the model_type its upstream config states. The
// record's family selects the descriptor — the installer resolved it from
// the same configuration — and the config is then validated against it;
// the runtime never re-guesses a family from the geometry.
struct FamilyDescriptor {
  const char *name;
  const char *modelType;
  // A model_type an early release of the family also stated, else null.
  const char *legacyType;
  ModelDescriptor (*make)(std::string name);
};

constexpr FamilyDescriptor kModelFamilies[] = {
    {"Qwen3.8-27B", "qwen3_5_text", nullptr, qwen38Descriptor},
    {"Bonsai-2-27B", "qwen3_5_text", nullptr, qwen38Descriptor},
    {"Ornith-1.5-9B", "qwen3_5_text", nullptr, ornithDescriptor},
    {"Qwen3.6-35B-A3B", "qwen3_5_moe_text", nullptr, qwen36Descriptor},
    {"Ornith-1.5-35B-A3B", "qwen3_5_moe_text", nullptr, qwen36Descriptor},
    // MiniCPM5's early releases stated model_type "minicpm".
    {"MiniCPM5-2B", "llama", "minicpm", denseDescriptor},
    {"LFM2.5-2.6B", "lfm2", nullptr, lfm2Descriptor},
    {"LFM2.5-8B-A1B", "lfm2_moe", nullptr, lfm2moeDescriptor},
    {"Granite-4.2-3B", "granite", nullptr, granite3BDescriptor},
    {"Granite-4.2-8B", "granite", nullptr, granite8BDescriptor},
    {"Gemma4-26B-A4B", "gemma4_text", nullptr, gemma4Descriptor},
    {"DiffusionGemma-26B-A4B", "diffusion_gemma_text", nullptr, diffusionGemmaDescriptor},
};

ModelDescriptor inspectSourceModel(const std::filesystem::path &root) {
  std::string sourceIdentity;
  NSDictionary *record = readObject(root / "model.json", "resolved model", &sourceIdentity);
  requireNumbers(record, "model record", {{"version", 1}});
  NSDictionary *config = readObject(root / "config.json", "upstream model config");
  // The Qwen families nest their text fields under text_config; MiniCPM5
  // ("minicpm") and LFM2 declare theirs flat.
  NSDictionary *text = config[@"text_config"];
  if (![text isKindOfClass:[NSDictionary class]]) text = config;
  const auto type = requireString(text, @"model_type", "text model type");
  const auto name = requireString(record, @"model", "model name");
  const auto family = requireString(record, @"family", "model family");
  const auto entry = std::ranges::find(kModelFamilies, family,
                                       &FamilyDescriptor::name);
  if (entry == std::end(kModelFamilies))
    throw std::invalid_argument("unsupported model family: " + family);
  if (type != entry->modelType &&
      !(entry->legacyType && type == entry->legacyType))
    throw std::invalid_argument(
        "model family " + family + " states model_type " + entry->modelType +
        ", not " + type);
  ModelDescriptor result = entry->make(name);
  std::visit([&](const auto &layout) {
    using Layout = std::decay_t<decltype(layout)>;
    if constexpr (std::is_same_v<Layout, DenseLayout>) {
      requireNumbers(text, {{"hidden_size", layout.hiddenSize}, {"num_hidden_layers", layout.layers},
          {"vocab_size", layout.vocabularySize}, {"max_position_embeddings", layout.maximumContextTokens},
          {"num_attention_heads", layout.attentionQueryHeads},
          {"num_key_value_heads", layout.attentionKvHeads}});
      // MiniCPM5's head_dim is implicit (hidden / heads).
      if (text[@"head_dim"])
        requireNumbers(text, {{"head_dim", layout.attentionHeadDimension}});
    } else if constexpr (std::is_same_v<Layout, GraniteLayout>) {
      requireNumbers(text, {{"hidden_size", layout.hiddenSize}, {"num_hidden_layers", layout.layers},
          {"vocab_size", layout.vocabularySize},
          {"num_attention_heads", layout.attentionQueryHeads},
          {"num_key_value_heads", layout.attentionKvHeads},
          {"intermediate_size", layout.intermediateSize},
          {"max_position_embeddings", layout.maximumContextTokens},
          {"rope_theta", 10000000u}});
      // attention_multiplier is a float the GGUF metadata check verifies.
      if (id value = text[@"attention_multiplier"];
          [value isKindOfClass:[NSNumber class]] &&
          std::abs([value doubleValue] - layout.attentionScale) >
              layout.attentionScale * 1e-6)
        throw std::invalid_argument(
            "unsupported granite attention_multiplier: " +
            std::to_string([value doubleValue]));
    } else if constexpr (std::is_same_v<Layout, Lfm2Layout>) {
      requireNumbers(text, {{"hidden_size", layout.hiddenSize}, {"num_hidden_layers", layout.layers},
          {"vocab_size", layout.vocabularySize},
          {"num_attention_heads", layout.attentionQueryHeads},
          {"num_key_value_heads", layout.attentionKvHeads},
          {"conv_dim", layout.convolutionDimension}, {"conv_L_cache", Lfm2Layout::convolutionTaps},
          {"intermediate_size", layout.intermediateSize}});
      // Exports disagree on the trained context (131072 vs 128000).
      requireAtMost(text, @"max_position_embeddings", layout.maximumContextTokens,
                    "max_position_embeddings");
      // The conv/full_attention layer schedule of layer_types.
      NSArray *types = requireArray(text, @"layer_types", "target layer_types");
      requireEqual(types.count, layout.layers, "target layer_types count");
      for (uint32_t layer = 0; layer < layout.layers; ++layer) {
        id value = types[layer];
        if (![value isKindOfClass:[NSString class]])
          throw std::invalid_argument("target layer type must be a string");
        const std::string expected =
            layout.isFullAttentionLayer(layer) ? "full_attention" : "conv";
        const char *actual = static_cast<NSString *>(value).UTF8String;
        requireEqual(actual ? actual : "", expected,
                     "target layer " + std::to_string(layer));
      }
    } else if constexpr (std::is_same_v<Layout, Lfm2MoeLayout>) {
      requireNumbers(text, {{"hidden_size", layout.hiddenSize}, {"num_hidden_layers", layout.layers},
          {"vocab_size", layout.vocabularySize},
          {"num_attention_heads", layout.attentionQueryHeads},
          {"num_key_value_heads", layout.attentionKvHeads},
          {"conv_dim", layout.convolutionDimension}, {"conv_L_cache", Lfm2MoeLayout::convolutionTaps},
          {"intermediate_size", layout.intermediateSize},
          {"num_experts", layout.experts}, {"num_experts_per_tok", layout.expertsPerToken},
          {"moe_intermediate_size", layout.expertIntermediateSize},
          {"num_dense_layers", Lfm2MoeLayout::denseLayers}});
      // Exports disagree on the trained context (131072 vs 128000).
      requireAtMost(text, @"max_position_embeddings", layout.maximumContextTokens,
                    "max_position_embeddings");
      // The conv/full_attention layer schedule of layer_types.
      NSArray *types = requireArray(text, @"layer_types", "target layer_types");
      requireEqual(types.count, layout.layers, "target layer_types count");
      for (uint32_t layer = 0; layer < layout.layers; ++layer) {
        id value = types[layer];
        if (![value isKindOfClass:[NSString class]])
          throw std::invalid_argument("target layer type must be a string");
        const std::string expected =
            layout.isFullAttentionLayer(layer) ? "full_attention" : "conv";
        const char *actual = static_cast<NSString *>(value).UTF8String;
        requireEqual(actual ? actual : "", expected,
                     "target layer " + std::to_string(layer));
      }
    } else if constexpr (std::is_same_v<Layout, Gemma4MoeLayout> ||
                         std::is_same_v<Layout, DiffusionGemmaLayout>) {
      requireNumbers(text, {{"hidden_size", layout.hiddenSize}, {"num_hidden_layers", layout.layers},
          {"vocab_size", layout.vocabularySize}, {"num_attention_heads", layout.attentionQueryHeads},
          {"num_key_value_heads", layout.attentionKvHeads}, {"head_dim", layout.attentionHeadDimension},
          {"global_num_key_value_heads", layout.globalKvHeads},
          {"global_head_dim", layout.globalHeadDimension},
          {"num_experts", layout.experts}, {"num_experts_per_tok", layout.expertsPerToken},
          {"moe_intermediate_size", layout.expertIntermediateSize},
          {"shared_expert_intermediate_size", layout.sharedIntermediateSize},
          {"sliding_window", layout.slidingWindowTokens}});
      requireAtMost(text, @"max_position_embeddings", layout.maximumContextTokens,
                    "max_position_embeddings");
    }
    // The Qwen families' text config is checked by validateTextConfig once
    // the target format is known.
  }, result.target);
  const auto target = requireString(record, @"target_format", "target format");
  if (target == "mlx-affine") result.targetSource = TargetSource::Mlx;
  else if (target == "gguf") result.targetSource = TargetSource::Gguf;
  else throw std::invalid_argument("unsupported target source format: " + target);
  std::visit([&](const auto &layout) {
    using Layout = std::decay_t<decltype(layout)>;
    if constexpr (std::is_same_v<Layout, Qwen3_8Layout> ||
                  std::is_same_v<Layout, Ornith9BLayout> ||
                  std::is_same_v<Layout, Qwen3_6MoeLayout>)
      validateTextConfig(text, layout, result.targetSource);
  }, result.target);
  // Gemma 4's default draft is Null, but an assembly that ships a draft/
  // config.json declares one anyway (its plain DFlash block).
  const bool gemmaDraftDeclared =
      std::holds_alternative<Gemma4MoeLayout>(result.target) &&
      std::filesystem::exists(root / "draft" / "config.json");
  if (std::holds_alternative<DiffusionGemmaLayout>(result.target) &&
      std::filesystem::exists(root / "draft" / "config.json"))
    throw std::invalid_argument("diffusion gemma targets take no draft");
  if (result.draft.kind != DraftKind::Null || gemmaDraftDeclared) {
  NSDictionary *draft = readObject(root / "draft" / "config.json", "draft config");
  NSArray *architectures = requireArray(draft, @"architectures", "draft architectures");
  if (architectures.count != 1)
    throw std::invalid_argument("draft must declare exactly one architecture");
  const std::string draftArch = architectures[0] && [architectures[0] isKindOfClass:[NSString class]]
      ? std::string(static_cast<NSString *>(architectures[0]).UTF8String) : "";
  const bool dflashV1Draft = draftArch == "DFlashDraftModel";
  const bool dsparkDraft = isDSparkArchitecture(draftArch);
  if (!dflashV1Draft && !dsparkDraft && draftArch != "DFlash2DraftModel")
    throw std::invalid_argument("draft is not a DFlash, DFlash2 or DSpark model");
  DFlashDraftLayout d = result.draft;
  if (!dflashV1Draft && !dsparkDraft && d.kind == DraftKind::DFlashV1) {
    // The descriptor's default draft for this target is plain, but the
    // assembly declares a DFlash2 draft: swap in its DFlash2 layout.
    std::visit(
        [&](const auto &layout) {
          using Layout = std::decay_t<decltype(layout)>;
          if constexpr (std::is_same_v<Layout, Ornith9BLayout>)
            d = ornith9DFlash2DraftLayout();
          else if constexpr (std::is_same_v<Layout, Qwen3_6MoeLayout>)
            d = qwen36DraftLayout();
          else
            d = DFlashDraftLayout{};
        },
        result.target);
    result.draft = d;
    result.stateLayout.draft = d.stateLayout();
  }
  // A DFlash2 draft's config is the checkpoint's: one rule checks it before
  // the dialect parsers below read the fields it names.
  if (!dflashV1Draft && !dsparkDraft) {
    std::visit([&](const auto &layout) {
      result.draft.blockSize = validateDraftConfig(draft, result.draft,
                                                 layout.maskToken,
                                                 layout.hiddenCaptureLayers);
    }, result.target);
  }
  if (dflashV1Draft) {
    // The plain transformer draft: the layout is read from the config.
    // The compiled attention core fixes the head pattern at 32 x 8 x 128
    // over a 6144-wide fused QKV (DraftAttention.cpp's kernel shapes).
    d = {};
    d.kind = DraftKind::DFlashV1;
    // No dynamic convolutions or candidate selector exist in this format.
    d.dynamicSize = 0;
    d.selectorRank = 0;
    d.layers = static_cast<uint32_t>(
        requireUnsigned(draft, @"num_hidden_layers", "num_hidden_layers"));
    d.hiddenSize = static_cast<uint32_t>(
        requireUnsigned(draft, @"hidden_size", "hidden_size"));
    d.vocabularySize = static_cast<uint32_t>(
        requireUnsigned(draft, @"vocab_size", "vocab_size"));
    d.intermediateSize = static_cast<uint32_t>(
        requireUnsigned(draft, @"intermediate_size", "intermediate_size"));
    const uint64_t heads =
        requireUnsigned(draft, @"num_attention_heads", "num_attention_heads");
    const uint64_t kvHeads =
        requireUnsigned(draft, @"num_key_value_heads", "num_key_value_heads");
    const uint64_t headDim = requireUnsigned(draft, @"head_dim", "head_dim");
    requireEqual(heads, 32, "num_attention_heads");
    requireEqual(kvHeads, 8, "num_key_value_heads");
    requireEqual(headDim, 128, "head_dim");
    d.kvHeads = static_cast<uint32_t>(kvHeads);
    d.attentionHeadDimension = static_cast<uint32_t>(headDim);
    d.attentionSize = static_cast<uint32_t>(heads * headDim);
    d.qkvSize = static_cast<uint32_t>((heads + 2 * kvHeads) * headDim);
    d.slidingWindow = static_cast<uint32_t>(std::min<uint64_t>(
        requireUnsigned(draft, @"sliding_window", "draft sliding_window"),
        ExecutionLimits::draftRingCapacity));
    // Sliding layers attend causally inside the block unless the config
    // overrides is_causal; a full_attention layer never does.
    NSArray *types = requireArray(draft, @"layer_types", "draft layer_types");
    requireEqual(types.count, d.layers, "draft layer_types count");
    for (uint32_t layer = 0; layer < d.layers; ++layer) {
      id value = types[layer];
      if (![value isKindOfClass:[NSString class]])
        throw std::invalid_argument("draft layer type must be a string");
      const std::string layerType = static_cast<NSString *>(value).UTF8String;
      if (layerType == "sliding_attention") d.causalLayers |= uint32_t{1} << layer;
      else if (layerType != "full_attention")
        throw std::invalid_argument("unsupported draft layer type: " + layerType);
    }
    id causalFlag = draft[@"is_causal"];
    if ([causalFlag isKindOfClass:[NSNumber class]]) {
      const uint32_t all = (uint32_t{1} << d.layers) - 1;
      d.causalLayers = [causalFlag boolValue] ? all : 0;
    }
    // Gemma 4's draft shares its target's logit softcap; other families'
    // plain drafts leave it disabled.
    std::visit([&](const auto &layout) {
      if constexpr (std::is_same_v<std::decay_t<decltype(layout)>,
                                   Gemma4MoeLayout>)
        d.logitSoftcap = layout.logitSoftcap;
    }, result.target);
  } else if (dsparkDraft) {
    // The DSpark draft: a block-bidirectional qwen3 block plus a Markov and
    // a confidence head. MiniCPM5's config states its fields flat, LFM2.5's
    // nests the block fields under dflash_config; both carry the same
    // tensors. The layer stack shares the plain draft's section order.
    d = {};
    d.kind = DraftKind::DSpark;
    d.dynamicSize = 0;
    d.selectorRank = 0;
    d.layers = static_cast<uint32_t>(
        requireUnsigned(draft, @"num_hidden_layers", "num_hidden_layers"));
    d.hiddenSize = static_cast<uint32_t>(
        requireUnsigned(draft, @"hidden_size", "hidden_size"));
    d.vocabularySize = static_cast<uint32_t>(
        requireUnsigned(draft, @"vocab_size", "vocab_size"));
    d.intermediateSize = static_cast<uint32_t>(
        requireUnsigned(draft, @"intermediate_size", "intermediate_size"));
    const uint64_t heads =
        requireUnsigned(draft, @"num_attention_heads", "num_attention_heads");
    const uint64_t kvHeads =
        requireUnsigned(draft, @"num_key_value_heads", "num_key_value_heads");
    const uint64_t headDim =
        requireUnsigned(draft, @"head_dim", "head_dim");
    const bool supported =
        (heads == 16 && kvHeads == 2 && headDim == 128) ||
        (heads == 32 && kvHeads == 8 && headDim == 64) ||
        (heads == 32 && kvHeads == 8 && headDim == 128);
    if (!supported)
      throw std::invalid_argument("unsupported DSpark draft head geometry");
    d.kvHeads = static_cast<uint32_t>(kvHeads);
    d.attentionHeadDimension = static_cast<uint32_t>(headDim);
    d.attentionSize = static_cast<uint32_t>(heads * headDim);
    d.qkvSize = static_cast<uint32_t>((heads + 2 * kvHeads) * headDim);
    d.markovRank = static_cast<uint32_t>(
        requireUnsigned(draft, @"markov_rank", "draft markov_rank"));
    requireEqual(d.markovRank, RICHENGINE_DRAFT_SELECTOR_RANK,
                 "draft markov_rank");
    if (id markovType = draft[@"markov_head_type"])
      requireEqual(
          std::string(static_cast<NSString *>(markovType).UTF8String),
          "vanilla", "draft markov_head_type");
    // rope_is_neox_style = false (LFM2.5) selects the interleaved pairing;
    // the field's absence means the half-split default.
    d.ropeInterleaved = [draft[@"rope_is_neox_style"] isEqual:@NO] ? 1 : 0;
    if (draft[@"enable_confidence_head"] &&
        ![draft[@"enable_confidence_head"] isEqual:@YES])
      throw std::invalid_argument("unsupported DSpark confidence head");
    NSArray *types = requireArray(draft, @"layer_types", "draft layer_types");
    requireEqual(types.count, d.layers, "draft layer_types count");
    for (uint32_t layer = 0; layer < d.layers; ++layer) {
      id value = types[layer];
      if (![value isKindOfClass:[NSString class]] ||
          std::string(static_cast<NSString *>(value).UTF8String) !=
              "full_attention")
        throw std::invalid_argument("unsupported DSpark draft layer type");
    }
    d.causalLayers = 0;
  } else {
    requireNumbers(draft, {{"num_hidden_layers", d.layers}, {"hidden_size", d.hiddenSize},
        {"vocab_size", d.vocabularySize}, {"intermediate_size", d.intermediateSize},
        {"num_attention_heads", d.attentionSize / d.attentionHeadDimension},
        {"num_key_value_heads", d.kvHeads}, {"head_dim", d.attentionHeadDimension},
        {"sliding_window", ExecutionLimits::draftContextTokens}});
    if (![draft[@"is_causal"] isEqual:@NO])
      throw std::invalid_argument("unsupported draft attention configuration");
  }
  // The draft norm epsilon follows the target family's (the LFM2 targets'
  // 1e-5).
  const double expectedRmsEps =
      std::holds_alternative<Lfm2Layout>(result.target) ||
              std::holds_alternative<Lfm2MoeLayout>(result.target)
          ? 1e-5
          : 1e-6;
  if (dsparkDraft) {
    d.rmsEpsilon = static_cast<float>(expectedRmsEps);
    if (![draft[@"rms_norm_eps"] isEqual:@(expectedRmsEps)] ||
        ![draft[@"hidden_act"] isEqual:@"silu"])
      throw std::invalid_argument(
          "unsupported DSpark draft normalization configuration");
    if (draft[@"attention_bias"] && ![draft[@"attention_bias"] isEqual:@NO])
      throw std::invalid_argument("unsupported DSpark draft attention bias");
    if (draft[@"tie_word_embeddings"] &&
        ![draft[@"tie_word_embeddings"] isEqual:@NO])
      throw std::invalid_argument("unsupported DSpark draft tied embeddings");
  } else if (![draft[@"attention_bias"] isEqual:@NO] ||
             ![draft[@"tie_word_embeddings"] isEqual:@NO] ||
             ![draft[@"rms_norm_eps"] isEqual:@(expectedRmsEps)] ||
             ![draft[@"hidden_act"] isEqual:@"silu"]) {
    throw std::invalid_argument("unsupported draft attention or normalization configuration");
  }
  // MiniCPM5's DSpark config states rope_parameters.rope_theta; LFM2.5's a
  // flat rope_theta. A plain draft's RoPE base is its own (the dense
  // target's draft shares its 5e6); a DFlash2 draft's is fixed at 1e7.
  uint64_t ropeTheta;
  if (id rope = draft[@"rope_parameters"];
      [rope isKindOfClass:[NSDictionary class]]) {
    requireEqual(requireString(rope, @"rope_type", "draft rope type"),
                 "default", "draft rope type");
    ropeTheta = requireUnsigned(rope, @"rope_theta", "draft rotary theta");
  } else {
    ropeTheta = requireUnsigned(draft, @"rope_theta", "draft rotary theta");
  }
  if (dsparkDraft || dflashV1Draft)
    d.rotaryTheta = static_cast<float>(ropeTheta);
  else
    requireEqual(ropeTheta, 10000000, "draft rotary theta");
  // The dflash fields sit nested under dflash_config in DFlash/DFLASH2 and
  // LFM2.5 DSpark configs; MiniCPM5's DSpark states them at the root.
  NSDictionary *flash = draft[@"dflash_config"];
  if (![flash isKindOfClass:[NSDictionary class]]) {
    if (dsparkDraft)
      flash = draft;
    else
      flash = requireObject(draft, @"dflash_config", "draft configuration");
  }
  if (dflashV1Draft) {
    const uint64_t blockSize =
        requireUnsigned(flash, @"block_size", "draft block_size");
    // The runtime always dispatches draftQueryRows query rows: a shorter
    // trained block is padded with mask rows whose proposals never accept.
    if (blockSize < 2)
      throw std::invalid_argument("draft block_size is below a proposal row");
    d.blockSize = static_cast<uint32_t>(blockSize);
  } else if (dsparkDraft) {
    // block_size is a root field in both DSpark dialects; only LFM2.5 nests
    // the mask and capture fields under dflash_config.
    d.blockSize = static_cast<uint32_t>(
        requireUnsigned(draft, @"block_size", "draft block_size"));
    // The runtime always dispatches draftQueryRows query rows: a shorter
    // trained block is padded with mask rows, a longer one truncated to the
    // proposal budget.
    if (d.blockSize < 1)
      throw std::invalid_argument("draft block_size is below a proposal row");
    if (id layers = flash[@"num_target_layers"]) {
      std::visit([&](const auto &layout) {
        requireEqual(requireUnsigned(flash, @"num_target_layers",
                                     "draft num_target_layers"),
                     layout.layers, "draft num_target_layers");
      }, result.target);
      static_cast<void>(layers);
    }
  } else {
    const uint64_t blockSize =
        requireUnsigned(flash, @"block_size", "draft block_size");
    // A DFlash2 block_size counts the block's query rows; the runtime may
    // dispatch wider and cap acceptance at the trained block's proposals.
    if (blockSize < 2)
      throw std::invalid_argument("draft block_size is below a proposal row");
    d.blockSize = static_cast<uint32_t>(blockSize);
    result.draft.blockSize = d.blockSize;
    requireNumbers(flash, {{"conv_group_size", 16},
        {"conv_kernel_size", 2}, {"selector_rank", d.selectorRank}, {"selector_top_k", 16}});
  }
  NSArray *capture = requireArray(flash, @"target_layer_ids", "draft target layers");
  std::visit([&](const auto &layout) {
    requireEqual(requireUnsigned(flash, @"mask_token_id", "draft mask token"), layout.maskToken, "draft mask token");
    requireEqual(capture.count, layout.hiddenCaptureLayers.size(), "draft target layer count");
    for (size_t i = 0; i < layout.hiddenCaptureLayers.size(); ++i) {
      id value = capture[i];
      if (![value isKindOfClass:[NSNumber class]] || [value unsignedLongLongValue] != layout.hiddenCaptureLayers[i])
        throw std::invalid_argument("draft target capture layers do not match this model");
    }
    if (dflashV1Draft || dsparkDraft)
      d.targetHiddenSize = static_cast<uint32_t>(
          capture.count * layout.hiddenSize);
  }, result.target);
  if (dflashV1Draft || dsparkDraft) {
    result.draft = d;
    result.stateLayout.draft = d.stateLayout();
  }
  }

  const auto vision = requireString(record, @"vision_format", "vision format");
  if (vision == "none") result.visionSource = VisionSource::None;
  else {
    // Ornith and the new text-only targets ship no vision tower this
    // runtime supports.
    if (std::holds_alternative<Ornith9BLayout>(result.target) ||
        std::holds_alternative<DenseLayout>(result.target) ||
        std::holds_alternative<GraniteLayout>(result.target) ||
        std::holds_alternative<Lfm2Layout>(result.target) ||
        std::holds_alternative<Lfm2MoeLayout>(result.target) ||
        std::holds_alternative<Gemma4MoeLayout>(result.target) ||
        std::holds_alternative<DiffusionGemmaLayout>(result.target))
      throw std::invalid_argument("unsupported vision source format: " + vision);
    if (vision == "safetensors") result.visionSource = VisionSource::Mlx;
    else if (vision == "gguf") result.visionSource = VisionSource::Gguf;
    else throw std::invalid_argument("unsupported vision source format: " + vision);
    NSDictionary *v = requireObject(config, @"vision_config", "vision config");
    const auto &l = result.vision;
    requireNumbers(v, "vision config", {{"depth", l.depth}, {"hidden_size", l.hiddenSize}, {"num_heads", l.heads},
        {"intermediate_size", l.intermediateSize}, {"out_hidden_size", l.outputHiddenSize},
        {"patch_size", l.patchSize}, {"spatial_merge_size", l.spatialMerge},
        {"temporal_patch_size", 2}, {"in_channels", 3}, {"num_position_embeddings", l.positionGridSide * l.positionGridSide}});
    requireEqual(requireString(v, @"hidden_act", "vision activation"), "gelu_pytorch_tanh", "vision activation");
    NSArray *deepstack = requireArray(v, @"deepstack_visual_indexes", "vision deepstack layers");
    if (deepstack.count) throw std::invalid_argument("vision deepstack layers are unsupported");
  }
  result.sourceIdentity = std::move(sourceIdentity);
  if (!result.valid()) throw std::invalid_argument("incompatible target and draft model");
  return result;
}

} // namespace

ModelDescriptor makeModelDescriptor(std::string name, TargetLayout target,
                                    DFlashDraftLayout draft,
                                    ops::VisionLayout vision) {
  ModelDescriptor result;
  result.name = std::move(name);
  result.target = target;
  result.draft = draft;
  result.vision = vision;
  std::visit(
      [&](const auto &layout) {
        result.capabilities = {layout.vocabularySize,
                               layout.maximumContextTokens};
        if constexpr (std::is_same_v<std::decay_t<decltype(layout)>,
                                     DiffusionGemmaLayout>)
          result.capabilities.maximumStepTokens =
              layout.diffusion.canvasLength;
        result.targetKvLayout = layout.kvLayout();
        result.stateLayout = {layout.gdnStateLayout(), draft.stateLayout()};
      },
      target);
  // Tree-verify default follows the draft kind a family picked: a Null
  // draft's only proposer is the n-gram predraft, whose alternates rescue
  // draft-free lanes cheaply, so its comb trees default on; the GPU drafts
  // stay env opt-in — a measured DFlash/DSpark comb verify costs more than
  // its leaves rescue under a 16-row stride. A maker that wants Off (or to
  // force On) edits result.tuning after this call.
  result.tuning.treeVerify = draft.kind == DraftKind::Null
                                 ? ModelTuning::TreeVerify::On
                                 : ModelTuning::TreeVerify::OptIn;
  return result;
}

bool ModelDescriptor::valid() const noexcept {
  if (name.empty() || !capabilities.vocabularySize ||
      !capabilities.maximumContextTokens ||
      !targetKvLayout.valid() || !stateLayout.valid() ||
      stateLayout.draft != draft.stateLayout() ||
      vision.outputHiddenSize != draft.hiddenSize) {
    return false;
  }
  return std::visit(
      [&](const auto &layout) {
        return layout.vocabularySize == capabilities.vocabularySize &&
               layout.maximumContextTokens ==
                   capabilities.maximumContextTokens &&
               layout.hiddenSize == draft.hiddenSize &&
               layout.capturedHiddenSize() == draft.targetHiddenSize &&
               layout.kvLayout() == targetKvLayout &&
               layout.gdnStateLayout() == stateLayout.target;
      },
      target);
}

ModelDescriptor inspectModelPackage(const std::filesystem::path &root) {
  @autoreleasepool {
    if (std::filesystem::exists(root / "model.json")) return inspectSourceModel(root);
    std::string sourceIdentity;
    NSDictionary *manifest = readObject(root / "manifest.json", "model manifest", &sourceIdentity);
    validateExecutionGeometry(manifest);
    const std::string model = requireString(manifest, @"model", "model name");
    const std::string format = requireString(
        requireObject(manifest, @"format", "model weight format"),
        @"name", "weight format");
    // Packages published before the rename name their formats splash-packed-*;
    // they are byte-identical to the richengine-packed-* names.
    const std::string canonicalFormat =
        format.starts_with("splash-packed-")
            ? "richengine-packed-" + format.substr(14)
            : format;
    ModelDescriptor descriptor;
    if (canonicalFormat == "richengine-packed-q4") {
      if (manifest[@"family"]) {
        // A manifest written by install/pack.py names its family outright;
        // it selects the descriptor the same way a source record's does.
        const std::string family =
            requireString(manifest, @"family", "model family");
        const auto entry = std::ranges::find(kModelFamilies, family,
                                             &FamilyDescriptor::name);
        if (entry == std::end(kModelFamilies))
          throw std::invalid_argument("unsupported model family: " + family);
        descriptor = entry->make(model);
      } else {
        // Both dense families share the packed format; a legacy manifest
        // without a family names the model by the tokenizer config's
        // hidden size.
        NSDictionary *tokenizerConfig =
            readObject(root / "tokenizer" / "config.json", "tokenizer model config");
        NSDictionary *text = requireObject(tokenizerConfig, @"text_config",
                                           "text model config");
        const uint64_t hiddenSize =
            requireUnsigned(text, @"hidden_size", "hidden_size");
        if (hiddenSize == Ornith9BLayout{}.hiddenSize)
          descriptor = ornithDescriptor(model);
        else
          descriptor = qwen38Descriptor(model);
      }
      NSDictionary *weightFormat =
          requireObject(manifest, @"format", "model weight format");
      const DFlashDraftLayout dflashV1Layout = ornith9DFlashV1DraftLayout();
      const bool ornith =
          std::holds_alternative<Ornith9BLayout>(descriptor.target);
      applyManifestDraftKind(
          weightFormat, descriptor, ornith ? &dflashV1Layout : nullptr,
          ornith ? ornith9DFlash2DraftLayout() : DFlashDraftLayout{});
      validateQwen38(manifest, root, descriptor);
    } else if (canonicalFormat == "richengine-packed-q4-moe") {
      descriptor = qwen36Descriptor(model);
      NSDictionary *weightFormat =
          requireObject(manifest, @"format", "model weight format");
      const DFlashDraftLayout dflashV1Layout = qwen36DFlashV1DraftLayout();
      applyManifestDraftKind(weightFormat, descriptor, &dflashV1Layout,
                             qwen36DraftLayout());
      validateQwen36(manifest, root, descriptor);
    } else if (canonicalFormat == "richengine-packed-q4-dense") {
      descriptor = denseDescriptor(model);
      NSDictionary *weightFormat =
          requireObject(manifest, @"format", "model weight format");
      applyDeclaredDraft(manifest, weightFormat, descriptor, denseDraftLayout());
      validateNewPackedFormat(manifest, root, descriptor, "minicpm");
    } else if (canonicalFormat == "richengine-packed-q4-lfm2") {
      descriptor = lfm2Descriptor(model);
      NSDictionary *weightFormat =
          requireObject(manifest, @"format", "model weight format");
      applyDeclaredDraft(manifest, weightFormat, descriptor, lfm2DraftLayout());
      validateNewPackedFormat(manifest, root, descriptor, "lfm2");
    } else if (canonicalFormat == "richengine-packed-q4-lfm2moe") {
      descriptor = lfm2moeDescriptor(model);
      NSDictionary *weightFormat =
          requireObject(manifest, @"format", "model weight format");
      applyDeclaredDraft(manifest, weightFormat, descriptor, lfm2moeDraftLayout());
      validateNewPackedFormat(manifest, root, descriptor, "lfm2_moe");
    } else if (canonicalFormat == "richengine-packed-q4-gemma4") {
      descriptor = gemma4Descriptor(model);
      descriptor.packedTiledEmbedding = true;
      // A package whose manifest declares a draft ships the plain DFlash
      // block; one without keeps the Null layout and its n-gram predraft.
      if ([manifest[@"draft"] isKindOfClass:[NSDictionary class]]) {
        NSDictionary *weightFormat =
            requireObject(manifest, @"format", "model weight format");
        applyDeclaredDraft(manifest, weightFormat, descriptor,
                           gemma4DFlashV1DraftDefaults());
      }
      validateGemma4(manifest, root, descriptor);
    } else if (canonicalFormat == "richengine-packed-q4-diffusiongemma") {
      descriptor = diffusionGemmaDescriptor(model);
      descriptor.packedTiledEmbedding = true;
      validateDiffusionGemma(manifest, root, descriptor);
    } else {
      throw std::invalid_argument("unsupported weight format: " + format);
    }
    descriptor.sourceIdentity = std::move(sourceIdentity);
    if (!descriptor.valid())
      throw std::logic_error("built-in model descriptor is inconsistent");
    return descriptor;
  }
}

} // namespace richengine::model
