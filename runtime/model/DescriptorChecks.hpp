#pragma once

// The manifest/source checks a family's ModelFamilyRegistration hooks call.
// inspectSourceModel's per-family config blocks and the packed manifest's
// format validators used to dispatch on TargetLayout::arch inside
// ModelDescriptor.mm; the per-family parts moved to the families'
// translation units, which reach the shared require* plumbing through
// these declarations (the definitions stay in ModelDescriptor.mm).
#import <Foundation/Foundation.h>

#include "model/ModelDescriptor.hpp"
#include "model/TargetLayout.hpp"

#include <initializer_list>
#include <string_view>

namespace richengine::model {

// A manifest field's name and the number it must equal.
struct GeometryField final {
  const char *name;
  uint64_t value;
};

NSArray *requireArray(NSDictionary *object, NSString *key,
                      std::string_view label);
void requireEqual(uint64_t actual, uint64_t expected,
                  std::string_view label);
void requireEqual(std::string_view actual, std::string_view expected,
                  std::string_view label);
void requireAtMost(NSDictionary *object, NSString *key, uint64_t maximum,
                   std::string_view label);
void requireNumber(id value, double expected, std::string_view label);
// The numbers object holds at the GeometryField keys, which errors name
// after each field's own name. (Named apart from the file-local
// requireNumbers overloads it would hide from them.)
void requireGeometryNumbers(NSDictionary *object,
                            std::initializer_list<GeometryField> fields);

// The target's text configuration: the Qwen families' refineSourceConfig
// hook (refineQwenSourceText) calls it once the record's target_format is
// known, as the source-dialect checks differ between MLX and GGUF.
void validateTextConfig(NSDictionary *text, const TargetLayout &target,
                        TargetSource source);
// The shared refineSourceConfig registration of every family whose text
// config follows the Qwen schema.
void refineQwenSourceText(id text, const ModelDescriptor &descriptor);

} // namespace richengine::model
