// The verify-geometry dispatcher. Each engine variant is the same sources
// compiled under its own namespace (the 8-row draft/verify geometry as
// richengine_8 via -Drichengine=richengine_8) with its own metallib; the
// kernel ABI and tile shapes bake the row counts, so a process runs exactly
// one. This main() reads the model's draft geometry from its files and calls
// the matching namespace's serveMain — what server/model_host.py's binary
// pick did when the variants were separate binaries.
//
// Compiled once, without the namespace rename: `richengine` names the
// default (16-row) variant, `richengine_8` the block-8 one.

#import <Foundation/Foundation.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <string_view>
#include <system_error>

#ifndef RICHENGINE_BUILD_ID
#error "production build requires the generated BuildIdentity.hpp"
#endif

namespace richengine { int serveMain(int argc, char **argv); }
namespace richengine_8 { int serveMain(int argc, char **argv); }

// The one embedded build-id string both variants record
// (dev/tools/build_identity.py requires exactly one in a binary).
const char *richengineBuildId() noexcept { return RICHENGINE_BUILD_ID; }

namespace {

// A dictionary entry that is a positive integer, or 0.
uint32_t jsonInt(id object) {
  if (![object isKindOfClass:NSNumber.class]) return 0;
  const int value = [object intValue];
  return value > 0 ? uint32_t(value) : 0;
}

NSDictionary *readJson(const std::filesystem::path &path) {
  @autoreleasepool {
    NSError *error = nil;
    NSData *data = [NSData dataWithContentsOfFile:
        [NSString stringWithUTF8String:path.c_str()] options:0 error:&error];
    if (!data) return nil;
    id object = [NSJSONSerialization JSONObjectWithData:data
                                                options:0 error:&error];
    return [object isKindOfClass:NSDictionary.class] ? object : nil;
  }
}

// The verify rows the model's draft trains at: 0 when unknown (the default
// 16-row variant then serves it). Mirrors ModelDescriptor's draft spellings:
// DFlash*/LFM drafts keep their block under dflash_config.block_size (the
// query rows); DSpark's root block_size is its proposal count, its verify
// window one more. Packed manifests state the compiled geometry directly.
uint32_t probeVerifyRows(const std::filesystem::path &root) {
  @autoreleasepool {
    if (NSDictionary *draft = readJson(root / "draft" / "config.json")) {
      if (const uint32_t block =
              jsonInt([draft objectForKey:@"dflash_config"] == nil
                          ? nil
                          : [draft[@"dflash_config"] objectForKey:@"block_size"]))
        return block;
      if (const uint32_t block = jsonInt(draft[@"block_size"]))
        return block + 1;
    }
    if (NSDictionary *manifest = readJson(root / "manifest.json")) {
      if ([manifest[@"execution_geometry"] isKindOfClass:NSDictionary.class])
        return jsonInt(manifest[@"execution_geometry"][@"draft_query_rows"]);
    }
    return 0;
  }
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    int rows = 0;
    if (const char *forced = std::getenv("RICHENGINE_VERIFY_GEOMETRY")) {
      if (!std::strcmp(forced, "8")) rows = 8;
      else if (!std::strcmp(forced, "16")) rows = 16;
    }
    if (!rows && argc >= 3 && std::string_view(argv[1]) == "serve-native") {
      std::error_code error;
      const std::filesystem::path root =
          std::filesystem::canonical(std::filesystem::path(argv[2]), error);
      if (!error) rows = int(probeVerifyRows(root));
      if (rows > 0 && rows <= 8)
        fprintf(stderr, "Verify geometry · 8-row block\n");
    }
    if (rows > 0 && rows <= 8)
      return richengine_8::serveMain(argc, argv);
    return richengine::serveMain(argc, argv);
  }
}
