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
#include <fstream>
#include <limits>
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

// Bounded little-endian GGUF v3 metadata skipping. The dispatcher runs
// before either variant's GgufFile exists in a namespace it can call, so it
// reads only the one integer its geometry choice needs.
class GgufProbe {
public:
  explicit GgufProbe(const std::filesystem::path &path)
      : stream_(path, std::ios::binary) {}

  template <class T> bool scalar(T &value) {
    return bytes(&value, sizeof value);
  }

  bool bytes(void *destination, uint64_t count) {
    return bool(stream_.read(static_cast<char *>(destination), count));
  }

  bool string(std::string &value) {
    uint64_t length = 0;
    if (!scalar(length) || length > (1u << 20)) return false;
    value.resize(length);
    return !length || bool(stream_.read(value.data(), length));
  }

  bool skip(uint64_t bytes) {
    if (bytes > uint64_t(std::numeric_limits<std::streamoff>::max()))
      return false;
    stream_.seekg(std::streamoff(bytes), std::ios::cur);
    return bool(stream_);
  }

  bool skipString() {
    uint64_t length = 0;
    return scalar(length) && skip(length);
  }

  static uint64_t scalarBytes(uint32_t type) {
    switch (type) {
    case 0: case 1: case 7: return 1;
    case 2: case 3: return 2;
    case 4: case 5: case 6: return 4;
    case 10: case 11: case 12: return 8;
    default: return 0;
    }
  }

  bool skipValue(uint32_t type, unsigned depth = 0) {
    if (depth > 16) return false;
    if (type == 8) return skipString();
    if (type != 9) {
      const uint64_t bytes = scalarBytes(type);
      return bytes && skip(bytes);
    }
    uint32_t element = 0;
    uint64_t count = 0;
    if (!scalar(element) || !scalar(count)) return false;
    if (element == 8 || element == 9) {
      for (uint64_t i = 0; i < count; ++i)
        if (!skipValue(element, depth + 1)) return false;
      return true;
    }
    const uint64_t bytes = scalarBytes(element);
    return bytes && count <= std::numeric_limits<uint64_t>::max() / bytes &&
           skip(count * bytes);
  }

  bool unsignedInteger(uint32_t type, uint64_t &value) {
    switch (type) {
    case 0: { uint8_t x; if (!scalar(x)) return false; value = x; return true; }
    case 1: { int8_t x; if (!scalar(x) || x < 0) return false; value = uint64_t(x); return true; }
    case 2: { uint16_t x; if (!scalar(x)) return false; value = x; return true; }
    case 3: { int16_t x; if (!scalar(x) || x < 0) return false; value = uint64_t(x); return true; }
    case 4: { uint32_t x; if (!scalar(x)) return false; value = x; return true; }
    case 5: { int32_t x; if (!scalar(x) || x < 0) return false; value = uint64_t(x); return true; }
    case 7: { uint8_t x; if (!scalar(x)) return false; value = x != 0; return true; }
    case 10: return scalar(value);
    case 11: { int64_t x; if (!scalar(x) || x < 0) return false; value = uint64_t(x); return true; }
    default: return skipValue(type);
    }
  }

private:
  std::ifstream stream_;
};

// A GGUF-only DFlash draft states dflash.block_size in its metadata. Any
// probe failure reads as unknown geometry, which safely selects the default
// 16-row variant.
uint32_t probeGgufDraftRows(const std::filesystem::path &directory) {
  std::filesystem::path gguf;
  std::error_code error;
  for (const auto &entry : std::filesystem::directory_iterator(directory, error)) {
    if (entry.path().extension() != ".gguf") continue;
    if (!gguf.empty()) return 0;
    gguf = entry.path();
  }
  if (error || gguf.empty()) return 0;
  GgufProbe probe(gguf);
  char magic[4]{};
  uint32_t version = 0;
  uint64_t tensors = 0, keys = 0;
  if (!probe.bytes(magic, 4) || std::memcmp(magic, "GGUF", 4) != 0 ||
      !probe.scalar(version) || version != 3 ||
      !probe.scalar(tensors) || !probe.scalar(keys) ||
      tensors > 16384 || keys > 16384)
    return 0;
  for (uint64_t i = 0; i < keys; ++i) {
    std::string key;
    uint32_t type = 0;
    if (!probe.string(key) || !probe.scalar(type)) return 0;
    if (key == "dflash.block_size") {
      uint64_t rows = 0;
      return probe.unsignedInteger(type, rows) && rows > 0 &&
                     rows <= std::numeric_limits<uint32_t>::max()
                 ? uint32_t(rows)
                 : 0;
    }
    if (!probe.skipValue(type)) return 0;
  }
  return 0;
}

// The verify rows the model's draft trains at: 0 when unknown (the default
// 16-row variant then serves it). Mirrors ModelDescriptor's draft spellings:
// DFlash*/LFM drafts keep their block under dflash_config.block_size (the
// query rows); DSpark's root block_size is its proposal count, its verify
// window one more. Packed manifests state the compiled geometry directly;
// GGUF-only drafts state dflash.block_size in the GGUF's own metadata.
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
    return probeGgufDraftRows(root / "draft");
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
