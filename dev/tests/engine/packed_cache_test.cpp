// The packed weight cache's shared-root keying, write completion and legacy
// in-package fallback, without loading a model.
#include "model/AffinePreparation.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

using namespace richengine::model;
namespace fs = std::filesystem;

namespace {

void write(const fs::path &path, std::string_view bytes) {
  fs::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out << bytes;
}

void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

} // namespace

int main() {
  try {
    const fs::path root = fs::temp_directory_path() /
        ("richengine-packed-cache-test-" + std::to_string(::getpid()));
    fs::remove_all(root);
    const fs::path source = root / "source";
    const fs::path sharedRoot = root / "shared";
    fs::create_directories(source);
    write(source / "model.safetensors", "source weights");

    std::vector<affine::Image> images(1);
    images[0].name = "layer-0.bin";
    images[0].magic = "wgt0";
    images[0].bytes = 32;

    ::setenv("RICHENGINE_PACKED_CACHE_ROOT", sharedRoot.c_str(), 1);
    const fs::path shared = affine::packedWriteDirectory(source, images);
    require(shared != source, "packedWriteDirectory did not choose the shared root");
    require(shared.parent_path() == sharedRoot, "the shared cache is not content-keyed below its root");

    const std::vector<uint8_t> payload(32, 'x');
    affine::cachePackedImage(shared / images[0].name, payload);
    require(affine::packedReadDirectory(source, images).empty(),
            "an incomplete cache became readable before its stamp");
    affine::refreshPackedStamp(shared, source, images);
    require(affine::packedReadDirectory(source, images) == shared,
            "a complete shared cache was not reused");

    // Assembly shards are symlinks into the content-addressed blob store:
    // the cache key uses that digest, not the blob's mutable ctime or the
    // snapshot's mtime. Two assemblies of the same shard therefore share one
    // entry, and each use registers the assembly for the cache collector.
    const std::string blobDigest(64, 'b');
    const std::string firstAssembly(64, 'a');
    const std::string secondAssembly(64, 'c');
    const fs::path blob = root / "models/.blobs" / blobDigest;
    write(blob, "source weights");
    const fs::path first = root / "models/.resolved" / firstAssembly / "target";
    const fs::path second = root / "models/.resolved" / secondAssembly / "target";
    fs::create_directories(first);
    fs::create_directories(second);
    fs::create_symlink(blob, first / "model.safetensors");
    fs::create_symlink(blob, second / "model.safetensors");
    // Thin installations persist the shard stamp lines, allowing the same key
    // after the payload symlinks are dropped.
    const std::string thinStamp = "model.safetensors digest " + blobDigest + "\n";
    write(first / "packed-sources", thinStamp);
    write(second / "packed-sources", thinStamp);
    ::setenv("RICHENGINE_PACKED_CACHE_ROOT", sharedRoot.c_str(), 1);
    const fs::path firstShared = affine::packedWriteDirectory(first, images);
    const fs::path secondShared = affine::packedWriteDirectory(second, images);
    require(firstShared == secondShared,
            "identical digest-named shards did not share one packed cache key");
    require(fs::exists(firstShared / ".owners" / firstAssembly),
            "packing did not register the first assembly");
    require(fs::exists(firstShared / ".owners" / secondAssembly),
            "packing did not register the second assembly");
    affine::cachePackedImage(firstShared / images[0].name, payload);
    affine::refreshPackedStamp(firstShared, first, images);
    require(affine::packedReadDirectory(second, images) == firstShared,
            "a content-keyed packed cache was not reused across assemblies");
    require(fs::exists(firstShared / ".validated" / firstAssembly),
            "completing a packed cache did not validate its writer");
    require(fs::exists(firstShared / ".validated" / secondAssembly),
            "reading a packed cache did not validate its reader");
    fs::remove(first / "model.safetensors");
    fs::remove(second / "model.safetensors");
    require(affine::packedReadDirectory(first, images) == firstShared,
            "a thin assembly could not read its packed cache without payload links");

    // A legacy in-package cache remains a read fallback when no shared entry
    // exists. Emptying the variable keeps writes in-package too.
    const fs::path legacySource = root / "legacy";
    fs::create_directories(legacySource);
    write(legacySource / "model.safetensors", "source weights");
    ::setenv("RICHENGINE_PACKED_CACHE_ROOT", "", 1);
    const fs::path legacy = affine::packedWriteDirectory(legacySource, images);
    require(legacy == legacySource, "an empty shared root did not keep legacy writes in-package");
    affine::cachePackedImage(legacy / images[0].name, payload);
    affine::refreshPackedStamp(legacy, legacySource, images);
    require(affine::packedReadDirectory(legacySource, images) == legacySource,
            "a legacy in-package cache was not readable");

    fs::remove_all(root);
    std::cout << "packed cache: content key, owner registry, thin reuse, legacy fallback PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
