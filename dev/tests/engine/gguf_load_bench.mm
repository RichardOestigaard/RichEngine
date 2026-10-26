// Times a real GGUF load through the production preparation path. With
// RICHENGINE_REPACK_TIMING=1 writeRepack logs its read/kernel split per
// repack.
//   gguf-load-bench METALLIB MODEL.gguf
#include "metal/MetalBackend.hpp"
#include "model/GgufTarget.hpp"
#include "model/Lfm2Moe.hpp"
#include "model/Lfm2Moe24B.hpp"
#include "model/Ornith9B.hpp"
#include "model/WeightImages.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc != 4 || (std::strcmp(argv[3], "lfm2moe") && std::strcmp(argv[3], "lfm2moe24b") &&
                      std::strcmp(argv[3], "ornith9b"))) {
      std::fprintf(stderr, "usage: gguf-load-bench METALLIB MODEL.gguf lfm2moe|lfm2moe24b|ornith9b\n");
      return 2;
    }
    using namespace richengine;
    const auto begin = std::chrono::steady_clock::now();
    metal::MetalBackend backend(argv[1]);
    model::WeightImages images(backend);
    model::gguf::TargetGeometry geometry;
    if (!std::strcmp(argv[3], "lfm2moe"))
      geometry = model::ggufTargetGeometry(model::lfm2MoeTargetLayout());
    else if (!std::strcmp(argv[3], "lfm2moe24b"))
      geometry = model::ggufTargetGeometry(model::lfm2Moe24BTargetLayout());
    else
      geometry = model::ggufTargetGeometry(model::ornith9TargetLayout());
    model::GgufTargetLoader loader(backend, images, argv[2], geometry);
    for (uint32_t layer = 0; layer < geometry.layers; ++layer) static_cast<void>(loader.layer(layer));
    static_cast<void>(loader.head());
    static_cast<void>(loader.embedding());
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    std::fprintf(stderr, "gguf-load-bench: %.3fs\n", seconds);
    return 0;
  }
}
