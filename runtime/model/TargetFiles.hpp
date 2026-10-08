#pragma once

#include <functional>
#include <variant>

namespace richengine::model {

// Names the files a target is read from without the loaders' headers, so a
// family's header declares its loader alone; TargetLoader.hpp defines
// PackedTargetFiles and reads the files.
template <class Layout> struct PackedTargetFiles;
class AffineTargetLoader;
class GgufTargetLoader;

// The files a target is read from: packed files (richengine-packed-q4 formats),
// or the images a loader writes from an MLX or GGUF source.
template <class Layout>
using TargetFiles = std::variant<PackedTargetFiles<Layout>, std::reference_wrapper<AffineTargetLoader>,
                                     std::reference_wrapper<GgufTargetLoader>>;

} // namespace richengine::model
