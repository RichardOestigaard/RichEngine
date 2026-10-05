#pragma once

#include "ops/PageStorage.hpp"

#include <array>
#include <cstdint>

namespace richengine::test {

// The entry kernels reach a page by, as PageStorage writes it into a
// one-entry GPU table.
inline RichKvPage entryOf(const kv::PageStorage &storage, uint32_t page,
                            const metal::MetalBuffer &table) {
    storage.writeEntries(std::array<uint32_t, 1>{page}, 0, table);
    return *static_cast<const RichKvPage *>(table.contents());
}

}  // namespace richengine::test
