#pragma once

#include "metal/MetalBackend.hpp"
#include "model/GgufImage.hpp"
#include "model/WeightSource.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace richengine::model {

// The Metal staging of the rows of one repack step, whatever the tensor,
// layer or expert count. An image's writer holds two (pipelineChunks).
inline constexpr uint64_t kGgufRepackStagingBytes = 32 << 20;

// Runs `count` chunks through two staging buffers: chunk i is staged into
// buffer i % 2 while the GPU command of chunk i - 1 runs, then that command
// is waited for and chunk i's submitted, so the reads and the GPU's writes
// overlap and one command is in flight at a time, as the backend requires.
// Buffer i % 2 is free when chunk i is staged: its last command, chunk
// i - 2's, ended before chunk i - 1's was submitted. stage(i, buffer)
// stages; submit(i, buffer) submits and returns the command's ticket, whose
// wait() ends it. A stage that throws leaves the command in flight to its
// ticket's destruction, which waits for it, so nothing writes the image
// after the writer failed. (Waiting for each command before staging the
// next kept the loading thread waiting on the GPU for 15% of a load; staged
// this way it waits 1-2%, and warm file-cached loads measure 1.29-1.54x
// faster upstream, 1.7.1.)
template <class Stage, class Submit>
void pipelineChunks(size_t count, Stage &&stage, Submit &&submit) {
  std::optional<decltype(submit(size_t{0}, 0U))> pending;
  for (size_t chunk = 0; chunk < count; ++chunk) {
    const unsigned buffer = chunk % 2;
    stage(chunk, buffer);
    if (pending) static_cast<void>(pending->wait());
    pending.emplace(submit(chunk, buffer));
  }
  if (pending) static_cast<void>(pending->wait());
}

// Writes every byte of a planned image into its buffer: the header and
// descriptors, the copied rows and the planes the GPU repacks into it.
void writeGgufImage(metal::MetalBackend &backend, const WeightSource &source, const metal::MetalBuffer &image,
                    const gguf::Image &plan);

} // namespace richengine::model
