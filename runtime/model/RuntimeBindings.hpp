#pragma once

// Runtime::Impl member bodies extracted from RuntimeImpl.hpp: request
// admission (activate and its image-row registry) and the decode-arena
// lane bindings. RuntimeImpl.hpp includes this file after the complete
// Impl definition.

inline   StateAdmission Runtime::Impl::activate(const ModelRequest &request, uint32_t stateLane,
                          std::vector<ImageState> &images) {
    if (request.images.empty())
      return laneAdmission(stateLane, states.tryActivateLane(stateLane, request.id));
    // The engine rejects image requests at submission when there is no vision.
    if (!package.descriptor.hasVision())
      throw std::logic_error("image request reached a model without vision");
    std::vector<ImageState> staged;
    staged.reserve(request.images.size());
    std::vector<std::shared_ptr<ImageRows>> shared;
    uint64_t bytes = 0;
    // The patches of the largest staged image that still needs its encode.
    uint32_t encodePatches = 0;
    for (const ImageSpan &span : request.images) {
      if (span.end() <= request.restoredTokens) {
        staged.push_back({span, nullptr});
        continue;
      }
      // New rows enter the registry now, so a repeated placement shares
      // them; they have no buffers until the admission allocates them.
      std::shared_ptr<ImageRows> rows = findRows(span);
      if (!rows) {
        rows = std::make_shared<ImageRows>();
        rows->key = imageKey(span);
        imageRows.insert_or_assign(rows->key, rows);
        bytes += span.pixelBytes() + embeddingBytes(span);
      } else if (rows->embeddings && std::ranges::find(shared, rows) == shared.end()) {
        shared.push_back(rows);
      }
      if (!rows->encoded)
        encodePatches = std::max(encodePatches, span.gridHeight * span.gridWidth);
      staged.push_back({span, std::move(rows)});
    }
    const uint64_t encoderBytes =
        encodePatches && !(vision && vision->maximumPatches() >= encodePatches)
            ? ops::Vision::scratchBytes(package.vision.tensors.layout, encodePatches)
            : 0;
    std::shared_ptr<ops::Vision> encoder;
    const uint8_t *pixels = request.imagePixels.data();
    const auto allocate = [&] {
      if (encoderBytes) {
        encoder = std::make_shared<ops::Vision>(
            backend, package.vision.tensors, encodePatches);
      }
      for (ImageState &image : staged) {
        const ImageSpan &span = image.span;
        if (image.rows && !image.rows->embeddings) {
          ImageRows &rows = *image.rows;
          rows.pixels = backend.allocateBuffer(
              span.pixelBytes(), BufferStorage::Shared, "image pixels");
          std::memcpy(contents<uint8_t>(rows.pixels, "image pixels"), pixels,
                      static_cast<size_t>(span.pixelBytes()));
          rows.embeddings = backend.allocateBuffer(
              embeddingBytes(span), BufferStorage::Private, "image embeddings");
        }
        pixels += span.pixelBytes();
      }
    };
    StateAdmission admission = laneAdmission(
        stateLane, states.tryActivateLane(stateLane, request.id, encoderBytes + bytes, allocate));
    if (!admission.granted()) {
      admission.held = std::make_shared<const Matched>(
          Matched{std::move(shared), encodePatches && !encoderBytes ? vision : nullptr});
      return admission;
    }
    if (encoder)
      vision = std::move(encoder);
    counters.imageEmbeddingReuses += shared.size();
    images = std::move(staged);
    return admission;
  }

inline   bool Runtime::Impl::visionIdle() noexcept {
    for (auto entry = imageRows.begin(); entry != imageRows.end();) {
      const std::shared_ptr<ImageRows> rows = entry->second.lock();
      if (!rows) {
        entry = imageRows.erase(entry);
        continue;
      }
      if (!rows->encoded)
        return false;
      ++entry;
    }
    return true;
  }

inline   uint64_t Runtime::Impl::embeddingBytes(const ImageSpan &span) const {
    return uint64_t{ops::Vision::embeddingRows(span.grid())} *
           geometry.target.hiddenSize * sizeof(uint16_t);
  }

inline Runtime::Impl::ImageKey Runtime::Impl::imageKey(const ImageSpan &span) noexcept {
    return {span.digestLo, span.digestHi, span.gridHeight, span.gridWidth};
  }

inline std::shared_ptr<Runtime::Impl::ImageRows> Runtime::Impl::findRows(const ImageSpan &span) {
    const auto found = imageRows.find(imageKey(span));
    if (found == imageRows.end())
      return {};
    std::shared_ptr<ImageRows> rows = found->second.lock();
    if (!rows) {
      imageRows.erase(found);
      return {};
    }
    if (rows->cached)
      embeddingCache.splice(embeddingCache.begin(), embeddingCache, *rows->cached);
    return rows;
  }

inline   void Runtime::Impl::retain(const std::shared_ptr<ImageRows> &rows) {
    if (rows->cached) {
      embeddingCache.splice(embeddingCache.begin(), embeddingCache, *rows->cached);
      return;
    }
    const uint64_t bytes = rows->embeddings.sizeBytes();
    embeddingCache.push_front(rows);
    rows->cached = embeddingCache.begin();
    embeddingCacheBytes += bytes;
    while (!embeddingCache.empty() &&
           embeddingCacheBytes + heldRowsBytes(true) > kEmbeddingCacheBytes)
      static_cast<void>(uncache(std::prev(embeddingCache.end())));
  }

inline   uint64_t Runtime::Impl::heldRowsBytes(bool uncachedOnly) const noexcept {
    uint64_t bytes = 0;
    for (auto hold = stateHolds.begin(); hold != stateHolds.end(); ++hold) {
      const std::shared_ptr<const HeldState> held = hold->lock();
      if (!held || (uncachedOnly && held->rows->cached))
        continue;
      const bool counted = std::any_of(
          stateHolds.begin(), hold, [&](const std::weak_ptr<const HeldState> &earlier) {
            const std::shared_ptr<const HeldState> other = earlier.lock();
            return other && other->rows == held->rows;
          });
      if (!counted)
        bytes += held->rows->embeddings.sizeBytes();
    }
    return bytes;
  }

inline std::shared_ptr<const CompositeState>
Runtime::Impl::holdStraddledRows(const Request &entry, std::shared_ptr<const CompositeState> state) {
    std::erase_if(stateHolds, [](const std::weak_ptr<const HeldState> &hold) {
      return hold.expired();
    });
    const uint64_t boundary = states.metadata(entry.stateLane).lengths.targetTokens;
    for (const ImageState &image : entry.images) {
      // The chunk that ended at the boundary encoded the image it reached.
      if (image.span.offset >= boundary || image.span.end() <= boundary)
        continue;
      if (image.span.end() - boundary >= kv::kPageTokens)
        break;
      const bool kept =
          image.rows->cached ||
          std::ranges::any_of(stateHolds, [&](const std::weak_ptr<const HeldState> &hold) {
            const std::shared_ptr<const HeldState> held = hold.lock();
            return held && held->rows == image.rows;
          });
      const uint64_t added = kept ? 0 : image.rows->embeddings.sizeBytes();
      if (embeddingCacheBytes + heldRowsBytes(true) + added > kEmbeddingCacheBytes)
        break;
      auto held = std::make_shared<const HeldState>(HeldState{std::move(state), image.rows});
      stateHolds.push_back(held);
      return {held, held->state.get()};
    }
    return state;
  }

inline   uint64_t Runtime::Impl::uncache(std::list<std::shared_ptr<ImageRows>>::iterator entry) noexcept {
    const uint64_t bytes = (*entry)->embeddings.sizeBytes();
    (*entry)->cached.reset();
    embeddingCache.erase(entry);
    embeddingCacheBytes -= bytes;
    return bytes;
  }

inline   void Runtime::Impl::releaseImages(Request &entry) {
    for (const ImageState &image : entry.images) {
      if (image.rows && image.rows->encoded)
        retain(image.rows);
    }
    entry.images.clear();
  }

inline   uint64_t Runtime::Impl::releaseOneCache() noexcept {
    if (vision && vision.use_count() == 1 && visionIdle()) {
      const uint64_t bytes = vision->arenaBytes();
      vision.reset();
      return bytes;
    }
    for (auto entry = embeddingCache.end(); entry != embeddingCache.begin();) {
      if ((--entry)->use_count() == 1)
        return uncache(entry);
    }
    return 0;
  }

inline   MetalBuffer Runtime::Impl::synchronizedPageTable(Request &entry,
                                                  const ModelBatchItem &item) {
    if (entry.stateLane >= pageTableBindings.size())
      throw std::out_of_range("request state lane is outside page tables");
    if (item.pageTable.empty() ||
        item.pageTable.size() > kMaximumPageTableEntries) {
      throw std::invalid_argument("request page table has invalid length");
    }
    if (!item.pageTableRevision)
      throw std::invalid_argument("request page table has no revision");
    PageTableBinding &binding = pageTableBindings[entry.stateLane];
    MetalBuffer destination =
        decodeArena->get(entry.stateLane, DecodeTensor::PageTable);
    // Rewrite only what changed since the table was written: nothing at the
    // same revision, the entries from the first changed page on at the next
    // one, and everything after two changes or for another request.
    const auto size = static_cast<uint32_t>(item.pageTable.size());
    uint32_t first = 0;
    if (binding.requestId == entry.id) {
      if (binding.revision == item.pageTableRevision)
        first = size;
      else if (binding.revision + 1 == item.pageTableRevision)
        first = std::min(item.pageTableFirstChanged, size);
    }
    if (first < size)
      kvPages.writeEntries(item.pageTable, first, destination);
    binding = {entry.id, item.pageTableRevision};
    return destination;
  }

inline   ops::SamplingBuffers Runtime::Impl::samplingBuffers(uint32_t lanes) const {
    auto d = [&](DecodeTensor tensor) {
      return decodeArena->packed(tensor, lanes);
    };
    return {d(DecodeTensor::Logits),
            d(DecodeTensor::TargetPartialMasses),
            d(DecodeTensor::TargetVocabularyRows),
            d(DecodeTensor::SamplingUniforms),
            d(DecodeTensor::ConstraintMasks),
            d(DecodeTensor::OutputTokens),
            d(DecodeTensor::ArgmaxValues),
            d(DecodeTensor::ArgmaxIndices),
            d(DecodeTensor::InputTokens),
            d(DecodeTensor::Candidates),
            d(DecodeTensor::ProposalProbs),
            d(DecodeTensor::TargetVocabularyRanges),
            d(DecodeTensor::TargetVocabularyArrivals),
            d(DecodeTensor::HeadArgmaxValues),
            d(DecodeTensor::HeadArgmaxIndices)};
  }

inline   std::span<uint32_t> Runtime::Impl::penaltyWords(uint32_t stateLane) const {
    return {contents<uint32_t>(
                decodeArena->get(stateLane, DecodeTensor::PenaltyState),
                "penalty words"),
            geometry.target.vocabularySize};
  }

inline   void Runtime::Impl::bindPenalties(const Request &entry,
                     std::span<const uint32_t> history) const {
    const ops::SamplingPenalties penalties = samplingPenalties(entry);
    if (!penalties.active())
      return;
    ops::Sampling::rebuildPenaltyWords(penaltyWords(entry.stateLane), history,
                                       entry.generatedTokens,
                                       entry.pendingToken,
                                       penalties.repetition != 1.0F);
  }

inline Runtime::Impl::Request &Runtime::Impl::laneEntry(std::span<Request *const> entries, uint32_t lane) {
    Request *entry = entries[std::min<size_t>(lane, entries.size() - 1)];
    if (!entry)
      throw std::invalid_argument("empty decode batch lane");
    return *entry;
  }

inline   void Runtime::Impl::bindPageTables(
      std::span<Request *const> entries,
      std::array<MetalBuffer, kLaneCount> &pageTables) const {
    for (uint32_t lane = 0; lane < kLaneCount; ++lane)
      pageTables[lane] =
          decodeArena->get(laneEntry(entries, lane).stateLane,
                           DecodeTensor::PageTable);
  }

inline   void Runtime::Impl::bindGdnStates(
      std::span<Request *const> entries,
      std::array<MetalBuffer, kLaneCount> &current,
      std::array<MetalBuffer, kLaneCount> &next) const {
    for (uint32_t lane = 0; lane < kLaneCount; ++lane) {
      Request &entry = laneEntry(entries, lane);
      current[lane] = states.current(entry.stateLane).stateBase;
      next[lane] = states.next(entry.stateLane).stateBase;
    }
  }
