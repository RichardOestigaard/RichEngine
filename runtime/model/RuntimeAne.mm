#include "model/RuntimeImpl.hpp"

namespace splash::model {

  // bf16 rows (the arena's hidden storage) to the fp16 the predictor
  // contracts name: a bf16 is the top half of a float's bits.
  void Runtime::Impl::bf16ToFp16Row(const uint16_t *source, _Float16 *target,
                            uint32_t count) {
    for (uint32_t index = 0; index < count; ++index) {
      const uint32_t bits = uint32_t{source[index]} << 16;
      float value;
      std::memcpy(&value, &bits, sizeof(value));
      target[index] = static_cast<_Float16>(value);
    }
  }

  // Feeds the predictors the state this batch just committed: the medusa
  // predictor's leaf alternates apply to the next tree batch, the predraft
  // predictor's proposal chain applies to whatever lane still holds the
  // anchor and position it predicted. Kicking here — before the results
  // reach the engine — gives a job the whole draft+verify of the next step.
  void Runtime::Impl::kickAnePredictors(std::span<const DecodeLaneResult> lanes,
                         std::span<const ModelBatchItem> items) {
    if (!aneMedusa_ && !anePredraft_)
      return;
    const uint32_t width = static_cast<uint32_t>(lanes.size());
    const uint32_t hiddenSize = geometry.target.hiddenSize;
    auto input = std::make_shared<AneStepInput>();
    // The artifacts take fixed shapes: all kLaneCount lanes every call, the
    // unused tail zero-padded.
    input->lanes = kLaneCount;
    input->hidden.resize(size_t{kLaneCount} * SPLASH_TARGET_VERIFY_ROWS *
                         hiddenSize);
    for (uint32_t lane = 0; lane < width; ++lane) {
      const DecodeLaneResult &result = lanes[lane];
      input->retained[lane] = static_cast<int32_t>(result.retained);
      const uint16_t *source = contents<uint16_t>(
          decodeArena->get(lane, DecodeTensor::CapturedTargetHidden),
          "captured verify hidden");
      _Float16 *rows = input->hidden.data() +
                       size_t{lane} * SPLASH_TARGET_VERIFY_ROWS * hiddenSize;
      for (uint32_t row = 0; row < result.retained; ++row)
        bf16ToFp16Row(source + size_t{row} * hiddenSize,
                      rows + size_t{row} * hiddenSize, hiddenSize);
      const uint32_t *emitted = contents<uint32_t>(
          decodeArena->get(lane, DecodeTensor::OutputTokens),
          "emitted tokens");
      input->anchors[lane] = emitted[result.retained - 1];
      input->positions[lane] = items[lane].logicalPosition + result.retained;
    }

    if (aneMedusa_) {
      AnePredictor *predictor = aneMedusa_.get();
      const uint32_t serial = ++medusaSerial_;
      aneMedusa_->submit([this, predictor, serial, input, hiddenSize] {
        std::vector<int32_t> tokens(
            size_t{input->lanes} * SPLASH_DRAFT_PROPOSAL_TOKENS, -1);
        AneTensor hiddenIn{"hidden", AneDType::Float16,
                           {input->lanes, SPLASH_TARGET_VERIFY_ROWS,
                            static_cast<int64_t>(hiddenSize)},
                           input->hidden.data()};
        AneTensor retainedIn{"retained", AneDType::Int32,
                             {input->lanes},
                             const_cast<int32_t *>(input->retained.data())};
        AneTensor out{"leaf_tokens", AneDType::Int32,
                      {input->lanes, SPLASH_DRAFT_PROPOSAL_TOKENS},
                      tokens.data()};
        const std::array<AneTensor, 2> inputs{hiddenIn, retainedIn};
        std::array<AneTensor, 1> outputs{out};
        std::string error;
        if (!predictor->predict(std::span<const AneTensor>(inputs),
                                std::span<AneTensor>(outputs), error))
          return;
        std::memcpy(contents<uint32_t>(aneLeafTokens_, "ane leaf tokens"),
                    tokens.data(), tokens.size() * sizeof(uint32_t));
        // The tokens must be visible before the serial they publish.
        std::atomic_thread_fence(std::memory_order_release);
        *static_cast<uint32_t *>(aneFlag_.contents()) = serial;
      });
    }

    if (anePredraft_) {
      AnePredictor *predictor = anePredraft_.get();
      {
        std::lock_guard<std::mutex> lock(aneMutex_);
        predraftValid_ = false;
        for (uint32_t lane = 0; lane < width; ++lane) {
          aneAnchors_[lane] = input->anchors[lane];
          anePositions_[lane] = input->positions[lane];
        }
        predraftLanes_ = width;
      }
      const uint32_t serial = ++predraftSerial_;
      anePredraft_->submit([this, predictor, serial, input, hiddenSize] {
        std::vector<int32_t> proposals(
            size_t{input->lanes} * SPLASH_DRAFT_PROPOSAL_TOKENS, -1);
        std::array<int32_t, kLaneCount> anchors32{};
        std::array<int32_t, kLaneCount> positions32{};
        for (uint32_t lane = 0; lane < input->lanes; ++lane) {
          anchors32[lane] = static_cast<int32_t>(input->anchors[lane]);
          positions32[lane] =
              static_cast<int32_t>(input->positions[lane]);
        }
        AneTensor anchorIn{"anchor", AneDType::Int32, {input->lanes},
                           anchors32.data()};
        AneTensor positionIn{"position", AneDType::Int32, {input->lanes},
                             positions32.data()};
        AneTensor hiddenIn{"hidden", AneDType::Float16,
                           {input->lanes, SPLASH_TARGET_VERIFY_ROWS,
                            static_cast<int64_t>(hiddenSize)},
                           input->hidden.data()};
        AneTensor retainedIn{"retained", AneDType::Int32, {input->lanes},
                             const_cast<int32_t *>(input->retained.data())};
        AneTensor out{"proposals", AneDType::Int32,
                      {input->lanes, SPLASH_DRAFT_PROPOSAL_TOKENS},
                      proposals.data()};
        const std::array<AneTensor, 4> inputs{anchorIn, positionIn, hiddenIn,
                                              retainedIn};
        std::array<AneTensor, 1> outputs{out};
        std::string error;
        const bool ok =
            predictor->predict(std::span<const AneTensor>(inputs),
                               std::span<AneTensor>(outputs), error);
        if (!ok && aneDebug_)
          fprintf(stderr, "ane-predraft predict failed: %s\n", error.c_str());
        {
          std::lock_guard<std::mutex> lock(aneMutex_);
          if (ok) {
            for (uint32_t lane = 0; lane < input->lanes; ++lane)
              std::copy_n(proposals.data() +
                              lane * SPLASH_DRAFT_PROPOSAL_TOKENS,
                          SPLASH_DRAFT_PROPOSAL_TOKENS,
                          aneProposals_[lane].data());
            predraftValid_ = true;
          }
          predraftDone_ = serial;
        }
        aneCv_.notify_all();
      });
    }
  }

  // A completed predraft whose assumed anchors and positions match this step
  // replaces the draft forward: its proposals land in the lanes'
  // ProposedTokens and the chain verify consumes them unchanged. Anything
  // else — a running job, a stale or different-laned result, a sampled lane —
  // keeps the GPU draft.
  bool Runtime::Impl::applyAnePredraft(std::span<Request *const> entries,
                        std::span<const ModelBatchItem> items,
                        uint32_t width) {
    if (!anePredraft_ || !predraftSerial_)
      return false;
    {
      std::unique_lock<std::mutex> lock(aneMutex_);
      aneCv_.wait_for(lock, std::chrono::milliseconds(aneWaitMs_), [&] {
        return predraftDone_.load() == predraftSerial_.load();
      });
    }
    if (predraftDone_ != predraftSerial_ || !predraftValid_ ||
        predraftLanes_ != width) {
      if (aneDebug_)
        fprintf(stderr,
                "ane-predraft miss: done=%u serial=%u valid=%d lanes=%u\n",
                predraftDone_.load(), predraftSerial_.load(),
                predraftValid_.load() ? 1 : 0, predraftLanes_.load());
      return false;
    }
    for (uint32_t lane = 0; lane < width; ++lane) {
      const Request &entry = *entries[lane];
      if (samplingEnabled(entry) || !entry.pendingToken ||
          *entry.pendingToken != aneAnchors_[lane] ||
          items[lane].logicalPosition != anePositions_[lane]) {
        if (aneDebug_)
          fprintf(stderr,
                  "ane-predraft lane %u mismatch: anchor=%u want=%u "
                  "pos=%llu want=%llu\n",
                  lane, entry.pendingToken ? *entry.pendingToken : 0,
                  aneAnchors_[lane],
                  (unsigned long long)items[lane].logicalPosition,
                  (unsigned long long)anePositions_[lane]);
        return false;
      }
    }
    for (uint32_t lane = 0; lane < width; ++lane)
      std::memcpy(contents<uint32_t>(
                      decodeArena->get(lane, DecodeTensor::ProposedTokens),
                      "ane proposals"),
                  aneProposals_[lane].data(),
                  SPLASH_DRAFT_PROPOSAL_TOKENS * sizeof(uint32_t));
    predraftValid_ = false;
    return true;
  }

} // namespace splash::model
