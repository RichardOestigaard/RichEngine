// The decode-step pack chain against a CPU reference: decode_step_pack
// mirrors a committed step's acceptance results into the pack block the next
// step binds, draft_predraft_pack stages each lane's draft input rows from
// it, and verify_input_tokens / verify_input_tokens_dev emit the next
// step's verify input. The buffers' per-lane strides are deliberately
// asymmetric — the arena allocates draft_input a row wider than its
// queryRows stride — so an implementation that strides the staging or the
// read by the buffer's lane size instead of draft_stride lands anchors on
// the wrong lanes: row 0 of every lane's verify input must hold that lane's
// own anchor, never a neighbour's.
#include "TestChecks.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/Embedding.h"
#include "metal/abi/StepPack.h"
#include "ops/KernelNames.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using richengine::metal::BufferStorage;
using richengine::metal::ComputeDispatch;
using richengine::metal::MetalBackend;
using richengine::metal::MetalBuffer;

constexpr uint32_t kLanes = RICHENGINE_MAXIMUM_BATCH_WIDTH;
// Deliberately narrower than the compiled strides so every packed layout
// differs from the arena's lane capacity — the bug class this covers
// staged anchors at the allocation stride instead of the kernel's
// draft_stride, misplacing them on every lane past the first.
constexpr uint32_t kVerifyStride = RICHENGINE_TARGET_VERIFY_ROWS / 2;
constexpr uint32_t kDraftStride = RICHENGINE_DRAFT_QUERY_ROWS / 2;
constexpr uint32_t kProposals = RICHENGINE_DRAFT_PROPOSAL_TOKENS;
constexpr uint32_t kMaskToken = 246813;

MetalBuffer shared(MetalBackend &backend, uint64_t bytes, const char *label) {
  return backend.allocateBuffer(bytes, BufferStorage::Shared, label);
}

template <class T> T *contents(const MetalBuffer &buffer) {
  return static_cast<T *>(buffer.contents());
}

using richengine::test::require;

void dispatch(MetalBackend &backend, std::string_view name,
              std::initializer_list<richengine::metal::BufferBinding> buffers,
              const void *params, uint64_t paramsBytes,
              richengine::metal::DispatchSize threadgroups) {
  ComputeDispatch d;
  d.pipelineName = std::string(name);
  d.buffers = buffers;
  d.bytes = {{static_cast<uint32_t>(buffers.size()), params, paramsBytes}};
  d.threadgroups = threadgroups;
  d.threadsPerThreadgroup = {1, 1, 1};
  static_cast<void>(backend.submit(d));
}

// decode_step_pack → draft_predraft_pack → verify_input_tokens over an
// oversized draft-input allocation: a lane's anchor is its last retained
// output token and lands on its own verify row 0.
void testStagedAnchorChain(MetalBackend &backend) {
  // Per-lane arena capacity deliberately wider than draft_stride, matching
  // production's kDecodeRows-wide DraftInputTokens.
  constexpr uint32_t kDraftCapacity = kDraftStride + 1;
  MetalBuffer retained =
      shared(backend, kLanes * sizeof(uint32_t), "vi-retained");
  MetalBuffer accepted =
      shared(backend, kLanes * sizeof(uint32_t), "vi-accepted");
  MetalBuffer output = shared(
      backend, kLanes * kVerifyStride * sizeof(uint32_t), "vi-output");
  MetalBuffer pack =
      shared(backend, kLanes * sizeof(DecodeStepPackLane), "vi-pack");
  MetalBuffer draftInput =
      shared(backend, kLanes * kDraftCapacity * sizeof(uint32_t),
             "vi-draft-input");
  MetalBuffer positions =
      shared(backend, kLanes * kVerifyStride * sizeof(uint32_t),
             "vi-positions");
  MetalBuffer proposals =
      shared(backend, kLanes * kProposals * sizeof(uint32_t),
             "vi-proposals");
  MetalBuffer verifyInput = shared(
      backend, kLanes * kVerifyStride * sizeof(uint32_t), "vi-verify");

  auto *out = contents<uint32_t>(output);
  auto *ret = contents<uint32_t>(retained);
  auto *acc = contents<uint32_t>(accepted);
  const uint32_t anchors[kLanes] = {4100, 4200, 4300, 4400};
  const uint32_t retCount[kLanes] = {1, 3, 2, 4};
  for (uint32_t lane = 0; lane < kLanes; ++lane) {
    ret[lane] = retCount[lane];
    acc[lane] = retCount[lane] - 1;
    for (uint32_t row = 0; row < kVerifyStride; ++row)
      out[lane * kVerifyStride + row] =
          anchors[lane] - (retCount[lane] - 1 - row);
  }
  std::memset(pack.contents(), 0, pack.sizeBytes());
  std::memset(draftInput.contents(), 0xFF, draftInput.sizeBytes());
  std::memset(proposals.contents(), 0x55, proposals.sizeBytes());
  std::memset(verifyInput.contents(), 0xAA, verifyInput.sizeBytes());

  DecodeStepPackParams packParams{};
  packParams.lanes = kLanes;
  packParams.output_stride = kVerifyStride;
  packParams.decode_rows = kVerifyStride;
  packParams.verify_stride = kVerifyStride;
  for (uint32_t lane = 0; lane < kLanes; ++lane) {
    packParams.request_id[lane] = 9000 + lane;
    packParams.position_base[lane] = 100 + lane;
    packParams.remaining[lane] = 50;
    packParams.budget[lane] = kVerifyStride - 1;
  }
  dispatch(backend, richengine::ops::kDecodeStepPack,
           {{0, retained}, {1, accepted}, {2, output}, {3, pack}},
           &packParams, sizeof(packParams), {kLanes, 1, 1});

  DraftPredraftPackParams draftParams{};
  draftParams.lanes = kLanes;
  draftParams.query_rows = kDraftStride;
  draftParams.decode_rows = kVerifyStride;
  draftParams.mask_token = kMaskToken;
  dispatch(backend, richengine::ops::kDraftPredraftPack,
           {{0, pack}, {1, draftInput}, {2, positions}}, &draftParams,
           sizeof(draftParams),
           {std::max(kDraftStride, kVerifyStride), kLanes, 1});

  VerifyInputBatchParams verify{};
  verify.vocabulary = 262144;
  verify.verify_stride = kVerifyStride;
  verify.draft_stride = kDraftStride;
  for (uint32_t lane = 0; lane < kLanes; ++lane)
    verify.live_rows[lane] = kVerifyStride;
  dispatch(backend, richengine::ops::kVerifyInputTokens,
           {{0, draftInput}, {1, proposals}, {2, verifyInput}}, &verify,
           sizeof(verify), {kLanes * kVerifyStride, 1, 1});

  const auto *packLanes = contents<DecodeStepPackLane>(pack);
  const auto *vin = contents<uint32_t>(verifyInput);
  for (uint32_t lane = 0; lane < kLanes; ++lane) {
    require(packLanes[lane].anchor == anchors[lane],
            "step pack anchor mismatch");
    require(vin[lane * kVerifyStride] == anchors[lane],
            "verify input staged another lane's anchor");
    for (uint32_t row = 1; row < kVerifyStride; ++row)
      require(vin[lane * kVerifyStride + row] ==
                  std::min(contents<uint32_t>(proposals)
                               [lane * kProposals + row - 1],
                           verify.vocabulary - 1),
              "verify input proposal row mismatch");
  }
}

// verify_input_tokens_dev's anchor choice: a pack that does not prove to
// be the lane's own predecessor's falls back to the staged draft_input
// anchor; a matching id with matching position and live rows takes the
// pack's anchor.
void testDeviceAnchorFallback(MetalBackend &backend) {
  MetalBuffer draftInput =
      shared(backend, kLanes * kDraftStride * sizeof(uint32_t),
             "vid-draft-input");
  MetalBuffer proposals =
      shared(backend, kLanes * kProposals * sizeof(uint32_t),
             "vid-proposals");
  MetalBuffer verifyInput = shared(
      backend, kLanes * kVerifyStride * sizeof(uint32_t), "vid-verify");
  MetalBuffer pack =
      shared(backend, kLanes * sizeof(DecodeStepPackLane), "vid-pack");

  auto *staged = contents<uint32_t>(draftInput);
  auto *packLanes = contents<DecodeStepPackLane>(pack);
  for (uint32_t lane = 0; lane < kLanes; ++lane) {
    staged[lane * kDraftStride] = 6100 + lane;
    packLanes[lane] = {};
    packLanes[lane].request_id = lane ? 8000 + lane : 1;
    packLanes[lane].anchor = 7700 + lane;
    packLanes[lane].position_base = 200 + lane;
    packLanes[lane].live_rows = 5;
  }
  std::memset(proposals.contents(), 0x33, proposals.sizeBytes());
  std::memset(verifyInput.contents(), 0xAA, verifyInput.sizeBytes());

  VerifyInputDeviceParams params{};
  params.verify.vocabulary = 262144;
  params.verify.verify_stride = kVerifyStride;
  params.verify.draft_stride = kDraftStride;
  params.device_lanes = 0;
  for (uint32_t lane = 0; lane < kLanes; ++lane) {
    params.request_id[lane] = 8000 + lane;
    params.position_base[lane] = 200 + lane;
    params.verify.live_rows[lane] = 5;
  }
  dispatch(backend, richengine::ops::kVerifyInputTokensDev,
           {{0, draftInput}, {1, proposals}, {2, verifyInput}, {3, pack}},
           &params, sizeof(params), {kLanes * kVerifyStride, 1, 1});

  const auto *vin = contents<uint32_t>(verifyInput);
  // Lane 0's pack id mismatches → staged anchor; lanes 1-3 prove →
  // pack anchors.
  require(vin[0] == 6100, "unproven lane did not take the staged anchor");
  for (uint32_t lane = 1; lane < kLanes; ++lane)
    require(vin[lane * kVerifyStride] == 7700 + lane,
            "proven lane did not take its pack anchor");
  // A proven pack whose live rows disagree with the staged count also
  // falls back.
  packLanes[1].live_rows = 9;
  std::memset(verifyInput.contents(), 0xAA, verifyInput.sizeBytes());
  dispatch(backend, richengine::ops::kVerifyInputTokensDev,
           {{0, draftInput}, {1, proposals}, {2, verifyInput}, {3, pack}},
           &params, sizeof(params), {kLanes * kVerifyStride, 1, 1});
  require(vin[kVerifyStride] == 6101,
          "live-row mismatch did not fall back to the staged anchor");
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::invalid_argument("usage: verify-input METALLIB");
    MetalBackend backend(argv[1]);
    testStagedAnchorChain(backend);
    testDeviceAnchorFallback(backend);
    std::cout << "verify_input_metal_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "verify_input_metal_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
