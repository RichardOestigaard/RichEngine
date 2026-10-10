#include "PagedAttention.hpp"

#include "ops/KernelNames.hpp"
#include "ops/Linear.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>
#include <set>
#include <stdexcept>
#include <vector>

namespace richengine::ops {
namespace {

[[nodiscard]] constexpr uint32_t elementwiseGroups(uint64_t elements) {
  return (elements + metal::CommandGraph::kDefaultThreads - 1) /
         metal::CommandGraph::kDefaultThreads;
}

// Each query tile's history splits: the maximum shared out over the chunk's
// tiles, at least one. RICHENGINE_PREFILL_SPLITS_MAX lowers the maximum —
// the workspace sizes to the same count, and the shader contract's baked
// bound still admits it.
[[nodiscard]] inline uint32_t prefillSplits(uint32_t tiles) {
  const uint32_t maximum = tuning().prefillSplitsMax;
  return std::clamp(maximum / tiles, 1u, maximum);
}

// Partials and statistics of `rows` fused rows of headDimension elements.
[[nodiscard]] constexpr AttentionWorkspace attentionWorkspace(uint64_t rows,
                                                              uint32_t headDimension) {
  return {rows * headDimension * sizeof(float), rows * 2 * sizeof(float)};
}

// The kernel family's attention geometry. HeadDim 256 keeps the original
// three layouts; headDim 128 (the dense target) and 64 (LFM2) run their own
// KV/group combinations.
enum class KernelLayout : uint8_t {
  Kv4Group6,
  Kv4Group4,
  Kv2Group8,
  Kv2Group8D128,
  Kv8Group4D64,
  // Granite's KV8 layouts: 3B's 40 query heads of 64 (group 5) and 8B's
  // 32 of 128 (group 4).
  Kv8Group5D64,
  Kv8Group4D128,
  // Gemma 4's layouts: the sliding layers' KV8 group-2 pages of 256 and the
  // k_eq_v global layers' KV2 group-8 pages of 512.
  GemmaKv8H256,
  GemmaKv2H512
};

[[nodiscard]] constexpr uint32_t kernelLayoutHeadDim(KernelLayout layout) noexcept {
  switch (layout) {
  case KernelLayout::Kv4Group6:
  case KernelLayout::Kv4Group4:
  case KernelLayout::Kv2Group8:
  case KernelLayout::GemmaKv8H256:
    return 256;
  case KernelLayout::Kv2Group8D128:
    return 128;
  case KernelLayout::Kv8Group4D64:
  case KernelLayout::Kv8Group5D64:
    return 64;
  case KernelLayout::Kv8Group4D128:
    return 128;
  case KernelLayout::GemmaKv2H512:
    return 512;
  }
  return 0;
}

// The kernel-name suffix of a layout: the original kernels carry the KV/Group
// tag, the head-dimension variants carry the dimension tag.
[[nodiscard]] constexpr std::string_view kernelSuffix(KernelLayout layout) noexcept {
  switch (layout) {
  case KernelLayout::Kv4Group6:
    return "";
  case KernelLayout::Kv4Group4:
    return "_kv4_g4";
  case KernelLayout::Kv2Group8:
    return "_kv2_g8";
  case KernelLayout::Kv2Group8D128:
    return "_hd128";
  case KernelLayout::Kv8Group4D64:
    return "_hd64";
  case KernelLayout::Kv8Group5D64:
    return "_k8q5d64";
  case KernelLayout::Kv8Group4D128:
    return "_k8q4d128";
  case KernelLayout::GemmaKv8H256:
    return "_gemma_h256";
  case KernelLayout::GemmaKv2H512:
    return "_gemma_hd512";
  }
  return {};
}

[[nodiscard]] KernelLayout storageKernelLayout(const kv::Layout &layout) {
  if (layout.headDimension == 256) {
    switch (layout.kvHeads) {
    case 4:
      return KernelLayout::Kv4Group6;
    case 2:
      return KernelLayout::Kv2Group8;
    case 8:
      return KernelLayout::GemmaKv8H256;
    }
  } else if (layout.headDimension == 512 && layout.kvHeads == 2)
    return KernelLayout::GemmaKv2H512;
  else if (layout.headDimension == 128 && layout.kvHeads == 2)
    return KernelLayout::Kv2Group8D128;
  else if (layout.headDimension == 64 && layout.kvHeads == 8)
    return KernelLayout::Kv8Group4D64;
  else if (layout.headDimension == 128 && layout.kvHeads == 8)
    return KernelLayout::Kv8Group4D128;
  throw std::invalid_argument("unsupported attention layout");
}

// The stores' suffix: KV4/256 stores have no query-group variants, so the
// tag derives from the page geometry alone.
[[nodiscard]] std::string_view storageSuffix(const kv::Layout &layout) {
  if (layout.headDimension == 256) {
    if (layout.kvHeads == 4) return "";
    if (layout.kvHeads == 2) return "_kv2_g8";
    if (layout.kvHeads == 8) return "_gemma_h256";
    throw std::invalid_argument("unsupported attention layout");
  }
  if (layout.headDimension == 512 && layout.kvHeads == 2) return "_gemma_hd512";
  if (layout.headDimension == 128 && layout.kvHeads == 2) return "_hd128";
  if (layout.headDimension == 64 && layout.kvHeads == 8) return "_hd64";
  if (layout.headDimension == 128 && layout.kvHeads == 8) return "_k8d128";
  throw std::invalid_argument("unsupported attention layout");
}

[[nodiscard]] KernelLayout attentionKernelLayout(const kv::Layout &layout,
                                                 uint32_t queryHeads) {
  if (layout.headDimension == 256 && layout.kvHeads == 4)
    switch (queryHeads) {
    case 24:
      return KernelLayout::Kv4Group6;
    case 16:
      return KernelLayout::Kv4Group4;
    }
  // Gemma 4's sliding layers: KV8 of 256 with 16 query heads (group 2); its
  // k_eq_v globals: KV2 of 512 with 16 (group 8).
  if (layout.headDimension == 256 && layout.kvHeads == 8 &&
      queryHeads == 16)
    return KernelLayout::GemmaKv8H256;
  if (layout.headDimension == 512 && layout.kvHeads == 2 &&
      queryHeads == 16)
    return KernelLayout::GemmaKv2H512;
  const KernelLayout kernel = storageKernelLayout(layout);
  switch (kernel) {
  case KernelLayout::Kv2Group8:
    if (queryHeads == 16) return kernel;
    break;
  case KernelLayout::Kv2Group8D128:
    if (queryHeads == 16) return kernel;
    break;
  case KernelLayout::Kv8Group4D64:
    if (queryHeads == 32) return kernel;
    if (queryHeads == 40) return KernelLayout::Kv8Group5D64;
    break;
  case KernelLayout::Kv8Group4D128:
    if (queryHeads == 32) return kernel;
    break;
  default:
    break;
  }
  throw std::invalid_argument("unsupported attention layout");
}

[[nodiscard]] const char *formatTag(kv::Format format) noexcept {
  switch (format) {
  case kv::Format::Int8:
    return "q8";
  case kv::Format::Int4:
    return "int4";
  case kv::Format::BFloat16:
    return "bf16";
  case kv::Format::Float8E4M3:
    return "fp8";
  }
  return "";
}

// The kernel of `stem` for `format` at `layout`, with the layout suffix
// before any trailing tag ("split"/"store"/"_f32" style).
[[nodiscard]] std::string splitKernel(std::string_view stem, kv::Format format,
                                      KernelLayout layout) {
  return std::string(stem) + "_" + formatTag(format) + "_split" +
         std::string(kernelSuffix(layout));
}

// The Gemma splits name their window differently from the plain layouts:
// prefill h256 exists only windowed (_split_swa_h256, window 0 = full
// causal), hd512 windowless is _split_hd512; verify's windowless Gemma
// splits carry the _gemma tag.
[[nodiscard]] std::string gemmaSplitKernel(std::string_view stem,
                                           kv::Format format,
                                           KernelLayout layout,
                                           uint32_t windowTokens) {
  const bool prefill = stem.starts_with("prefill");
  std::string kernel = std::string(stem) + "_" + formatTag(format) + "_split";
  if (layout == KernelLayout::GemmaKv8H256) {
    kernel += (windowTokens || prefill) ? "_swa_h256" : "_gemma_h256";
  } else {
    kernel += windowTokens ? "_swa_hd512_m2"
                           : (prefill ? "_hd512_m2" : "_gemma_hd512_m2");
  }
  return kernel;
}

// The split a plan dispatches: the Gemma layouts route through
// gemmaSplitKernel, every other layout through splitKernel. A tree lane's
// scratch doubles the fused rows, so the Group-8 layouts' 128-row tile
// overflows threadgroup memory and they take the two-pass _m2 variants.
[[nodiscard]] std::string splitPipeline(std::string_view stem,
                                        const kv::Layout &layout,
                                        KernelLayout kernel) {
  if (kernel == KernelLayout::GemmaKv8H256 ||
      kernel == KernelLayout::GemmaKv2H512)
    return gemmaSplitKernel(stem, layout.format, kernel, layout.windowTokens);
  std::string name = splitKernel(stem, layout.format, kernel);
  if (stem == kVerifyTreeAttention &&
      (kernel == KernelLayout::Kv2Group8 ||
       kernel == KernelLayout::Kv2Group8D128))
    name += "_m2";
  return name;
}

[[nodiscard]] std::string storeKernel(std::string_view stem, kv::Format format,
                                      const kv::Layout &layout) {
  return std::string(stem) + "_" + formatTag(format) + "_store" +
         std::string(storageSuffix(layout));
}

[[nodiscard]] std::string reduceKernel(std::string_view stem, KernelLayout layout) {
  return std::string(stem) + std::string(kernelSuffix(layout));
}

[[nodiscard]] std::string gateKernel(std::string_view stem, bool queryGate,
                                     KernelLayout layout,
                                     std::string_view tableTag = {}) {
  std::string kernel(stem);
  kernel += queryGate ? "_gate" : "_gather";
  kernel += tableTag;
  kernel += kernelSuffix(layout);
  return kernel;
}

// The variant of `kernel` whose q/k norms are F32 (_f32), bf16 otherwise.
[[nodiscard]] std::string qkNormKernel(std::string kernel, const NormWeights &queryNorm,
                                       const NormWeights &keyNorm) {
  if (queryNorm.float32 != keyNorm.float32)
    throw std::invalid_argument("query and key norms differ in type");
  if (queryNorm.float32) kernel += "_f32";
  return kernel;
}

// The canvas splits name their variant like the Gemma prefill splits with
// "canvas_" inserted before the window tag: _split_canvas_swa_h256,
// _split_canvas_hd512 and so on. Every canvas split takes the trailing
// window_tokens constant; the windowless ones ignore it.
[[nodiscard]] std::string canvasSplitKernel(kv::Format format,
                                            KernelLayout layout,
                                            uint32_t windowTokens) {
  std::string kernel =
      std::string(kPrefillAttention) + "_" + formatTag(format) + "_split_canvas";
  if (windowTokens) kernel += "_swa";
  if (layout == KernelLayout::GemmaKv8H256)
    kernel += "_h256";
  else if (layout == KernelLayout::GemmaKv2H512)
    kernel += "_hd512_m2";
  else
    throw std::invalid_argument("no canvas attention kernel for layout");
  return kernel;
}

[[nodiscard]] std::string canvasReduceKernel(KernelLayout layout) {
  if (layout == KernelLayout::GemmaKv8H256)
    return std::string(kPrefillAttentionReduceCanvasGemmaH256);
  if (layout == KernelLayout::GemmaKv2H512)
    return std::string(kPrefillAttentionReduceCanvasGemmaHd512);
  throw std::invalid_argument("no canvas attention reducer for layout");
}

} // namespace

AttentionWorkspace PagedAttention::prefillWorkspace(uint32_t maximumRows, uint32_t queryHeads,
                                                      kv::Layout layout) {
  const KernelLayout kernelLayout = attentionKernelLayout(layout, queryHeads);
  if (!maximumRows || maximumRows > kv::kChunkedPrefillMaximumRows)
    throw std::invalid_argument("invalid attention workspace rows");
  // Allocation-time bound for every shorter sequence.
  uint64_t slots = 0;
  for (uint32_t tiles = 1; tiles <= kv::prefillAttentionTiles(maximumRows); ++tiles)
    slots = std::max(slots, uint64_t{tiles} * prefillSplits(tiles));
  return attentionWorkspace(
      slots * kv::kPrefillAttentionTileRows * queryHeads,
      kernelLayoutHeadDim(kernelLayout));
}

AttentionWorkspace PagedAttention::verifyWorkspace(uint32_t lanes, uint32_t queryHeads,
                                                     kv::Layout layout) {
  const KernelLayout kernelLayout = attentionKernelLayout(layout, queryHeads);
  if (!lanes || lanes > RICHENGINE_MAXIMUM_BATCH_WIDTH)
    throw std::invalid_argument("invalid attention workspace batch width");
  // Sized for a tree lane's node capacity so one workspace covers both
  // verify modes.
  return attentionWorkspace(uint64_t{lanes} * RICHENGINE_TREE_VERIFY_NODES *
                                kv::kVerifyMaximumSplits * queryHeads,
                            kernelLayoutHeadDim(kernelLayout));
}

PrefillAttentionPlan PagedAttention::prefillPlan(uint32_t rows, uint32_t queryHeads,
                                                   kv::Layout layout) {
  const KernelLayout kernelLayout = attentionKernelLayout(layout, queryHeads);
  if (!rows || rows > kv::kChunkedPrefillMaximumRows ||
      !kv::validFormat(layout.format))
    throw std::invalid_argument("invalid paged prefill attention geometry");
  const uint32_t tiles = kv::prefillAttentionTiles(rows);
  const uint32_t splits = prefillSplits(tiles);
  const uint32_t fusedRows =
      kv::kPrefillAttentionTileRows * (queryHeads / layout.kvHeads);
  return {rows,
          splits,
          attentionWorkspace(uint64_t{tiles} * splits * layout.kvHeads * fusedRows,
                             kernelLayoutHeadDim(kernelLayout)),
          splitPipeline(kPrefillAttention, layout, kernelLayout),
          reduceKernel(kPrefillAttentionReduce, kernelLayout),
          {layout.kvHeads, tiles, splits},
          {layout.kvHeads, fusedRows, tiles},
          metal::DispatchSize{layout.headDimension, 1, 1},
          layout.format, layout.scoreScale, layout.windowTokens};
}

PrefillAttentionPlan
PagedAttention::prefillCanvasPlan(uint32_t rows, uint32_t queryHeads,
                                  kv::Layout layout) {
  const KernelLayout kernelLayout =
      attentionKernelLayout(layout, queryHeads);
  if (kernelLayout != KernelLayout::GemmaKv8H256 &&
      kernelLayout != KernelLayout::GemmaKv2H512)
    throw std::invalid_argument("canvas attention requires a Gemma layout");
  const PrefillAttentionPlan plan = prefillPlan(rows, queryHeads, layout);
  // The canvas split covers every canvas page for each tile, so its reducer
  // must use the same full-canvas page partition.
  return {plan.rows,
          plan.splits,
          plan.workspace,
          canvasSplitKernel(layout.format, kernelLayout, layout.windowTokens),
          canvasReduceKernel(kernelLayout),
          plan.splitGroups,
          plan.reduceGroups,
          plan.reduceThreads,
          plan.format,
          plan.scoreScale,
          layout.windowTokens};
}

VerifyAttentionPlan PagedAttention::verifyPlan(
    uint32_t lanes, uint32_t queryHeads, kv::Layout layout,
    std::span<const uint32_t> historyTokens, bool tree, uint32_t liveNodes) {
  const KernelLayout kernelLayout = attentionKernelLayout(layout, queryHeads);
  if (!kv::validFormat(layout.format))
    throw std::invalid_argument("paged attention requires a kv format");
  if (tree && layout.format == kv::Format::Float8E4M3)
    throw std::invalid_argument("tree verify has no fp8 kernels");
  if (!lanes || lanes > RICHENGINE_MAXIMUM_BATCH_WIDTH ||
      historyTokens.size() != lanes)
    throw std::invalid_argument("verify lane count is out of range");
  const uint32_t rows =
      tree ? RICHENGINE_TREE_VERIFY_NODES : kv::kVerifyRows;
  // A host-emitted comb knows its batch-maximum node count at encode time;
  // a kernel-emitted table's counts are on the GPU, so it keeps the full
  // scratch. A leafless comb then stores and attends a chain's eight rows.
  const uint32_t liveRows =
      tree ? (liveNodes && liveNodes < RICHENGINE_TREE_VERIFY_NODES
                  ? liveNodes
                  : RICHENGINE_TREE_VERIFY_NODES)
           : kv::kVerifyRows;
  std::array<uint32_t, RICHENGINE_MAXIMUM_BATCH_WIDTH> laneSplits{};
  uint32_t splits = 0;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    if (uint64_t{historyTokens[lane]} + liveRows > kv::kMaximumPhysicalTokens)
      throw std::invalid_argument(
          "verify attention history exceeds physical context");
    laneSplits[lane] = kv::verifyAttentionSplits(historyTokens[lane], liveRows);
    splits = std::max(splits, laneSplits[lane]);
  }
  // The plan's exact scratch uses its own row capacity; the public
  // verifyWorkspace() allocation bound stays sized for a tree lane.
  const AttentionWorkspace workspace =
      attentionWorkspace(uint64_t{lanes} * rows * kv::kVerifyMaximumSplits *
                             queryHeads,
                         kernelLayoutHeadDim(kernelLayout));
  return {lanes,
          laneSplits,
          splits,
          workspace,
          splitPipeline(tree ? kVerifyTreeAttention : kVerifyAttention,
                        layout, kernelLayout),
          reduceKernel(tree ? kVerifyTreeAttentionReduce
                            : kVerifyAttentionReduce,
                       kernelLayout),
          {layout.kvHeads, splits, lanes},
          {layout.kvHeads, rows * (queryHeads / layout.kvHeads), lanes},
          // The verify stores are row-count generic; only the emitted rows
          // dispatch, not the tree tile's spare capacity row.
          storeKernel(kVerifyAttention, layout.format, layout),
          {uint64_t{lanes} * 2 * liveRows * layout.kvHeads, 1, 1},
          metal::DispatchSize{layout.headDimension, 1, 1},
          metal::DispatchSize{layout.headDimension, 1, 1},
          layout.format,
          rows, layout.scoreScale, layout.windowTokens};
}

void PagedAttention::addPrefillProjection(
    metal::CommandGraph &graph, metal::MetalBuffer packed,
    const NormWeights &queryNorm, const NormWeights &keyNorm,
    metal::MetalBuffer ropeCos, metal::MetalBuffer ropeSin,
    metal::MetalBuffer queries, metal::MetalBuffer chunkKeys,
    metal::MetalBuffer chunkValues, uint32_t tokens, uint32_t stride,
    uint32_t queryHeads, kv::Layout layout) {
  const KernelLayout kernelLayout = attentionKernelLayout(layout, queryHeads);
  const bool hasNorms = queryNorm.buffer && keyNorm.buffer;
  std::string kernel = std::string(kPrefillAttentionQkv) +
                       std::string(kernelSuffix(kernelLayout));
  if (hasNorms) kernel = qkNormKernel(kernel, queryNorm, keyNorm);
  FullPrefillParams params{.tokens = tokens, .stride = stride};
  // Unused norm slots keep the binding contract satisfied.
  graph.add(kernel,
            {packed, hasNorms ? queryNorm.buffer : queries,
             hasNorms ? keyNorm.buffer : queries, ropeCos, ropeSin, queries,
             chunkKeys, chunkValues},
            params,
            {tokens * (queryHeads + layout.kvHeads), 1, 1},
            {layout.headDimension, 1, 1});
}

void PagedAttention::addPrefillGate(
    metal::CommandGraph &graph, metal::MetalBuffer packed,
    metal::MetalBuffer attention, metal::MetalBuffer hidden, uint32_t tokens,
    uint32_t stride, uint32_t queryHeads, kv::Layout layout, bool queryGate) {
  const KernelLayout kernelLayout = attentionKernelLayout(layout, queryHeads);
  FullPrefillParams params{.tokens = tokens, .stride = stride};
  graph.add(gateKernel(kPrefillAttention, queryGate, kernelLayout),
            {packed, attention, hidden}, params,
            {elementwiseGroups(uint64_t{tokens} * queryHeads *
                               layout.headDimension),
             1, 1});
}

void PagedAttention::addPrefillGateSums(
    metal::CommandGraph &graph, metal::MetalBuffer packed,
    metal::MetalBuffer attention, metal::MetalBuffer hidden,
    metal::MetalBuffer sums, uint32_t tokens, uint32_t stride,
    uint32_t queryHeads, kv::Layout layout) {
  const KernelLayout kernelLayout = attentionKernelLayout(layout, queryHeads);
  if (kernelLayoutHeadDim(kernelLayout) != RICHENGINE_KV_HEAD_DIMENSION)
    throw std::invalid_argument("no paged prefill gate sums kernel for layout");
  FullPrefillParams params{.tokens = tokens, .stride = stride};
  graph.add(gateKernel(kPrefillAttention, true, kernelLayout, "_sums"),
            {packed, attention, hidden, sums}, params,
            {elementwiseGroups(uint64_t{tokens} * queryHeads *
                               layout.headDimension),
             1, 1});
}

void PagedAttention::addVerifyProjection(
    metal::CommandGraph &graph, metal::MetalBuffer packed,
    const NormWeights &queryNorm, const NormWeights &keyNorm,
    metal::MetalBuffer ropeCos, metal::MetalBuffer ropeSin,
    metal::MetalBuffer queries, metal::MetalBuffer chunkKeys,
    metal::MetalBuffer chunkValues, uint32_t queryHeads,
    kv::Layout layout, uint32_t lanes, uint32_t rows) {
  const KernelLayout kernelLayout = attentionKernelLayout(layout, queryHeads);
  const bool hasNorms = queryNorm.buffer && keyNorm.buffer;
  std::string kernel = std::string(kVerifyAttentionQkv) +
                       std::string(kernelSuffix(kernelLayout));
  if (hasNorms) kernel = qkNormKernel(kernel, queryNorm, keyNorm);
  FullDecodeBatchParams params{.lanes = lanes, .rows = rows};
  graph.add(kernel,
            {packed, hasNorms ? queryNorm.buffer : queries,
             hasNorms ? keyNorm.buffer : queries, ropeCos, ropeSin, queries,
             chunkKeys, chunkValues},
            params,
            {uint64_t{rows} * (queryHeads + layout.kvHeads), lanes, 1},
            {layout.headDimension, 1, 1});
}

PreparedInput PagedAttention::addVerifyGate(
    metal::CommandGraph &graph, metal::MetalBuffer packed,
    metal::MetalBuffer attention, metal::MetalBuffer hidden,
    uint32_t queryHeads, kv::Layout layout, uint32_t lanes,
    LinearScratch scratch, LinearInput input, bool queryGate, uint32_t rows) {
  const KernelLayout kernelLayout = attentionKernelLayout(layout, queryHeads);
  FullDecodeBatchParams params{.lanes = lanes, .rows = rows};
  if (input == LinearInput::Plain) {
    graph.addPatchable(gateKernel(kVerifyAttention, queryGate, kernelLayout),
              {packed, attention, hidden}, params,
              {elementwiseGroups(uint64_t{lanes} * rows * queryHeads *
                                 layout.headDimension),
               1, 1});
    return {};
  }
  const uint32_t width = queryHeads * layout.headDimension;
  requireTableScratch(scratch, input, width, lanes * rows);
  const std::string kernel =
      gateKernel(kVerifyAttention, queryGate, kernelLayout, tableSuffix(input));
  graph.add(kernel,
            {packed, attention, hidden, scratch.input, scratch.sums},
            params,
            {uint64_t{width} / 64 * lanes * rows / 8, 1, 1}, {256, 1, 1});
  return {hidden, input};
}

kv::ChunkedPrefillParams PagedAttention::prefillParams(
    uint64_t logicalPosition, uint32_t chunkTokens, uint32_t chunkStride,
    uint32_t pageTableEntries) {
  if (logicalPosition > std::numeric_limits<uint32_t>::max())
    throw std::overflow_error("KV logical position exceeds kernel ABI");
  kv::ChunkedPrefillParams params{static_cast<uint32_t>(logicalPosition),
                                  chunkTokens, chunkStride, pageTableEntries,
                                  {}};
  const std::string_view error = kv::chunkedPrefillValidationError(params);
  if (!error.empty()) throw std::invalid_argument(std::string(error));
  return params;
}

kv::ChunkedPrefillParams PagedAttention::verifyParams(
    uint64_t logicalPosition, uint32_t pageTableEntries) {
  return prefillParams(logicalPosition, kv::kVerifyRows,
                       kv::kVerifyChunkStride, pageTableEntries);
}

kv::ChunkedPrefillParams PagedAttention::verifyTreeParams(
    uint64_t logicalPosition, uint32_t pageTableEntries, uint32_t liveNodes) {
  return prefillParams(
      logicalPosition,
      std::min(liveNodes, RICHENGINE_TREE_VERIFY_NODES),
      kv::kVerifyChunkStride, pageTableEntries);
}

void PagedAttention::addPrefillStore(
    metal::CommandGraph &graph, RichKvLayer layer, metal::MetalBuffer chunkKeys,
    metal::MetalBuffer chunkValues, metal::MetalBuffer pageTable,
    const kv::ChunkedPrefillParams &params, kv::Layout layout) {
  (void)storageKernelLayout(layout);
  RichChunkedPrefillParams chunk{};
  chunk.committed_tokens = params.committed_tokens;
  chunk.chunk_tokens = params.chunk_tokens;
  chunk.chunk_stride = params.chunk_stride;
  chunk.page_table_entries = params.page_table_entries;
  chunk.kv = layer;
  if (std::string_view error = kv::chunkedPrefillValidationError(chunk); !error.empty())
    throw std::invalid_argument(std::string(error));
  graph.add(storeKernel(kPrefillAttention, layout.format, layout),
            {chunkKeys, chunkValues, pageTable}, chunk,
            {2 * params.chunk_tokens * layout.kvHeads, 1, 1},
            {layout.headDimension, 1, 1});
}

void PagedAttention::addPrefill(
    metal::CommandGraph &graph, RichKvLayer layer, metal::MetalBuffer queries,
    metal::MetalBuffer output, metal::MetalBuffer partials,
    metal::MetalBuffer statistics, metal::MetalBuffer pageTable,
    const kv::ChunkedPrefillParams &chunk, const PrefillAttentionPlan &plan) {
  if (chunk.chunk_tokens != plan.rows)
    throw std::invalid_argument("prefill attention rows do not match plan");
  if (const std::string_view error = kv::chunkedPrefillValidationError(chunk);
      !error.empty())
    throw std::invalid_argument(std::string(error));
  if (partials.sizeBytes() < plan.workspace.partialsBytes ||
      statistics.sizeBytes() < plan.workspace.statisticsBytes)
    throw std::invalid_argument("prefill attention scratch is smaller than its bound");
  const RichPrefillAttentionParams attention{
      chunk.committed_tokens, chunk.chunk_tokens, chunk.chunk_stride,
      chunk.page_table_entries, layer, plan.splits, plan.scoreScale};
  // The _swa splits take the window as a constant after the parameters;
  // every canvas split declares the same trailing constant and ignores it
  // when its name is windowless.
  if (plan.splitPipeline.find("_swa_") != std::string::npos ||
      plan.splitPipeline.find("_canvas") != std::string::npos) {
    graph.addPatchableTail(plan.splitPipeline,
              {queries, partials, statistics, pageTable}, attention,
              plan.windowTokens, plan.splitGroups);
  } else {
    graph.addPatchable(plan.splitPipeline,
              {queries, partials, statistics, pageTable}, attention,
              plan.splitGroups);
  }
  graph.addPatchable(plan.reducePipeline,
            {partials, statistics, output}, attention, plan.reduceGroups,
            plan.reduceThreads);
}

void PagedAttention::addVerify(
    metal::CommandGraph &graph, RichKvLayer layer,
    PagedVerifyBuffers buffers,
    std::span<const kv::ChunkedPrefillParams> chunks,
    const VerifyAttentionPlan &plan, metal::MetalBuffer gatePacked,
    metal::MetalBuffer gateHidden) {
  constexpr uint32_t maximumLanes = RICHENGINE_MAXIMUM_BATCH_WIDTH;
  if (chunks.size() != plan.lanes || buffers.pageTables.size() != maximumLanes)
    throw std::invalid_argument("invalid paged verify batch");
  if (buffers.partials.sizeBytes() < plan.workspace.partialsBytes ||
      buffers.statistics.sizeBytes() < plan.workspace.statisticsBytes)
    throw std::invalid_argument("verify attention scratch is smaller than its bound");
  std::array<kv::ChunkedPrefillParams, maximumLanes> stores{};
  std::array<kv::VerifyAttentionParams, maximumLanes> attention{};
  for (uint32_t lane = 0; lane < plan.lanes; ++lane) {
    stores[lane] = chunks[lane];
    stores[lane].kv = layer;
    attention[lane] = {chunks[lane].committed_tokens,
                       chunks[lane].page_table_entries,
                       layer,
                       plan.laneSplits[lane],
                       plan.splits,
                       chunks[lane].chunk_tokens,
                       plan.rowCapacity, plan.scoreScale};
  }
  std::array<metal::MetalBuffer, 4> tables = {buffers.pageTables[0], buffers.pageTables[1],
                                              buffers.pageTables[2], buffers.pageTables[3]};
  // The parameters carry each lane's committed length and split count; the
  // dispatches write them once per verify encoding, so their payloads are
  // patchable for a baked span's replays.
  graph.addPatchable(plan.storePipeline_,
            {buffers.chunkKeys, buffers.chunkValues, tables[0], tables[1], tables[2],
             tables[3]},
            stores, plan.storeGroups_, plan.storeThreads_);
  // A tree plan's split kernels bind each lane's ancestor bitmasks after the
  // page tables; the chain kernels have no such binding. The stem names it —
  // a tree tile's row capacity can equal the chain's row count.
  const bool tree = plan.splitPipeline.starts_with(kVerifyTreeAttention);
  if (tree && !buffers.treeMasks)
    throw std::invalid_argument("tree verify requires its ancestor masks");
  const auto splitBuffers =
      tree ? std::vector<metal::MetalBuffer>{buffers.queries,
               buffers.partials, buffers.statistics, tables[0],
               tables[1], tables[2], tables[3], buffers.treeMasks}
           : std::vector<metal::MetalBuffer>{buffers.queries,
               buffers.partials, buffers.statistics, tables[0],
               tables[1], tables[2], tables[3]};
  // The _swa splits take the window as a constant after the lane parameters.
  if (plan.splitPipeline.find("_swa_") != std::string::npos) {
    graph.addPatchableTail(plan.splitPipeline, std::move(splitBuffers),
                           attention, plan.windowTokens, plan.splitGroups);
  } else {
    graph.addPatchable(plan.splitPipeline, std::move(splitBuffers),
                       attention, plan.splitGroups);
  }
  // The fused reduce applies the query gate only when the out-projection
  // reads Plain input; a table or packed consumer keeps the separate gate
  // dispatch that writes its operand (addVerifyGate).
  if (gateHidden) {
    // plan.reducePipeline's stem is the plan's own; the fused gate keeps
    // its layout suffix and swaps the stem for the gate variant.
    const std::string_view pipeline = plan.reducePipeline;
    const std::string suffix = std::string(pipeline.substr(
        pipeline.starts_with(kVerifyTreeAttentionReduce)
            ? kVerifyTreeAttentionReduce.size()
            : kVerifyAttentionReduce.size()));
    if (gatePacked) {
      const std::string gate =
          std::string(tree ? kVerifyTreeAttentionReduceGate
                           : kVerifyAttentionReduceGate) +
          suffix;
      graph.addPatchable(gate,
                {buffers.partials, buffers.statistics, buffers.output,
                 std::move(gatePacked), std::move(gateHidden)},
                attention, plan.reduceGroups);
    } else {
      // The no-gate targets' reduce/gather fusion: the hidden layout alone
      // is written, so the reduce threads bound is the head dimension.
      if (tree)
        throw std::invalid_argument("tree verify has no fused gather");
      graph.addPatchable(std::string(kVerifyAttentionReduceGather) + suffix,
                {buffers.partials, buffers.statistics, std::move(gateHidden)},
                attention, plan.reduceGroups, plan.reduceThreads_);
    }
  } else {
    graph.addPatchable(plan.reducePipeline,
              {buffers.partials, buffers.statistics, buffers.output},
              attention, plan.reduceGroups, plan.reduceThreads_);
  }
}

void PagedAttention::addVerifyTreeCompact(
    metal::CommandGraph &graph, RichKvLayer layer,
    std::span<const metal::MetalBuffer> pageTables,
    metal::MetalBuffer retainedPath, metal::MetalBuffer retained,
    std::span<const kv::ChunkedPrefillParams> chunks, uint32_t lanes,
    kv::Layout layout) {
  constexpr uint32_t maximumLanes = RICHENGINE_MAXIMUM_BATCH_WIDTH;
  if (!lanes || lanes > maximumLanes || chunks.size() != lanes ||
      pageTables.size() != maximumLanes)
    throw std::invalid_argument("invalid tree verify compact batch");
  if (layout.format == kv::Format::Float8E4M3)
    throw std::invalid_argument("tree verify has no fp8 kernels");
  std::array<kv::ChunkedPrefillParams, maximumLanes> params{};
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    params[lane] = chunks[lane];
    params[lane].kv = layer;
  }
  std::string kernel = std::string(kVerifyTreeAttention) + "_" +
                       formatTag(layout.format) + "_compact" +
                       std::string(storageSuffix(layout));
  graph.add(kernel,
            {pageTables[0], pageTables[1], pageTables[2], pageTables[3],
             retainedPath, retained},
            params,
            {uint64_t{lanes} * layout.kvHeads, 1, 1},
            metal::DispatchSize{layout.headDimension, 1, 1});
}

} // namespace richengine::ops
