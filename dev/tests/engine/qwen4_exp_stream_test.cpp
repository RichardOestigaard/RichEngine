// The streamed routed experts of a GGUF MoE block against the resident
// path: one synthetic packed layer whose Q8_0 planes carry per-expert
// patterns, loaded once as resident WeightFile segments and once through
// PackedExpertStore + ExpertArena. The resident dispatch is ops::MoE::add
// with the fused gate/up tile off (RICHENGINE_MOE_GGUF_FUSED_OFF), so both
// sides run the same three split staged passes and may differ only in how
// a tile's expert resolves its planes — resident segment offsets against
// the arena's slot and chunk tables. The streamed side replays the
// encoder's route-first sequence: routing, host union read, per-round
// stage + passes + combine. Two arena geometries cover the cases: every
// expert resident across a floor and an elastic chunk, and one slot,
// which forces a stage/evict round per expert.

#include "model/ExpertArena.hpp"
#include "model/GgufFile.hpp"
#include "model/GgufImageLayout.hpp"
#include "model/PackedExpertStore.hpp"
#include "model/SsdReaders.hpp"
#include "model/WeightImages.hpp"
#include "model/WeightLayout.hpp"
#include "model/WeightSource.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "ops/MoE.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <system_error>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

namespace {

using richengine::model::ExpertArena;
using richengine::model::ExpertArenaOptions;
using richengine::model::ExpertSlotPlanes;
using richengine::model::GgufPlaneBytes;
using richengine::model::GgufTensorDescriptor;
using richengine::model::ggufPlaneBytes;
using richengine::model::kPackedExpertPlanes;
using richengine::model::kPackedExpertProjections;
using richengine::model::kPackedExpertRegions;
using richengine::model::kSsdReaders;
using richengine::model::kWeightFileAlignment;
using richengine::model::openRowReader;
using richengine::model::packedExpertTable;
using richengine::model::PackedExpertStore;
using richengine::model::PackedExpertStoreOptions;
using richengine::model::PackedExpertUse;
using richengine::model::packedImage;
using richengine::model::readQuantizedSegment;
using richengine::model::SsdReaders;
using richengine::model::WeightFile;
using richengine::model::WeightImages;
using richengine::model::WeightSource;
using richengine::metal::BufferStorage;
using richengine::metal::CommandGraph;
using richengine::metal::MetalBackend;
using richengine::metal::MetalBuffer;
using richengine::ops::BlockMoeWeights;
using richengine::ops::kMoeScratchFields;
using richengine::ops::moeGgufPrefillTile;
using richengine::ops::MoeBuffers;
using richengine::ops::MoeConfig;
using richengine::ops::MoeExpertTile;
using richengine::ops::MoeGgufTile;
using richengine::ops::MoePlan;
using richengine::ops::MoeScratch;
using richengine::ops::MoeShape;
using richengine::ops::MoeStreamBinding;
using richengine::ops::MoeWeights;
using richengine::ops::MoeWorkspace;
using richengine::ops::MoE;
using richengine::ops::WeightLayout;

[[noreturn]] void fail(const std::string &message) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
}

void require(bool condition, const std::string &message) {
    if (!condition) fail(message);
}

class TempDirectory final {
public:
    TempDirectory() {
        std::string pattern =
            (std::filesystem::temp_directory_path() /
             "richengine-qwen4exp-stream.XXXXXX").string();
        char *created = mkdtemp(pattern.data());
        if (!created) fail("unable to create temporary directory");
        path_ = created;
    }
    ~TempDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }
    [[nodiscard]] const std::filesystem::path &path() const noexcept {
        return path_;
    }
private:
    std::filesystem::path path_;
};

uint64_t alignPacked(uint64_t value) {
    return (value + kWeightFileAlignment - 1) &
        ~(uint64_t{kWeightFileAlignment} - 1);
}

// One tensor of the synthetic packed file: the descriptor's words and its
// plane byte counts.
struct SyntheticTensor {
    std::array<uint32_t, 7> words;
    std::array<uint64_t, 3> planes;
};

SyntheticTensor q8oTensor(uint32_t rows, uint32_t columns,
                          const QuantFormat &format) {
    const GgufPlaneBytes planes = ggufPlaneBytes(format, rows, columns);
    return {{format.ggml_type, rows, columns, format.plane0_bytes,
             format.plane1_bytes, format.meta_bytes, format.meta_groups},
            {planes.plane0, planes.plane1, planes.meta}};
}

SyntheticTensor f32Tensor(uint32_t rows, uint32_t columns) {
    return {{richengine::model::ggml::kF32, rows, columns, 0, 0, 0, 0},
            {uint64_t{rows} * columns * sizeof(float), 0, 0}};
}

// A packed file holding the tensors back to back like the packer's, each
// plane section filled by fill(tensor, plane, byte): plane is 0 plane0,
// 1 plane1, 2 meta; byte its offset inside the plane.
void writePackedGguf(const std::filesystem::path &path,
                     const std::vector<SyntheticTensor> &tensors,
                     const std::function<uint8_t(size_t, size_t, uint64_t)>
                         &fill) {
    std::vector<uint64_t> sizes;
    std::vector<size_t> descriptorSection;
    std::vector<std::array<int64_t, 3>> planeSection(tensors.size());
    for (size_t i = 0; i < tensors.size(); ++i) {
        planeSection[i] = {-1, -1, -1};
        descriptorSection.push_back(sizes.size());
        sizes.push_back(sizeof(GgufTensorDescriptor));
        for (uint32_t p = 0; p < 3; ++p)
            if (tensors[i].planes[p]) {
                planeSection[i][p] = static_cast<int64_t>(sizes.size());
                sizes.push_back(tensors[i].planes[p]);
            }
    }
    uint64_t offset = 16;
    std::vector<uint64_t> starts;
    for (uint64_t bytes : sizes) {
        starts.push_back(alignPacked(offset));
        offset = starts.back() + bytes;
    }
    const uint64_t fileBytes = alignPacked(offset);
    std::filesystem::create_directories(path.parent_path());
    int descriptor = open(path.c_str(), O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC,
                          0600);
    if (descriptor < 0) fail("unable to create synthetic packed file");
    std::array<uint8_t, 16> header{};
    std::memcpy(header.data(), "MDFQ0001", 8);
    if (pwrite(descriptor, header.data(), header.size(), 0) !=
        static_cast<ssize_t>(header.size())) {
        close(descriptor);
        fail("unable to write synthetic packed header");
    }
    if (ftruncate(descriptor, static_cast<off_t>(fileBytes)) != 0) {
        close(descriptor);
        fail("unable to size synthetic packed file");
    }
    for (size_t i = 0; i < tensors.size(); ++i) {
        const SyntheticTensor &tensor = tensors[i];
        std::array<uint8_t, sizeof(GgufTensorDescriptor)> bytes{};
        std::memcpy(bytes.data(), tensor.words.data(),
                    tensor.words.size() * sizeof(uint32_t));
        std::memcpy(bytes.data() + 32, tensor.planes.data(),
                    tensor.planes.size() * sizeof(uint64_t));
        if (pwrite(descriptor, bytes.data(), bytes.size(),
                   static_cast<off_t>(starts[descriptorSection[i]])) !=
            static_cast<ssize_t>(bytes.size())) {
            close(descriptor);
            fail("unable to write a synthetic GGUF descriptor");
        }
        for (uint32_t p = 0; p < 3; ++p) {
            const int64_t section = planeSection[i][p];
            if (section < 0) continue;
            const uint64_t planeBytes = tensor.planes[p];
            std::vector<uint8_t> plane(planeBytes);
            for (uint64_t b = 0; b < planeBytes; ++b)
                plane[b] = fill(i, static_cast<size_t>(p), b);
            if (pwrite(descriptor, plane.data(), plane.size(),
                       static_cast<off_t>(starts[section])) !=
                static_cast<ssize_t>(plane.size())) {
                close(descriptor);
                fail("unable to write a synthetic GGUF plane");
            }
        }
    }
    close(descriptor);
}

// The synthetic block: hidden 256, eight experts (the F32 router's addGgufFloat
// needs its output a multiple of 8), top-2 plus the shared route, a 256-wide
// intermediate — every dimension a whole 256-row tile.
constexpr uint32_t kHidden = 256;
constexpr uint32_t kExperts = 8;
constexpr uint32_t kTopK = 2;
constexpr uint32_t kIntermediate = 256;

// The file's tensors in readMoeBlock's order. Q8_0 plane0 holds signed
// quants that vary per 256-row expert tile (each tile's slice is 65536
// bytes at hidden 256) and per byte; meta is fp16 1.0 pairs; the floats
// are a small ramp so the router and scalar gate stay finite.
void writeFixtureFile(const std::filesystem::path &path) {
    const QuantFormat &q80 = kQuantFormats[GGUF_FMT_Q80];
    const std::vector<SyntheticTensor> tensors{
        f32Tensor(kExperts, kHidden),
        q8oTensor(kExperts * kIntermediate, kHidden, q80),
        q8oTensor(kExperts * kIntermediate, kHidden, q80),
        q8oTensor(kExperts * kHidden, kIntermediate, q80),
        q8oTensor(kIntermediate, kHidden, q80),
        q8oTensor(kIntermediate, kHidden, q80),
        q8oTensor(kHidden, kIntermediate, q80),
        f32Tensor(1, kHidden),
    };
    const auto fill = [](size_t tensor, size_t plane, uint64_t byte) {
        if (tensor == 0 || tensor == 7) {
            const uint32_t word = static_cast<uint32_t>(byte / 4);
            const uint32_t bits = 0x3C000000u + (word % 1024u) * 0x110u;
            return static_cast<uint8_t>(bits >> ((byte % 4) * 8));
        }
        if (plane == 2)
            return static_cast<uint8_t>((byte & 1) ? 0x3C : 0x00);
        const uint64_t expert = byte / 65536;
        return static_cast<uint8_t>(((byte * 7 + expert * 13 +
                                      tensor * 5) %
                                     17) -
                                    8);
    };
    writePackedGguf(path, tensors, fill);
}

MoeWeights loadWeights(MetalBackend &backend,
                       const std::filesystem::path &packed) {
    WeightImages images(backend);
    WeightFile file = images.load(
        packedImage(packed, "test/layer-0.bin", "MDFQ0001", 0, 0));
    BlockMoeWeights block;
    block.router = readQuantizedSegment(file, "router");
    block.gate.routed = readQuantizedSegment(file, "experts-gate");
    block.up.routed = readQuantizedSegment(file, "experts-up");
    block.down.routed = readQuantizedSegment(file, "experts-down");
    block.gate.shared = readQuantizedSegment(file, "shared-expert-gate");
    block.up.shared = readQuantizedSegment(file, "shared-expert-up");
    block.down.shared = readQuantizedSegment(file, "shared-expert-down");
    block.sharedScalarGate =
        readQuantizedSegment(file, "shared-expert-scalar-gate");
    file.finish();
    return MoeWeights{std::move(block)};
}

// The routed union the encoder reads off selectedExperts: rows *
// routesPerToken ids, routed ids below `experts`, the shared route's id
// `experts`, ~0u padding.
std::vector<uint32_t> routedUnion(const MetalBuffer &selectedExperts,
                                  const MoePlan &plan) {
    const uint32_t rows = plan.rows();
    const uint32_t routes = plan.shape().routesPerToken();
    const uint32_t experts = plan.shape().experts;
    const auto *ids =
        static_cast<const uint32_t *>(selectedExperts.contents());
    require(ids != nullptr, "selectedExperts is not CPU-visible");
    std::vector<uint32_t> unionExperts;
    std::vector<uint8_t> seen(experts, 0);
    for (uint64_t i = 0; i < uint64_t{rows} * routes; ++i) {
        const uint32_t expert = ids[i];
        if (expert < experts && !seen[expert]) {
            seen[expert] = 1;
            unionExperts.push_back(expert);
        }
    }
    return unionExperts;
}

struct StepBuffers final {
    MetalBuffer input;
    MetalBuffer residual;
    MetalBuffer output;
    MoeScratch scratch;
};

StepBuffers makeBuffers(MetalBackend &backend, const MoePlan &plan,
                        const MetalBuffer &output) {
    StepBuffers buffers;
    const uint64_t rowBytes =
        uint64_t{plan.rows()} * plan.shape().hiddenSize * sizeof(uint16_t);
    buffers.input = backend.allocateBuffer(rowBytes, BufferStorage::Shared,
                                           "stream-input");
    buffers.residual = backend.allocateBuffer(rowBytes, BufferStorage::Shared,
                                              "stream-residual");
    std::memset(buffers.residual.contents(), 0, rowBytes);
    buffers.output = output;
    const MoeWorkspace &workspace = plan.workspace();
    for (const auto &field : kMoeScratchFields)
        buffers.scratch.*field.buffer =
            workspace.*field.bytes
                ? backend.allocateBuffer(workspace.*field.bytes,
                                         BufferStorage::Shared,
                                         "stream-moe-scratch")
                : MetalBuffer{};
    // bf16 rows of small mixed values: {1.0, 0.5, -1.0, 2.0} cycling.
    static constexpr uint16_t values[4] = {0x3F80, 0x3F00, 0xBF80, 0x4000};
    auto *input = static_cast<uint16_t *>(buffers.input.contents());
    const uint64_t elements = rowBytes / sizeof(uint16_t);
    for (uint64_t i = 0; i < elements; ++i) input[i] = values[i % 4];
    return buffers;
}

MoeBuffers moeBuffers(const StepBuffers &buffers) {
    return {buffers.input, buffers.residual, buffers.output, buffers.scratch,
            {}, {}};
}

// The encoder's streamed sequence for one block: routing alone, the union
// read, then per arena-sized round a stage and that round's passes, and
// the combine last.
void runStreamed(MetalBackend &backend, const MoeWeights &weights,
                 const MoePlan &plan, ExpertArena &arena,
                 PackedExpertStore &store, const StepBuffers &buffers) {
    CommandGraph route;
    MoE::addRouting(route, moeBuffers(buffers), weights, plan);
    (void)backend.submitCommand(route.dispatches());
    const std::vector<uint32_t> unionExperts =
        routedUnion(buffers.scratch.selectedExperts, plan);
    require(!unionExperts.empty(), "the synthetic routes selected nothing");
    CommandGraph work;
    MoE::addGgufGather(work, moeBuffers(buffers), weights, plan);
    const MoeStreamBinding binding{arena.slotTable(),
                                   arena.chunkAddresses(),
                                   arena.layout().floorSlots,
                                   arena.layout().chunkSlots};
    const uint32_t slots = arena.layout().slots;
    const std::span<const uint32_t> all(unionExperts);
    uint64_t load = 0;
    for (size_t begin = 0; begin < all.size(); begin += slots) {
        const size_t end = std::min(begin + size_t{slots}, all.size());
        arena.stage(store, all.subspan(begin, end - begin), ++load,
                    PackedExpertUse::Prefill);
        MoE::addGgufPasses(work, moeBuffers(buffers), weights, plan,
                           &binding);
        (void)backend.submitCommand(work.dispatches());
        work = CommandGraph{};
    }
    CommandGraph combine;
    MoE::addCombine(combine, moeBuffers(buffers), weights, plan);
    (void)backend.submitCommand(combine.dispatches());
}

void runResident(MetalBackend &backend, const MoeWeights &weights,
                 const MoePlan &plan, const StepBuffers &buffers) {
    CommandGraph graph;
    MoE::add(graph, moeBuffers(buffers), weights, plan);
    (void)backend.submitCommand(graph.dispatches());
}

// One plan shape's streamed-vs-resident comparison at one arena geometry.
void checkPlan(MetalBackend &backend, const MoeWeights &weights,
               PackedExpertStore &store, uint32_t rows, bool decode,
               uint32_t arenaSlots) {
    const MoeShape shape{kHidden, kExperts, kTopK, kIntermediate,
                         WeightLayout::Block32, GGUF_FMT_Q80, true};
    require(shape.valid(), "the synthetic MoE shape is refused");
    const MoePlan plan =
        decode ? MoE::decodePlan(shape, 1, 8,
                                 MoeConfig{.expertTile = MoeExpertTile::M8,
                                           .ggufTile = MoeGgufTile::Staged})
               : MoE::prefillPlan(
                     shape, rows,
                     MoeConfig{.expertTile = moeGgufPrefillTile(
                                   shape, rows, MoeGgufTile::Staged),
                               .ggufTile = MoeGgufTile::Staged});
    const uint64_t rowBytes =
        uint64_t{plan.rows()} * shape.hiddenSize * sizeof(uint16_t);
    const auto table = packedExpertTable(
        "test-experts", weights.blocks().gate.routed,
        weights.blocks().up.routed, weights.blocks().down.routed, kExperts);
    ExpertSlotPlanes planes{};
    for (uint32_t z = 0; z < kPackedExpertProjections; ++z)
        for (uint32_t p = 0; p < kPackedExpertPlanes; ++p)
            planes[z][p] =
                table.regions[z * kPackedExpertPlanes + p].sliceBytes;
    ExpertArenaOptions arenaOptions;
    arenaOptions.bytes = uint64_t{1} << 40;
    arenaOptions.largestBuffer =
        backend.capabilities().maxBufferLengthBytes;
    arenaOptions.experts = kExperts;
    arenaOptions.slots = arenaSlots;
    ExpertArena arena(backend, planes, table, arenaOptions);
    require(arena.layout().slots == arenaSlots,
            "the arena did not take its slot count");

    MetalBuffer residentOut = backend.allocateBuffer(
        rowBytes, BufferStorage::Shared, "stream-resident-out");
    MetalBuffer streamedOut = backend.allocateBuffer(
        rowBytes, BufferStorage::Shared, "stream-streamed-out");
    const StepBuffers resident =
        makeBuffers(backend, plan, residentOut);
    const StepBuffers streamed =
        makeBuffers(backend, plan, streamedOut);
    runResident(backend, weights, plan, resident);
    runStreamed(backend, weights, plan, arena, store, streamed);
    require(std::memcmp(residentOut.contents(), streamedOut.contents(),
                        rowBytes) == 0,
            decode ? "streamed decode output differs from resident"
                   : "streamed prefill output differs from resident");
}

} // namespace

int main(int argc, const char *argv[]) {
    if (argc != 2) {
        std::cerr << "usage: qwen4-exp-stream <metallib>\n";
        return 2;
    }
    // The split passes are what the streamed kernels mirror; the fused
    // gate/up tile is covered by gguf-moe's own numerics tests.
    setenv("RICHENGINE_MOE_GGUF_FUSED_OFF", "1", 1);
    try {
        MetalBackend backend(argv[1]);
        TempDirectory temporary;
        const auto packed = temporary.path() / "layer-0.bin";
        writeFixtureFile(packed);
        MoeWeights weights = loadWeights(backend, packed);
        const auto table = packedExpertTable(
            "test-experts", weights.blocks().gate.routed,
            weights.blocks().up.routed, weights.blocks().down.routed,
            kExperts);
        PackedExpertStoreOptions storeOptions;
        storeOptions.cacheBytes = uint64_t{64} << 20;
        storeOptions.readers = std::make_shared<SsdReaders>(kSsdReaders);
        storeOptions.prefetchSlices = 4 * kPackedExpertRegions;
        PackedExpertStore store(
            table,
            openRowReader(std::make_shared<WeightSource>(packed)),
            std::move(storeOptions));
        // Every expert resident at once (a floor chunk plus one elastic
        // slot), then one slot, which stages and evicts per expert.
        checkPlan(backend, weights, store, 8, false, kExperts);
        checkPlan(backend, weights, store, 2, false, kExperts);
        checkPlan(backend, weights, store, 0, true, kExperts);
        checkPlan(backend, weights, store, 0, true, 1);
        checkPlan(backend, weights, store, 2, false, 1);
    } catch (const std::exception &error) {
        fail(error.what());
    }
    std::cout << "qwen4-exp-stream: ok\n";
    return 0;
}
