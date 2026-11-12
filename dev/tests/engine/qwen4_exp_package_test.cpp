// The qwen4exp packed MoE block's GGUF segments at the padded expert
// width. The model's 640-wide expert intermediate cannot ride the GGUF
// image's 256-row/256-column tiles — moe_gguf_segment strides expert e's
// planes by e * output_size rows, which the tile-major planes address
// exactly only at a whole tile boundary — so the packer zero-pads every
// expert dimension to packedExpertWidth (768) and the runtime geometry,
// MoE shape and segment readers all run at that width. This test pins
// both halves: the geometry resolves to 768 and the packed segments'
// descriptors load, while the unpadded 640 dimensions the packer must
// never emit stay refused.

#include "TestBuffers.hpp"
#include "model/GgufFile.hpp"
#include "model/GgufImageLayout.hpp"
#include "model/Qwen4Exp.hpp"
#include "model/TargetModel.hpp"
#include "model/WeightImages.hpp"
#include "model/WeightLayout.hpp"
#include "metal/MetalBackend.hpp"
#include "ops/MoE.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <system_error>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

namespace {

using richengine::model::GgufPlaneBytes;
using richengine::model::GgufTensorDescriptor;
using richengine::model::WeightFile;
using richengine::model::WeightImages;
using richengine::model::WeightStoreError;
using richengine::model::kWeightFileAlignment;
using richengine::model::ggufPlaneBytes;
using richengine::model::packedImage;
using richengine::model::qwen4ExpTargetLayout;
using richengine::model::readQuantizedSegment;
using richengine::model::targetModelGeometry;
using richengine::model::TargetModelWeights;
using richengine::metal::MetalBackend;
using richengine::ops::MoeConfig;
using richengine::ops::MoeShape;
using richengine::ops::MoE;
using richengine::ops::QuantizedSegment;
using richengine::ops::WeightLayout;

[[noreturn]] void fail(const std::string &message) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
}

void require(bool condition, const std::string &message) {
    if (!condition) fail(message);
}

template <typename Function>
void requirePackedError(Function &&function, const std::string &message) {
    try {
        function();
    } catch (const WeightStoreError &) {
        return;
    }
    fail(message);
}

class TempDirectory final {
public:
    TempDirectory() {
        std::string pattern =
            (std::filesystem::temp_directory_path() /
             "richengine-qwen4exp-package.XXXXXX").string();
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

void storeLittleEndian32(uint8_t *destination, uint32_t value) {
    destination[0] = static_cast<uint8_t>(value);
    destination[1] = static_cast<uint8_t>(value >> 8);
    destination[2] = static_cast<uint8_t>(value >> 16);
    destination[3] = static_cast<uint8_t>(value >> 24);
}

uint64_t alignPacked(uint64_t value) {
    return (value + kWeightFileAlignment - 1) &
        ~(kWeightFileAlignment - 1);
}

// One tensor of a packed GGUF image: the descriptor's words and its plane
// byte counts, in the file's section order (descriptor, plane0, plane1,
// meta — the zero-sized ones absent).
struct SyntheticTensor {
    std::array<uint32_t, 7> words;
    std::array<uint64_t, 3> planes;
};

// The Q8_0 descriptor of a [rows, columns] tensor whose planes occupy
// ggufPlaneBytes' counts.
SyntheticTensor q8oTensor(uint32_t rows, uint32_t columns,
                          const QuantFormat &format) {
    const GgufPlaneBytes planes = ggufPlaneBytes(format, rows, columns);
    return {{format.ggml_type, rows, columns, format.plane0_bytes,
             format.plane1_bytes, format.meta_bytes, format.meta_groups},
            {planes.plane0, planes.plane1, planes.meta}};
}

// The F32 descriptor of a [rows, columns] float tensor.
SyntheticTensor f32Tensor(uint32_t rows, uint32_t columns) {
    return {{richengine::model::ggml::kF32, rows, columns, 0, 0, 0, 0},
            {uint64_t{rows} * columns * sizeof(float), 0, 0}};
}

// A packed file holding the given tensors back to back, each descriptor
// written at its 16 KiB-aligned section like the packer's.
void writePackedGguf(const std::filesystem::path &path,
                     std::string_view magic, uint32_t layer, uint32_t type,
                     const std::vector<SyntheticTensor> &tensors) {
    require(magic.size() == 8, "synthetic magic has the wrong size");
    // The section sizes in file order, and each tensor's descriptor
    // offset among them.
    std::vector<uint64_t> sizes;
    std::vector<size_t> descriptorSection;
    for (const SyntheticTensor &tensor : tensors) {
        descriptorSection.push_back(sizes.size());
        sizes.push_back(sizeof(GgufTensorDescriptor));
        for (uint64_t bytes : tensor.planes)
            if (bytes) sizes.push_back(bytes);
    }
    for (uint64_t bytes : sizes)
        require(bytes > 0, "synthetic section is empty");

    uint64_t offset = 16;
    std::vector<uint64_t> starts;
    for (uint64_t bytes : sizes) {
        starts.push_back(alignPacked(offset));
        offset = starts.back() + bytes;
    }
    const uint64_t fileBytes = alignPacked(offset);
    require(fileBytes <=
                static_cast<uint64_t>(std::numeric_limits<off_t>::max()),
            "synthetic packed file is too large");

    std::filesystem::create_directories(path.parent_path());
    int descriptor = open(path.c_str(), O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC,
                          0600);
    if (descriptor < 0) fail("unable to create synthetic packed file");
    std::array<uint8_t, 16> header{};
    std::memcpy(header.data(), magic.data(), magic.size());
    storeLittleEndian32(header.data() + 8, layer);
    storeLittleEndian32(header.data() + 12, type);
    ssize_t written = pwrite(descriptor, header.data(), header.size(), 0);
    if (written != static_cast<ssize_t>(header.size())) {
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
        written = pwrite(descriptor, bytes.data(), bytes.size(),
                         static_cast<off_t>(
                             starts[descriptorSection[i]]));
        if (written != static_cast<ssize_t>(bytes.size())) {
            close(descriptor);
            fail("unable to write a synthetic GGUF descriptor");
        }
    }
    close(descriptor);
}

// The geometry the packed files drive: the 640-wide model intermediate
// resolves to the 768 the GGUF tiles store, and the wide 512-expert MoE
// shape and its plans accept it.
void testQwen4ExpGeometry() {
    TargetModelWeights weights;
    weights.layout = qwen4ExpTargetLayout();
    const auto geometry = targetModelGeometry(weights);
    require(geometry.expertIntermediateSize == 768,
            "qwen4exp geometry did not resolve the packed expert width");
    require(geometry.ffnScratchWidth() == 768,
            "qwen4exp FFN scratch did not take the packed expert width");
    const MoeShape shape{geometry.hiddenSize, geometry.experts,
                         geometry.expertsPerToken,
                         geometry.expertIntermediateSize,
                         WeightLayout::Block32, GGUF_FMT_Q80, true};
    require(shape.valid() && shape.wideExperts() &&
                shape.routerWidth() == 512,
            "the padded qwen4exp MoE shape is refused");
    const auto decode = MoE::decodePlan(
        shape, 1, richengine::model::ExecutionLimits::targetVerifyRows,
        MoeConfig{});
    require(decode.workspace().expertIntermediateBytes,
            "qwen4exp decode plan sized no expert intermediate");
    const auto prefill = MoE::prefillPlan(shape, 64, MoeConfig{});
    require(prefill.workspace().expertIntermediateBytes,
            "qwen4exp prefill plan sized no expert intermediate");
}

// The MoE block's sections in readMoeBlock's order at shrunken-but-tiled
// dimensions: hidden 256, two experts, the packed 768 intermediate. Every
// Q8_0 descriptor must load with its logical dimensions intact.
void testPackedMoeBlock(MetalBackend &backend,
                        const std::filesystem::path &root) {
    constexpr uint32_t hidden = 256, experts = 2, packed = 768;
    const QuantFormat &q80 = kQuantFormats[GGUF_FMT_Q80];
    const std::vector<SyntheticTensor> tensors{
        f32Tensor(experts, hidden),
        q8oTensor(experts * packed, hidden, q80),
        q8oTensor(experts * packed, hidden, q80),
        q8oTensor(experts * hidden, packed, q80),
        q8oTensor(packed, hidden, q80),
        q8oTensor(packed, hidden, q80),
        q8oTensor(hidden, packed, q80),
        f32Tensor(1, hidden),
    };
    const auto path = root / "layer-0.bin";
    writePackedGguf(path, "MDFQ0001", 0, 0, tensors);
    WeightImages images(backend);
    WeightFile file = images.load(
        packedImage(path, "test/layer-0.bin", "MDFQ0001", 0, 0));
    const std::array<std::pair<uint32_t, uint32_t>, 8> expected{{
        {experts, hidden},          // router, F32
        {experts * packed, hidden}, // routed gate
        {experts * packed, hidden}, // routed up
        {experts * hidden, packed}, // routed down
        {packed, hidden},           // shared gate
        {packed, hidden},           // shared up
        {hidden, packed},           // shared down
        {1, hidden},                // shared scalar gate, F32
    }};
    const std::array<bool, 8> floats{{true, false, false, false,
                                      false, false, false, true}};
    for (size_t i = 0; i < tensors.size(); ++i) {
        const QuantizedSegment segment =
            readQuantizedSegment(file, "moe-" + std::to_string(i));
        require(segment.outputSize == expected[i].first &&
                    segment.inputSize == expected[i].second,
                "packed MoE segment dimensions did not survive the "
                "descriptor");
        require(segment.isFloat() == floats[i],
                "packed MoE segment float flag is wrong");
        if (!floats[i])
            require(segment.formatId == GGUF_FMT_Q80,
                    "packed MoE segment is not Q8_0");
    }
    file.finish();
}

// The dimensions the packer must never emit: the model's true 640
// intermediate fails the tiled descriptor check on whichever axis it
// lands, and a plane count that does not match the format's layout stays
// refused at the padded width too.
void testUnpaddedExpertDimensions(MetalBackend &backend,
                                  const std::filesystem::path &root) {
    constexpr uint32_t hidden = 256;
    const QuantFormat &q80 = kQuantFormats[GGUF_FMT_Q80];
    // A shared-down shape [hidden, 640]: the 640 input is no 256-column
    // tile multiple.
    {
        const auto path = root / "bad-input.bin";
        writePackedGguf(path, "MDFQ0001", 0, 0,
                        {q8oTensor(hidden, 640, q80)});
        WeightImages images(backend);
        WeightFile file = images.load(
            packedImage(path, "test/bad-input.bin", "MDFQ0001", 0, 0));
        requirePackedError(
            [&] { (void)readQuantizedSegment(file, "bad-input"); },
            "a 640-column expert segment was accepted");
    }
    // A shared gate/up shape [640, hidden]: the 640 output is no
    // 256-row tile multiple.
    {
        const auto path = root / "bad-rows.bin";
        writePackedGguf(path, "MDFQ0001", 0, 0,
                        {q8oTensor(640, hidden, q80)});
        WeightImages images(backend);
        WeightFile file = images.load(
            packedImage(path, "test/bad-rows.bin", "MDFQ0001", 0, 0));
        requirePackedError(
            [&] { (void)readQuantizedSegment(file, "bad-rows"); },
            "a 640-row expert segment was accepted");
    }
    // The padded width with a plane count off the format's layout.
    {
        const auto path = root / "bad-planes.bin";
        SyntheticTensor bad = q8oTensor(hidden, 768, q80);
        bad.planes[0] -= q80.plane0_bytes;
        writePackedGguf(path, "MDFQ0001", 0, 0, {bad});
        WeightImages images(backend);
        WeightFile file = images.load(
            packedImage(path, "test/bad-planes.bin", "MDFQ0001", 0, 0));
        requirePackedError(
            [&] { (void)readQuantizedSegment(file, "bad-planes"); },
            "an expert segment with inconsistent plane sizes was "
            "accepted");
    }
}

} // namespace

int main(int argc, const char *argv[]) {
    if (argc != 2) {
        std::cerr << "usage: qwen4-exp-package <metallib>\n";
        return 2;
    }
    try {
        MetalBackend backend(argv[1]);
        testQwen4ExpGeometry();
        TempDirectory temporary;
        testPackedMoeBlock(backend, temporary.path());
        testUnpaddedExpertDimensions(backend, temporary.path());
    } catch (const std::exception &error) {
        fail(error.what());
    }
    std::cout << "qwen4-exp-package: ok\n";
    return 0;
}
