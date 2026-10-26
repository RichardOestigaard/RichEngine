#include "WeightStore.hpp"

#include "Checked.hpp"
#include "metal/abi/Gguf.h"
#include "model/GgufFile.hpp"
#include "model/GgufImageLayout.hpp"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <cstring>
#include <sstream>
#include <tuple>
#include <utility>

namespace richengine::model {

namespace {

[[nodiscard]] uint64_t q4Elements(uint32_t outputSize, uint32_t inputSize) {
    if (!outputSize || !inputSize || inputSize % kQ4GroupElements) {
        throw WeightStoreError(
            "Q4 projection dimensions must be positive and input-aligned");
    }
    return checkedMultiply<WeightStoreError>(outputSize, inputSize, "Q4 element count");
}

} // namespace

uint64_t q4PackedBytes(uint32_t outputSize, uint32_t inputSize) {
    uint64_t elements = q4Elements(outputSize, inputSize);
    return checkedMultiply<WeightStoreError>(elements / 16, 9, "Q4 packed byte count");
}

void validateQ4Layout(uint32_t outputSize, uint32_t inputSize) {
    static_cast<void>(q4Elements(outputSize, inputSize));
    if (outputSize % kQ4StorageN) {
        throw WeightStoreError(
            "Q4 output dimension is incompatible with StorageN=256");
    }
}

namespace {

// The header weightFileHeader writes, which the first section follows.
constexpr uint64_t kHeaderBytes = std::tuple_size_v<decltype(weightFileHeader({}, 0, 0))>;

uint64_t alignPacked(uint64_t value) {
    static_cast<void>(
        checkedAdd<WeightStoreError>(value, kWeightFileAlignment - 1, "packed file alignment"));
    return alignWeightOffset(value);
}

} // namespace

struct WeightFile::Impl {
    metal::MetalBackend *backend = nullptr;
    metal::MetalBuffer base;
    WeightFileRecord record;
    uint64_t offset = kHeaderBytes;
};

namespace {
void checkWeightHeader(const uint8_t *header, uint64_t bytes, std::string_view expectedMagic,
                       uint32_t expectedLayer, uint32_t expectedType,
                       const std::string &what) {
    const auto expected = weightFileHeader(expectedMagic, expectedLayer, expectedType);
    if (bytes < expected.size() || bytes % kWeightFileAlignment) {
        throw WeightStoreError("packed file size is not 16 KiB-aligned: " + what);
    }
    if (std::memcmp(header, expected.data(), expected.size()) != 0) {
        throw WeightStoreError("packed file header mismatch: " + what);
    }
}
} // namespace

WeightFile::WeightFile(metal::MetalBackend &backend, metal::MetalBuffer image,
                       std::string relativePath, std::string_view expectedMagic,
                       uint32_t expectedLayer, uint32_t expectedType,
                       std::string contentIdentity)
    : impl_(std::make_unique<Impl>()) {
    const auto *header = static_cast<const uint8_t *>(image.contents());
    if (!header) throw WeightStoreError("weight image is not host visible: " + relativePath);
    checkWeightHeader(header, image.sizeBytes(), expectedMagic, expectedLayer, expectedType, relativePath);
    impl_->backend = &backend;
    impl_->record = {std::move(relativePath), std::string(expectedMagic), expectedLayer, expectedType,
                     image.sizeBytes(), std::move(contentIdentity)};
    impl_->base = std::move(image);
}

WeightFile::WeightFile(WeightFile &&) noexcept = default;
WeightFile &WeightFile::operator=(WeightFile &&) noexcept = default;

WeightFile::~WeightFile() = default;

metal::MetalBuffer WeightFile::section(uint64_t bytes,
                                       std::string_view label) {
    if (!bytes) throw WeightStoreError("packed section must not be empty");
    uint64_t start = alignPacked(impl_->offset);
    uint64_t end = checkedAdd<WeightStoreError>(start, bytes, "packed section end");
    if (start % kWeightFileAlignment || end > impl_->base.sizeBytes()) {
        throw WeightStoreError(
            "packed file is truncated at section " + std::string(label));
    }
    impl_->offset = end;
    return impl_->backend->view(impl_->base, start, bytes);
}

std::vector<metal::MetalBuffer> WeightFile::split(std::initializer_list<uint64_t> parts,
                                                  std::string_view label) {
    uint64_t bytes = 0;
    for (uint64_t part : parts) bytes = checkedAdd<WeightStoreError>(bytes, part, "packed section size");
    const metal::MetalBuffer whole = section(bytes, label);
    std::vector<metal::MetalBuffer> views;
    uint64_t offset = 0;
    for (uint64_t part : parts) {
        views.push_back(impl_->backend->view(whole, offset, part));
        offset += part;
    }
    return views;
}

void WeightFile::finish() {
    uint64_t consumed = alignPacked(impl_->offset);
    if (consumed != impl_->base.sizeBytes()) {
        throw WeightStoreError(
            "packed file has unconsumed or missing bytes: " +
            impl_->record.relativePath);
    }
}

const WeightFileRecord &WeightFile::record() const noexcept {
    return impl_->record;
}

metal::MetalBackend &WeightFile::backend() const noexcept { return *impl_->backend; }

ops::Projection readAffineProjection(WeightFile &file, uint32_t outputSize, uint32_t inputSize,
                                     std::string_view label) {
    validateQ4Layout(outputSize, inputSize);
    const uint64_t elements = q4Elements(outputSize, inputSize);
    const std::vector<metal::MetalBuffer> planes =
        file.split({elements / 2, elements / 32, elements / 32}, label);
    return {outputSize, inputSize, ops::AffineWeights{planes[0], planes[1], planes[2]}};
}

// The same three affine planes as an AF4P segment: plane0 the 64-group
// codes, plane1 the biases, meta the scales (the staged tile's af4p format
// reads its own order — kernels/common/quant_formats.h FmtAF4P).
// af4p's 64-group-ordered linear-nibble plane0, repacked to af4's
// per-group slot order, its scales and biases interleaved into af4's
// packed meta (gguf-projection-benchmark: af4 runs ~15-20% faster than
// af4p at every lane on the M5 Pro).
ops::Projection readAf4pProjection(WeightFile &file, uint32_t outputSize, uint32_t inputSize,
                                   std::string_view label) {
    validateQ4Layout(outputSize, inputSize);
    const uint64_t elements = q4Elements(outputSize, inputSize);
    const std::vector<metal::MetalBuffer> planes =
        file.split({elements / 2, elements / 32, elements / 32}, label);
    const char *af4only = std::getenv("RICHENGINE_AF4_ONLY");
    if (af4only && label.find(af4only) == std::string_view::npos)
        return {outputSize, inputSize,
                ops::BlockWeights{{ops::QuantizedSegment::planes(GGUF_FMT_AF4P, outputSize, inputSize,
                                                                 planes[0], planes[2], planes[1])}}};
    // af4 is the default: RICHENGINE_AF4_OFF restores af4p outright,
    // RICHENGINE_AF4_SKIP keeps a named projection on af4p.
    const char *af4skip = std::getenv("RICHENGINE_AF4_SKIP");
    if (std::getenv("RICHENGINE_AF4_OFF") ||
        (af4skip && label.find(af4skip) != std::string_view::npos)) {
        ops::QuantizedSegment segment =
            ops::QuantizedSegment::planes(GGUF_FMT_AF4P, outputSize, inputSize, planes[0], planes[2], planes[1]);
        return {outputSize, inputSize, ops::BlockWeights{{std::move(segment)}}};
    }
    // Repack af4p -> af4: codes [tile][64-group][row][32B] -> af4's
    // [tile][group][row][16B] deinterleaved to slot order; meta the scales
    // and biases interleaved {scale, bias} per 64-group unit. Both planes are
    // QUANT_TILE_ROWS-tiled — quant_tile_index maps (row, unit) for each.
    metal::MetalBackend &backend = file.backend();
    const uint32_t groups = inputSize / 32, units = groups / 2;
    metal::MetalBuffer codes = backend.allocateBuffer(elements / 2, metal::BufferStorage::Shared, label);
    metal::MetalBuffer meta = backend.allocateBuffer(elements / 16, metal::BufferStorage::Shared, label);
    auto *dst = static_cast<uint8_t *>(codes.contents());
    auto *md = static_cast<uint8_t *>(meta.contents());
    const auto *cs = static_cast<const uint8_t *>(planes[0].contents());
    const auto *sc = static_cast<const uint8_t *>(planes[1].contents());
    const auto *bs = static_cast<const uint8_t *>(planes[2].contents());
    for (uint64_t i = 0; i < uint64_t{outputSize} * units; ++i)
        reinterpret_cast<uint32_t *>(md)[i] =
            uint32_t(reinterpret_cast<const uint16_t *>(sc)[i]) |
            (uint32_t(reinterpret_cast<const uint16_t *>(bs)[i]) << 16);
    if (const char *dumpfile = std::getenv("RICHENGINE_AF4_PLANES")) {
        if (std::string_view{label} == "mlp-down") {
            FILE *fo = fopen(dumpfile, "wb");
            if (fo) {
                fwrite(cs, 1, elements / 2, fo);
                fwrite(sc, 1, elements / 32, fo);
                fwrite(bs, 1, elements / 32, fo);
                fclose(fo);
            }
        }
    }
    if (std::getenv("RICHENGINE_AF4_DUMP")) {
        fprintf(stderr, "af4 repack %s: out=%u in=%u elements=%llu\n", std::string(label).c_str(),
                outputSize, inputSize, (unsigned long long)elements);
        fprintf(stderr, "  plane0[0..31]:");
        for (int i = 0; i < 32; ++i) fprintf(stderr, " %02x", cs[i]);
        fprintf(stderr, "\n  scales[0..15]:");
        for (int i = 0; i < 16; ++i) fprintf(stderr, " %02x", sc[i]);
        fprintf(stderr, "\n  biases[0..15]:");
        for (int i = 0; i < 16; ++i) fprintf(stderr, " %02x", bs[i]);
        fprintf(stderr, "\n");
    }
    for (uint64_t n = 0; n < outputSize; ++n)
        for (uint64_t u = 0; u < units; ++u) {
            const uint8_t *src = cs + quant_tile_index(n, u, units) * 32;
            for (uint64_t half = 0; half < 2; ++half) {
                uint8_t *out = dst + quant_tile_index(n, 2 * u + half, groups) * 16;
                const uint8_t *in = src + half * 16;
                for (uint64_t c = 0; c < 4; ++c) {
                    const uint32_t a = reinterpret_cast<const uint16_t *>(in)[c];
                    const uint32_t b = reinterpret_cast<const uint16_t *>(in)[4 + c];
                    const uint32_t ae = (a & 0xFu) | (((a >> 8) & 0xFu) << 4);
                    const uint32_t be = (b & 0xFu) | (((b >> 8) & 0xFu) << 4);
                    const uint32_t ao = ((a >> 4) & 0xFu) | (((a >> 12) & 0xFu) << 4);
                    const uint32_t bo = ((b >> 4) & 0xFu) | (((b >> 12) & 0xFu) << 4);
                    reinterpret_cast<uint32_t *>(out)[c] = ae | (be << 8) | (ao << 16) | (bo << 24);
                }
            }
        }
    if (std::getenv("RICHENGINE_AF4_PLANES") && std::string_view{label} == "mlp-down") {
        FILE *fa = fopen("/tmp/af4mlp.bin", "wb");
        if (fa) { fwrite(dst, 1, elements / 2, fa); fwrite(md, 1, elements / 16, fa); fclose(fa); }
    }
    if (std::getenv("RICHENGINE_AF4_RAW")) {
        return {outputSize, inputSize,
                ops::BlockWeights{{ops::QuantizedSegment::planes(GGUF_FMT_AF4, outputSize, inputSize,
                                                                 planes[0], {}, meta)}}};
    }
    ops::QuantizedSegment segment =
        ops::QuantizedSegment::planes(GGUF_FMT_AF4, outputSize, inputSize, std::move(codes), {}, std::move(meta));
    return {outputSize, inputSize, ops::BlockWeights{{std::move(segment)}}};
}

ops::EmbeddingWeights readAffineEmbedding(WeightFile &file,
                                             uint32_t outputSize,
                                             uint32_t inputSize,
                                             std::string_view label,
                                             bool tiled) {
    const uint64_t elements = q4Elements(outputSize, inputSize);
    const std::string prefix(label);
    // Braced initializers read the sections in file order.
    return {outputSize, inputSize,
            ops::AffineWeights{
                file.section(elements / 2, prefix + "-weights"),
                file.section(elements / 32, prefix + "-scales"),
                file.section(elements / 32, prefix + "-biases"),
            },
            tiled};
}

ops::NormWeights readNorm(WeightFile &file, uint32_t width, bool float32,
                          std::string_view label) {
    ops::NormWeights norm{{}, float32};
    norm.buffer = file.section(norm.bytes(width), label);
    return norm;
}

namespace {
GgufTensorDescriptor readGgufDescriptor(WeightFile &file, std::string_view label,
                                        bool tiled = true) {
    metal::MetalBuffer section = file.section(sizeof(GgufTensorDescriptor), std::string(label) + "-desc");
    const uint8_t *bytes = static_cast<const uint8_t *>(section.contents());
    if (!bytes) throw WeightStoreError("GGUF descriptor is not host visible");
    GgufTensorDescriptor d;
    std::memcpy(&d, bytes, sizeof d);
    // Float tensors are rows as stored; quantized ones fill whole tiles. An
    // affine-tile section (ggml::kAffineQ4) takes the StorageN/64-group
    // alignment its planes need instead of the block one. A native-rows
    // section (the token table) stores flat rows, so a vocabulary that is
    // not a tile multiple is fine.
    const bool affine = d.type == ggml::kAffineQ4;
    if (!d.outputSize || !d.inputSize ||
        (tiled && d.type != ggml::kF32 && !affine &&
         (d.outputSize % QUANT_TILE_ROWS || d.inputSize % kGgufBlockColumns)) ||
        (tiled && affine && (d.outputSize % kQ4StorageN || d.inputSize % kQ4GroupElements)))
        throw WeightStoreError("GGUF tensor shape is not tile aligned: " + std::string(label));
    return d;
}

// The segment a GGUF descriptor's payload holds: an F32 tensor's rows, or
// the plane0/plane1/meta planes of its declared format.
ops::QuantizedSegment segmentFromDescriptor(WeightFile &file,
                                            const GgufTensorDescriptor &d,
                                            std::string_view label) {
    if (d.type == ggml::kF32) {
        if (d.p0 || d.p1 || d.metaBytes || d.metaGroups || d.plane1Bytes || d.metaTotalBytes ||
            d.plane0Bytes != uint64_t{d.outputSize} * d.inputSize * sizeof(float))
            throw WeightStoreError("GGUF float section sizes are inconsistent: " + std::string(label));
        return ops::QuantizedSegment::floats(d.outputSize, d.inputSize,
                                             file.section(d.plane0Bytes, std::string(label) + "-floats"));
    }
    const uint32_t format = gguf_format_of(d.type);
    if (format == GGUF_FMT_COUNT)
        throw WeightStoreError("unsupported GGUF tensor type " + std::to_string(d.type));
    const QuantFormat &layout = kQuantFormats[format];
    const GgufPlaneBytes planes = ggufPlaneBytes(layout, d.outputSize, d.inputSize);
    if (d.p0 != layout.plane0_bytes || d.p1 != layout.plane1_bytes ||
        d.metaBytes != layout.meta_bytes || d.metaGroups != layout.meta_groups ||
        d.plane0Bytes != planes.plane0 || d.plane1Bytes != planes.plane1 || d.metaTotalBytes != planes.meta)
        throw WeightStoreError("GGUF section sizes are inconsistent: " + std::string(label));
    metal::MetalBuffer plane0 = file.section(d.plane0Bytes, std::string(label) + "-plane0");
    metal::MetalBuffer plane1 =
        d.plane1Bytes ? file.section(d.plane1Bytes, std::string(label) + "-plane1") : metal::MetalBuffer{};
    metal::MetalBuffer meta = file.section(d.metaTotalBytes, std::string(label) + "-meta");
    return ops::QuantizedSegment::planes(format, d.outputSize, d.inputSize, std::move(plane0),
                                         std::move(plane1), std::move(meta));
}
} // namespace

ops::QuantizedSegment readQuantizedSegment(WeightFile &file, std::string_view label) {
    return segmentFromDescriptor(file, readGgufDescriptor(file, label), label);
}

ops::Projection readBlockProjection(WeightFile &file, uint32_t outputSize, uint32_t inputSize,
                                    std::string_view label) {
    ops::QuantizedSegment segment = readQuantizedSegment(file, label);
    if (segment.outputSize != outputSize || segment.inputSize != inputSize)
        throw WeightStoreError("GGUF tensor does not match the layout: " + std::string(label));
    return {outputSize, inputSize, ops::BlockWeights{{std::move(segment)}}};
}

ops::Projection readProjection(WeightFile &file, uint32_t outputSize, uint32_t inputSize,
                               std::string_view label) {
    return file.record().type & kBlockWeightsImageType
               ? readBlockProjection(file, outputSize, inputSize, label)
               : readAffineProjection(file, outputSize, inputSize, label);
}

ops::EmbeddingWeights readBlockEmbedding(WeightFile &file, uint32_t outputSize, uint32_t inputSize,
                                         std::string_view label) {
    const GgufTensorDescriptor d = readGgufDescriptor(file, label, /*tiled=*/false);
    if (d.outputSize != outputSize || d.inputSize != inputSize)
        throw WeightStoreError("GGUF embedding does not match the layout: " + std::string(label));
    const uint32_t format = gguf_format_of(d.type);
    if (format == GGUF_FMT_COUNT ||
        d.plane0Bytes != d.outputSize * ggufRowBytes(kQuantFormats[format], d.inputSize))
        throw WeightStoreError("GGUF embedding rows are not native GGUF blocks: " + std::string(label));
    return {outputSize, inputSize,
            ops::NativeRows(file.section(d.plane0Bytes, std::string(label) + "-native"), format)};
}

ops::EmbeddingWeights readOqEmbedding(WeightFile &file, uint32_t outputSize, uint32_t inputSize,
                                      std::string_view label) {
    const GgufTensorDescriptor d = readGgufDescriptor(file, label, /*tiled=*/false);
    if (d.outputSize != outputSize || d.inputSize != inputSize)
        throw WeightStoreError("oQ embedding does not match the layout: " + std::string(label));
    if (d.type == ggml::kF32) {
        if (d.plane0Bytes != uint64_t{d.outputSize} * d.inputSize * sizeof(float))
            throw WeightStoreError("oQ float embedding rows are inconsistent: " + std::string(label));
        return {outputSize, inputSize,
                ops::NativeRows(file.section(d.plane0Bytes, std::string(label) + "-native"),
                                ops::QuantizedSegment::kFloat32)};
    }
    const uint32_t format = gguf_format_of(d.type);
    if (format == GGUF_FMT_COUNT ||
        d.plane0Bytes != d.outputSize * ggufRowBytes(kQuantFormats[format], d.inputSize))
        throw WeightStoreError("oQ embedding rows are not native affine blocks: " + std::string(label));
    return {outputSize, inputSize,
            ops::NativeRows(file.section(d.plane0Bytes, std::string(label) + "-native"), format)};
}

ops::Projection readOqFused(WeightFile &file, uint32_t outputSize, uint32_t inputSize,
                          std::string_view label) {
    ops::BlockWeights weights;
    uint32_t covered = 0;
    while (covered < outputSize) {
        const GgufTensorDescriptor d = readGgufDescriptor(file, label);
        if (d.type == ggml::kAffineQ4) {
            if (covered || d.outputSize != outputSize || d.inputSize != inputSize)
                throw WeightStoreError("oQ affine sections lead a whole projection: " +
                                       std::string(label));
            validateQ4Layout(outputSize, inputSize);
            const uint64_t elements = q4Elements(outputSize, inputSize);
            const std::vector<metal::MetalBuffer> planes =
                file.split({elements / 2, elements / 32, elements / 32}, label);
            return {outputSize, inputSize,
                    ops::AffineWeights{planes[0], planes[1], planes[2]}};
        }
        ops::QuantizedSegment segment = segmentFromDescriptor(file, d, label);
        if (segment.outputSize > outputSize - covered)
            throw WeightStoreError("oQ segments overshoot the projection: " + std::string(label));
        segment.columnOffset = covered;
        covered += segment.outputSize;
        weights.segments.push_back(std::move(segment));
    }
    return {outputSize, inputSize, std::move(weights)};
}

// One descriptor-prefixed projection of an oQ image: the affine tiles of a
// ggml::kAffineQ4 section, or the single block segment any other covers.
ops::Projection readOqProjection(WeightFile &file, uint32_t outputSize, uint32_t inputSize,
                                 std::string_view label) {
    const GgufTensorDescriptor d = readGgufDescriptor(file, label);
    if (d.type == ggml::kAffineQ4) {
        if (d.outputSize != outputSize || d.inputSize != inputSize)
            throw WeightStoreError("oQ affine tensor does not match the layout: " +
                                   std::string(label));
        validateQ4Layout(outputSize, inputSize);
        const uint64_t elements = q4Elements(outputSize, inputSize);
        const std::vector<metal::MetalBuffer> planes =
            file.split({elements / 2, elements / 32, elements / 32}, label);
        return {outputSize, inputSize,
                ops::AffineWeights{planes[0], planes[1], planes[2]}};
    }
    const ops::QuantizedSegment segment = segmentFromDescriptor(file, d, label);
    if (segment.outputSize != outputSize || segment.inputSize != inputSize)
        throw WeightStoreError("oQ tensor does not match the layout: " + std::string(label));
    return {outputSize, inputSize, ops::BlockWeights{{std::move(segment)}}};
}

ops::Q8Projection readAffineQ8Projection(WeightFile &file, uint32_t outputSize, uint32_t inputSize,
                                         std::string_view label) {
    validateQ4Layout(outputSize, inputSize);
    const uint64_t elements = q4Elements(outputSize, inputSize);
    const std::vector<metal::MetalBuffer> planes = file.split({elements, elements / 32, elements / 32}, label);
    return {{planes[0], planes[1], planes[2]}, outputSize, inputSize};
}

ops::ExpertProjection
readAffineExpertProjection(WeightFile &file, uint32_t experts,
                           uint32_t outputSize, uint32_t inputSize,
                           std::string_view label) {
    if (!experts)
        throw WeightStoreError("expert projection requires experts");
    validateQ4Layout(outputSize, inputSize);
    const uint64_t stride = q4PackedBytes(outputSize, inputSize);
    return {
        file.section(checkedMultiply<WeightStoreError>(experts, stride,
                                                       "expert Q4 slab bytes"),
                     label),
        experts,
        outputSize,
        inputSize,
        stride,
    };
}

std::string weightManifestFingerprint(
    std::span<const WeightFileRecord> records) {
    std::vector<WeightFileRecord> sorted(records.begin(), records.end());
    std::sort(sorted.begin(), sorted.end(),
              [](const WeightFileRecord &left,
                 const WeightFileRecord &right) {
                  return left.relativePath < right.relativePath;
              });
    std::ostringstream canonical;
    canonical << "richengine-packed-manifest-v1\n";
    for (const WeightFileRecord &record : sorted) {
        canonical << record.relativePath << '\t' << record.declaredBytes
                  << '\t' << record.magic << '\t' << record.layer << '\t'
                  << record.type;
        if (!record.contentIdentity.empty()) canonical << '\t' << record.contentIdentity;
        canonical << '\n';
    }
    return weightDigest(canonical.str());
}

std::string weightDigest(std::span<const uint8_t> bytes) {
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    // CommonCrypto takes 32-bit lengths.
    constexpr size_t kPieceBytes = size_t(1) << 30;
    for (size_t at = 0; at < bytes.size(); at += kPieceBytes)
        CC_SHA256_Update(&context, bytes.data() + at,
                         static_cast<CC_LONG>(std::min(kPieceBytes, bytes.size() - at)));
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256_Final(digest, &context);
    return digestHex(digest);
}

std::string weightDigest(std::string_view text) {
    return weightDigest({reinterpret_cast<const uint8_t *>(text.data()), text.size()});
}

} // namespace richengine::model
