#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "Fp8PageFormatReference.hpp"
#include "Q8PageFormatReference.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace richengine::kv;

namespace {

id<MTLBuffer> buffer(id<MTLDevice> device, uint64_t bytes) {
    id<MTLBuffer> result = [device
        newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!result) throw std::runtime_error("Metal buffer allocation failed");
    return result;
}

id<MTLComputePipelineState> pipeline(id<MTLDevice> device,
                                     id<MTLLibrary> library,
                                     const char *name) {
    NSString *functionName = [NSString stringWithUTF8String:name];
    id<MTLFunction> function = [library newFunctionWithName:functionName];
    if (!function) throw std::runtime_error(std::string("missing kernel: ") + name);
    NSError *error = nil;
    id<MTLComputePipelineState> result =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (!result) {
        throw std::runtime_error(error.localizedDescription.UTF8String);
    }
    return result;
}

void finish(id<MTLCommandBuffer> command) {
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
        std::string message = "Metal command failed";
        if (command.error) {
            message += ": ";
            message += command.error.localizedDescription.UTF8String;
        }
        throw std::runtime_error(message);
    }
}

void requireEqual(const void *left, const void *right, uint64_t bytes,
                  const char *name) {
    const auto *observed = static_cast<const uint8_t *>(left);
    const auto *expected = static_cast<const uint8_t *>(right);
    for (uint64_t index = 0; index < bytes; ++index) {
        if (observed[index] != expected[index]) {
            throw std::runtime_error(
                std::string(name) + " differs from CPU reference at byte " +
                std::to_string(index) + " (Metal=" +
                std::to_string(observed[index]) + ", CPU=" +
                std::to_string(expected[index]) + ")");
        }
    }
}

void requireFloatClose(const float *left, const float *right, uint64_t count,
                       const char *name) {
    for (uint64_t index = 0; index < count; ++index) {
        float tolerance = std::max(1.0e-8f, std::abs(right[index]) * 2.0e-6f);
        if (!std::isfinite(left[index]) ||
            std::abs(left[index] - right[index]) > tolerance) {
            throw std::runtime_error(
                std::string(name) + " differs from CPU reference at " +
                std::to_string(index));
        }
    }
}

// The encoder agrees with the saturating RNE pack on a dense sweep across
// normals, subnormals, the saturation boundary and zero before the GPU side
// is trusted bit for bit.
void checkEncoder() {
    for (int code = 0; code < 256; ++code) {
        const uint8_t byte = static_cast<uint8_t>(code);
        const float decoded = fp8E4m3ToFloat(byte);
        if (std::isnan(decoded)) continue;
        if (floatToFp8E4m3(decoded) != byte)
            throw std::runtime_error("e4m3 round trip failed at " +
                                     std::to_string(code));
    }
    const float samples[] = {0.0f, 1.0f, -1.0f, 448.0f, -448.0f, 449.0f,
                             464.0f, 465.0f, 1.0e6f, 0x1p-9f, 0x1p-10f,
                             0x1p-11f, 4096.0f, -0x1p-6f};
    for (float sample : samples) {
        const float back = fp8E4m3ToFloat(floatToFp8E4m3(sample));
        if (!std::isfinite(back) || std::abs(back) > kFp8E4m3Maximum)
            throw std::runtime_error("e4m3 encode escaped the format");
    }
}

void run(const char *libraryPath) {
    checkEncoder();
    constexpr uint32_t validTokens = 17;
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) throw std::runtime_error("Metal device unavailable");
    NSError *error = nil;
    NSURL *url = [NSURL fileURLWithPath:
        [NSString stringWithUTF8String:libraryPath]];
    id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];
    if (!library) {
        throw std::runtime_error(error.localizedDescription.UTF8String);
    }
    auto quantize = pipeline(
        device, library, "richengine_fp8_quantize_kv_page");
    auto dequantize = pipeline(
        device, library, "richengine_fp8_dequantize_kv_page");
    auto gather = pipeline(
        device, library, "richengine_fp8_gather_logical_kv_page");

    std::vector<float> logicalKeys(
        uint64_t{validTokens} * kFp8KvHeads * kFp8HeadDimension);
    std::vector<float> logicalValues(logicalKeys.size());
    std::vector<BFloat16Bits> physicalKeys(kFp8ElementsPerLayerPage);
    std::vector<BFloat16Bits> physicalValues(kFp8ElementsPerLayerPage);
    for (uint32_t token = 0; token < validTokens; ++token) {
        for (uint32_t head = 0; head < kFp8KvHeads; ++head) {
            for (uint32_t dimension = 0; dimension < kFp8HeadDimension;
                 ++dimension) {
                uint64_t logical = fp8LogicalIndex(token, head, dimension);
                float key = float(int((token * 17 + head * 23 + dimension * 7) %
                                      1009) - 504) / 113.0f;
                float value = float(int((token * 29 + head * 13 + dimension * 11) %
                                        1013) - 506) / 97.0f;
                BFloat16Bits keyBits = floatToBFloat16(key);
                BFloat16Bits valueBits = floatToBFloat16(value);
                logicalKeys[logical] = bfloat16ToFloat(keyBits);
                logicalValues[logical] = bfloat16ToFloat(valueBits);
                physicalKeys[richengine_kv_key_element(head, token, dimension)] =
                    keyBits;
                physicalValues[richengine_kv_value_element(head, token, dimension)] =
                    valueBits;
            }
        }
    }

    auto reference = std::make_unique<Fp8LayerPage>();
    quantizeFp8LayerPage(logicalKeys, logicalValues, validTokens, *reference);

    id<MTLBuffer> sourceKeys = buffer(
        device, kFp8ElementsPerLayerPage * sizeof(BFloat16Bits));
    id<MTLBuffer> sourceValues = buffer(
        device, kFp8ElementsPerLayerPage * sizeof(BFloat16Bits));
    id<MTLBuffer> fp8Keys = buffer(device, kFp8DataBytesPerLayerPage);
    id<MTLBuffer> keyScales = buffer(device, kFp8ScaleBytesPerLayerPage);
    id<MTLBuffer> fp8Values = buffer(device, kFp8DataBytesPerLayerPage);
    id<MTLBuffer> valueScales = buffer(device, kFp8ScaleBytesPerLayerPage);
    std::memcpy(sourceKeys.contents, physicalKeys.data(), sourceKeys.length);
    std::memcpy(sourceValues.contents, physicalValues.data(), sourceValues.length);

    Fp8KVMetalPageParams params{0, 0, validTokens, 0};
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> quantizeCommand = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [quantizeCommand computeCommandEncoder];
    [encoder setComputePipelineState:quantize];
    [encoder setBuffer:sourceKeys offset:0 atIndex:0];
    [encoder setBuffer:sourceValues offset:0 atIndex:1];
    [encoder setBuffer:fp8Keys offset:0 atIndex:2];
    [encoder setBuffer:keyScales offset:0 atIndex:3];
    [encoder setBuffer:fp8Values offset:0 atIndex:4];
    [encoder setBuffer:valueScales offset:0 atIndex:5];
    [encoder setBytes:&params length:sizeof(params) atIndex:6];
    [encoder dispatchThreadgroups:
                 MTLSizeMake(2 * kFp8ScalesPerTensorLayerPage, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(kFp8HeadDimension, 1, 1)];
    [encoder endEncoding];
    finish(quantizeCommand);

    requireEqual(fp8Keys.contents, reference->keys.data(), fp8Keys.length,
                 "FP8 keys");
    requireFloatClose(static_cast<const float *>(keyScales.contents),
                      reference->keyScales.data(),
                      kFp8ScalesPerTensorLayerPage, "FP8 key scales");
    requireFloatClose(static_cast<const float *>(valueScales.contents),
                      reference->valueScales.data(),
                      kFp8ScalesPerTensorLayerPage, "FP8 value scales");
    requireEqual(fp8Values.contents, reference->values.data(),
                 fp8Values.length, "FP8 values");

    id<MTLBuffer> decodedKeys = buffer(device, sourceKeys.length);
    id<MTLBuffer> decodedValues = buffer(device, sourceValues.length);
    id<MTLBuffer> logicalGatherKeys = buffer(device, sourceKeys.length);
    id<MTLBuffer> logicalGatherValues = buffer(device, sourceValues.length);
    id<MTLCommandBuffer> decodeCommand = [queue commandBuffer];
    encoder = [decodeCommand computeCommandEncoder];
    [encoder setComputePipelineState:dequantize];
    [encoder setBuffer:fp8Keys offset:0 atIndex:0];
    [encoder setBuffer:keyScales offset:0 atIndex:1];
    [encoder setBuffer:fp8Values offset:0 atIndex:2];
    [encoder setBuffer:valueScales offset:0 atIndex:3];
    [encoder setBuffer:decodedKeys offset:0 atIndex:4];
    [encoder setBuffer:decodedValues offset:0 atIndex:5];
    [encoder setBytes:&params length:sizeof(params) atIndex:6];
    [encoder dispatchThreads:MTLSizeMake(2 * kFp8ElementsPerLayerPage, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder setComputePipelineState:gather];
    [encoder setBuffer:fp8Keys offset:0 atIndex:0];
    [encoder setBuffer:keyScales offset:0 atIndex:1];
    [encoder setBuffer:fp8Values offset:0 atIndex:2];
    [encoder setBuffer:valueScales offset:0 atIndex:3];
    [encoder setBuffer:logicalGatherKeys offset:0 atIndex:4];
    [encoder setBuffer:logicalGatherValues offset:0 atIndex:5];
    [encoder setBytes:&params length:sizeof(params) atIndex:6];
    [encoder dispatchThreads:MTLSizeMake(kFp8ElementsPerLayerPage, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    finish(decodeCommand);

    std::vector<float> decodedLogicalKeys(kFp8ElementsPerLayerPage);
    std::vector<float> decodedLogicalValues(kFp8ElementsPerLayerPage);
    dequantizeFp8LayerPage(*reference, validTokens,
                           decodedLogicalKeys, decodedLogicalValues);
    std::vector<BFloat16Bits> expectedLogicalKeys(kFp8ElementsPerLayerPage);
    std::vector<BFloat16Bits> expectedLogicalValues(kFp8ElementsPerLayerPage);
    std::vector<BFloat16Bits> expectedPhysicalKeys(kFp8ElementsPerLayerPage);
    std::vector<BFloat16Bits> expectedPhysicalValues(kFp8ElementsPerLayerPage);
    for (uint32_t token = 0; token < kPageTokens; ++token) {
        for (uint32_t head = 0; head < kFp8KvHeads; ++head) {
            for (uint32_t dimension = 0; dimension < kFp8HeadDimension;
                 ++dimension) {
                uint64_t logical = fp8LogicalIndex(token, head, dimension);
                BFloat16Bits key = floatToBFloat16(decodedLogicalKeys[logical]);
                BFloat16Bits value =
                    floatToBFloat16(decodedLogicalValues[logical]);
                expectedLogicalKeys[logical] = key;
                expectedLogicalValues[logical] = value;
                expectedPhysicalKeys[
                    richengine_kv_key_element(head, token, dimension)] = key;
                expectedPhysicalValues[
                    richengine_kv_value_element(head, token, dimension)] = value;
            }
        }
    }
    requireEqual(decodedKeys.contents, expectedPhysicalKeys.data(),
                 decodedKeys.length, "dequantized physical keys");
    requireEqual(decodedValues.contents, expectedPhysicalValues.data(),
                 decodedValues.length, "dequantized physical values");
    requireEqual(logicalGatherKeys.contents, expectedLogicalKeys.data(),
                 logicalGatherKeys.length, "gathered logical keys");
    requireEqual(logicalGatherValues.contents, expectedLogicalValues.data(),
                 logicalGatherValues.length, "gathered logical values");

    // A uniform tensor checks exact CPU/Metal agreement for the common
    // per-token/head scale and the format's maximum code path.
    std::fill(logicalValues.begin(), logicalValues.end(), 1.0f);
    std::fill(physicalValues.begin(), physicalValues.end(), BFloat16Bits{0});
    for (uint32_t token = 0; token < validTokens; ++token) {
        for (uint32_t head = 0; head < kFp8KvHeads; ++head) {
            for (uint32_t dimension = 0; dimension < kFp8HeadDimension;
                 ++dimension) {
                physicalValues[richengine_kv_value_element(head, token, dimension)] =
                    floatToBFloat16(1.0f);
            }
        }
    }
    quantizeFp8LayerPage(logicalKeys, logicalValues, validTokens, *reference);
    std::memcpy(sourceValues.contents, physicalValues.data(), sourceValues.length);
    quantizeCommand = [queue commandBuffer];
    encoder = [quantizeCommand computeCommandEncoder];
    [encoder setComputePipelineState:quantize];
    [encoder setBuffer:sourceKeys offset:0 atIndex:0];
    [encoder setBuffer:sourceValues offset:0 atIndex:1];
    [encoder setBuffer:fp8Keys offset:0 atIndex:2];
    [encoder setBuffer:keyScales offset:0 atIndex:3];
    [encoder setBuffer:fp8Values offset:0 atIndex:4];
    [encoder setBuffer:valueScales offset:0 atIndex:5];
    [encoder setBytes:&params length:sizeof(params) atIndex:6];
    [encoder dispatchThreadgroups:
                 MTLSizeMake(2 * kFp8ScalesPerTensorLayerPage, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(kFp8HeadDimension, 1, 1)];
    [encoder endEncoding];
    finish(quantizeCommand);
    requireEqual(fp8Values.contents, reference->values.data(),
                 fp8Values.length, "same-sign maximum FP8 values");
    requireFloatClose(static_cast<const float *>(valueScales.contents),
                      reference->valueScales.data(),
                      kFp8ScalesPerTensorLayerPage,
                      "same-sign maximum FP8 value scales");
}

}  // namespace

int main(int argc, char **argv) {
    @autoreleasepool {
        try {
            if (argc != 2) {
                std::cerr << "usage: fp8_paged_kv_metal_test METALLIB\n";
                return 2;
            }
            run(argv[1]);
            std::cout << "fp8_paged_kv_metal_test: ok\n";
        } catch (const std::exception &error) {
            std::cerr << "fp8_paged_kv_metal_test: " << error.what() << "\n";
            return 1;
        }
    }
    return 0;
}
