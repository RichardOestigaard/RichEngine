#import "AnePredictor.hpp"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <cstring>
#include <dispatch/dispatch.h>
#include <utility>

namespace splash::model {

struct AnePredictor::Impl {
  MLModel *model = nil;
  dispatch_queue_t queue = nullptr;
  // Drains in-flight work before the queue (ARC-managed) can be released.
  ~Impl() {
    if (queue) {
      dispatch_sync(queue, ^{
      });
    }
  }
};

static MLMultiArrayDataType mlDType(AneDType type) {
  switch (type) {
  case AneDType::Float16:
    return MLMultiArrayDataTypeFloat16;
  case AneDType::Int32:
    return MLMultiArrayDataTypeInt32;
  case AneDType::Float32:
    return MLMultiArrayDataTypeFloat32;
  }
  return MLMultiArrayDataTypeFloat32;
}

static size_t aneBytes(AneDType type) {
  switch (type) {
  case AneDType::Float16:
    return 2;
  case AneDType::Int32:
  case AneDType::Float32:
    return 4;
  }
  return 4;
}

std::unique_ptr<AnePredictor>
AnePredictor::load(std::string_view path, std::string &error) {
  @autoreleasepool {
    NSString *nsPath = [NSString stringWithUTF8String:std::string(path).c_str()];
    NSURL *url = [NSURL fileURLWithPath:nsPath];
    NSError *nsError = nil;

    NSURL *compiled = url;
    NSString *extension = url.pathExtension;
    if (![extension isEqualToString:@"mlmodelc"]) {
      // .mlpackage and .mlmodel must be compiled before loading. CoreML
      // caches compiled bundles under a system location; repeat loads of an
      // unchanged package skip the compiler.
      compiled = [MLModel compileModelAtURL:url error:&nsError];
      if (!compiled) {
        error = "CoreML compilation failed: " +
                std::string(nsError.localizedDescription.UTF8String ?: "?");
        return nullptr;
      }
    }

    MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
    config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
    MLModel *model = [MLModel modelWithContentsOfURL:compiled
                                       configuration:config
                                               error:&nsError];
    if (!model) {
      error = "CoreML load failed: " +
              std::string(nsError.localizedDescription.UTF8String ?: "?");
      return nullptr;
    }

    Impl *impl = new Impl;
    impl->model = model;
    impl->queue =
        dispatch_queue_create("splash.ane.predictor", DISPATCH_QUEUE_SERIAL);
    return std::unique_ptr<AnePredictor>(new AnePredictor(impl));
  }
}

AnePredictor::AnePredictor(Impl *impl) : impl_(impl) {}

AnePredictor::~AnePredictor() { delete impl_; }

void AnePredictor::submit(std::function<void()> job) {
  std::function<void()> moved = std::move(job);
  dispatch_async(impl_->queue, ^{
    moved();
  });
}

bool AnePredictor::predict(std::span<const AneTensor> inputs,
                           std::span<AneTensor> outputs, std::string &error) {
  @autoreleasepool {
    NSError *nsError = nil;
    NSMutableDictionary<NSString *, MLFeatureValue *> *features =
        [NSMutableDictionary dictionaryWithCapacity:inputs.size()];

    for (const AneTensor &input : inputs) {
      NSMutableArray<NSNumber *> *shape =
          [NSMutableArray arrayWithCapacity:input.shape.size()];
      NSMutableArray<NSNumber *> *strides =
          [NSMutableArray arrayWithCapacity:input.shape.size()];
      int64_t stride = 1;
      for (int64_t dim = static_cast<int64_t>(input.shape.size()) - 1;
           dim >= 0; --dim) {
        [shape insertObject:@(input.shape[dim]) atIndex:0];
        [strides insertObject:@(stride) atIndex:0];
        stride *= input.shape[dim];
      }
      MLMultiArray *array =
          [[MLMultiArray alloc] initWithShape:shape
                                     dataType:mlDType(input.type)
                                        error:&nsError];
      if (!array) {
        error = "input '" + input.name + "' allocation failed: " +
                std::string(nsError.localizedDescription.UTF8String ?: "?");
        return false;
      }
      std::memcpy(array.dataPointer, input.data,
                  static_cast<size_t>(stride) * aneBytes(input.type));
      NSString *key =
          [NSString stringWithUTF8String:input.name.c_str()];
      features[key] = [MLFeatureValue featureValueWithMultiArray:array];
    }

    MLDictionaryFeatureProvider *provider = [[MLDictionaryFeatureProvider alloc]
        initWithDictionary:features
                     error:&nsError];
    if (!provider) {
      error = "feature provider failed: " +
              std::string(nsError.localizedDescription.UTF8String ?: "?");
      return false;
    }

    id<MLFeatureProvider> result =
        [impl_->model predictionFromFeatures:provider error:&nsError];
    if (!result) {
      error = "prediction failed: " +
              std::string(nsError.localizedDescription.UTF8String ?: "?");
      return false;
    }

    for (AneTensor &output : outputs) {
      NSString *key =
          [NSString stringWithUTF8String:output.name.c_str()];
      MLMultiArray *array =
          [result featureValueForName:key].multiArrayValue;
      if (!array) {
        error = "output '" + output.name + "' missing from model result";
        return false;
      }
      size_t expected = 1;
      for (int64_t dim : output.shape) {
        expected *= static_cast<size_t>(dim);
      }
      if (static_cast<size_t>(array.count) < expected) {
        error = "output '" + output.name + "' smaller than expected";
        return false;
      }
      const MLMultiArrayDataType produced = array.dataType;
      if (produced == mlDType(output.type)) {
        std::memcpy(output.data, array.dataPointer,
                    expected * aneBytes(output.type));
      } else if (output.type == AneDType::Int32 &&
                 produced == MLMultiArrayDataTypeFloat32) {
        // Legacy spec backends export integer outputs as fp32; token ids
        // survive the round trip as whole floats.
        const float *source = static_cast<const float *>(array.dataPointer);
        int32_t *target = static_cast<int32_t *>(output.data);
        for (size_t index = 0; index < expected; ++index) {
          target[index] = static_cast<int32_t>(source[index]);
        }
      } else if (output.type == AneDType::Float32 &&
                 produced == MLMultiArrayDataTypeFloat16) {
        const _Float16 *source =
            static_cast<const _Float16 *>(array.dataPointer);
        float *target = static_cast<float *>(output.data);
        for (size_t index = 0; index < expected; ++index) {
          target[index] = static_cast<float>(source[index]);
        }
      } else {
        error = "output '" + output.name + "' has an unsupported dtype";
        return false;
      }
    }
    return true;
  }
}

} // namespace splash::model
