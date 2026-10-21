// Dispatches a Metal ML package through MTL4MachineLearningCommandEncoder and
// times it. Used to test whether ANE offload of decode-row FFN work is viable.
//   ml-encoder-bench PACKAGE.mtlpackage [iters]
#include <Metal/Metal.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

id<MTLTensor> tensorFor(id<MTLDevice> device, id<MTLTensorBinding> binding) {
  MTLTensorDescriptor *desc = [MTLTensorDescriptor new];
  desc.dataType = binding.tensorDataType;
  desc.usage = MTLTensorUsageMachineLearning;
  const NSInteger rank = binding.dimensions ? binding.dimensions.rank : 0;
  std::vector<NSInteger> dims(rank);
  for (NSInteger i = 0; i < rank; ++i) dims[i] = [binding.dimensions extentAtDimensionIndex:i];
  desc.dimensions = [[MTLTensorExtents alloc] initWithRank:rank values:dims.data()];
  NSError *error = nil;
  id<MTLTensor> tensor = [device newTensorWithDescriptor:desc error:&error];
  if (!tensor)
    fprintf(stderr, "tensor for binding %s: %s\n", binding.name.UTF8String,
            error.localizedDescription.UTF8String);
  return tensor;
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc < 2) {
      std::fprintf(stderr, "usage: ml-encoder-bench PACKAGE.mtlpackage [iters]\n");
      return 2;
    }
    const int iters = argc > 2 ? std::atoi(argv[2]) : 200;
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      std::fprintf(stderr, "no Metal device\n");
      return 1;
    }
    NSError *error = nil;
    NSURL *url = [NSURL fileURLWithPath:@(argv[1])];
    id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];
    if (!library) {
      std::fprintf(stderr, "library: %s\n", error.localizedDescription.UTF8String);
      return 1;
    }
    NSArray<NSString *> *names = library.functionNames;
    std::fprintf(stderr, "functions:");
    for (NSString *name in names) std::fprintf(stderr, " %s", name.UTF8String);
    std::fprintf(stderr, "\n");

    MTL4LibraryFunctionDescriptor *fn = [MTL4LibraryFunctionDescriptor new];
    fn.name = names.count ? names[0] : @"main";
    fn.library = library;
    MTL4MachineLearningPipelineDescriptor *desc = [MTL4MachineLearningPipelineDescriptor new];
    desc.machineLearningFunctionDescriptor = fn;
    // Extents index 0 is the innermost (last) dimension.
    const NSInteger ioDims[] = {5120, 8};
    MTLTensorExtents *ioExtents = [[MTLTensorExtents alloc] initWithRank:2 values:ioDims];
    [desc setInputDimensions:ioExtents atBufferIndex:0];
    [desc setInputDimensions:ioExtents atBufferIndex:1];
    id<MTL4Compiler> compiler = [device newCompilerWithDescriptor:[MTL4CompilerDescriptor new]
                                                          error:&error];
    if (!compiler) {
      std::fprintf(stderr, "compiler: %s\n", error.localizedDescription.UTF8String);
      return 1;
    }
    id<MTL4MachineLearningPipelineState> pipeline =
        [compiler newMachineLearningPipelineStateWithDescriptor:desc error:&error];
    if (!pipeline) {
      std::fprintf(stderr, "pipeline: %s\n", error.localizedDescription.UTF8String);
      return 1;
    }
    std::fprintf(stderr, "intermediatesHeapSize %llu bindings %lu\n",
                 (unsigned long long)pipeline.intermediatesHeapSize,
                 (unsigned long)pipeline.reflection.bindings.count);

    // A tensor per binding; print their shapes to see what the network wants.
    // Empty reflections bind by position: input 0, output 1 (8x5120 fp16).
    NSMutableArray<id<MTLTensor>> *tensors = [NSMutableArray new];
    auto makeTensor = [&](NSInteger rank, const NSInteger *dims) -> id<MTLTensor> {
      MTLTensorDescriptor *desc = [MTLTensorDescriptor new];
      desc.dataType = MTLTensorDataTypeFloat16;
      desc.usage = MTLTensorUsageMachineLearning;
      desc.dimensions = [[MTLTensorExtents alloc] initWithRank:rank values:dims];
      NSError *error = nil;
      return [device newTensorWithDescriptor:desc error:&error];
    };
    if (pipeline.reflection.bindings.count == 0) {
      const NSInteger io[] = {5120, 8};
      tensors[0] = makeTensor(2, io);
      tensors[1] = makeTensor(2, io);
    }
    for (id<MTLBinding> binding in pipeline.reflection.bindings) {
      if (binding.type != MTLBindingTypeTensor) {
        std::fprintf(stderr, "binding %s type %ld index %lu — skipped\n", binding.name.UTF8String,
                     (long)binding.type, (unsigned long)binding.index);
        continue;
      }
      id<MTLTensorBinding> tb = (id<MTLTensorBinding>)binding;
      id<MTLTensor> tensor = tensorFor(device, tb);
      if (!tensor) return 1;
      std::fprintf(stderr, "binding %s index %lu dtype %d dims", binding.name.UTF8String,
                   (unsigned long)binding.index, (int)tb.tensorDataType);
      for (NSUInteger i = 0; i < tb.dimensions.rank; ++i)
        std::fprintf(stderr, " %ld", (long)[tb.dimensions extentAtDimensionIndex:i]);
      std::fprintf(stderr, "\n");
      tensors[binding.index] = tensor;
    }

    MTL4ArgumentTableDescriptor *tableDesc = [MTL4ArgumentTableDescriptor new];
    tableDesc.maxBufferBindCount = tensors.count;
    id<MTL4ArgumentTable> table = [device newArgumentTableWithDescriptor:tableDesc error:&error];
    if (!table) {
      std::fprintf(stderr, "argument table: %s\n", error.localizedDescription.UTF8String);
      return 1;
    }
    for (NSUInteger i = 0; i < tensors.count; ++i)
      [table setResource:tensors[i].gpuResourceID atBufferIndex:i];

    MTLHeapDescriptor *heapDesc = [MTLHeapDescriptor new];
    heapDesc.size = pipeline.intermediatesHeapSize;
    heapDesc.type = MTLHeapTypePlacement;
    id<MTLHeap> heap = [device newHeapWithDescriptor:heapDesc];

    id<MTL4CommandQueue> queue = [device newMTL4CommandQueue];
    id<MTL4CommandAllocator> allocator = [device newCommandAllocator];

    // Timed region: encode N dispatches per command buffer, wall + queue.
    const int perBuffer = 16;
    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; i += perBuffer) {
      id<MTL4CommandBuffer> commandBuffer = [device newCommandBuffer];
      [commandBuffer beginCommandBufferWithAllocator:allocator];
      id<MTL4MachineLearningCommandEncoder> encoder = [commandBuffer machineLearningCommandEncoder];
      [encoder setPipelineState:pipeline];
      [encoder setArgumentTable:table];
      for (int j = 0; j < perBuffer && i + j < iters; ++j)
        [encoder dispatchNetworkWithIntermediatesHeap:heap];
      [encoder endEncoding];
      [commandBuffer endCommandBuffer];
      MTL4CommitOptions *options = [MTL4CommitOptions new];
      __block BOOL done = NO;
      [options addFeedbackHandler:^(id<MTL4CommitFeedback>) {
        done = YES;
      }];
      id<MTL4CommandBuffer> buffers[] = {commandBuffer};
      [queue commit:buffers count:1 options:options];
      while (!done)
        ;
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    std::fprintf(stderr, "%d dispatches in %.3fs — %.1f us/dispatch\n", iters, seconds,
                 1e6 * seconds / iters);
    return 0;
  }
}
