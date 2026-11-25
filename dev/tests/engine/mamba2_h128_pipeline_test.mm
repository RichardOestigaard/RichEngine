// Bolt (_h128) Mamba-2 pipeline creation: every 128-head verify, commit and
// prefill variant must build its compute pipeline within the device's 32KB
// threadgroup memory budget at both verify geometries. Bolt's sixteen-head
// groups overflowed the budget at sixteen verify rows when the scan staged
// its operands in threadgroup memory (43136 bytes); the unstaged layout
// reads them back out of `mixed` instead. The Lightning 64-head variants
// are checked too — their staged layout must keep compiling.
//
// Standalone like canvas-kernels: the kernels live in the production
// metallib, so the test links no engine objects and takes $(LIB) as argv[1].
//
// usage: mamba2-h128-pipelines METALLIB
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdio>

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    if (argc != 2) {
      fprintf(stderr, "usage: mamba2-h128-pipelines METALLIB\n");
      return 2;
    }
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    NSError *error = nil;
    NSData *fileData =
        [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:argv[1]]];
    if (!fileData) {
      fprintf(stderr, "unable to read metallib\n");
      return 1;
    }
    dispatch_data_t data = dispatch_data_create(
        fileData.bytes, fileData.length, nullptr, ^{ (void)fileData; });
    id<MTLLibrary> library = [device newLibraryWithData:data error:&error];
    if (!library) {
      fprintf(stderr, "library: %s\n", error.localizedDescription.UTF8String);
      return 1;
    }
    const char *names[] = {
        "verify_mamba2_fused",
        "verify_mamba2_fused_f32",
        "verify_mamba2_fused_table64",
        "verify_mamba2_fused_table16",
        "verify_mamba2_fused_table16_f32",
        "verify_mamba2_fused_packed",
        "verify_mamba2_fused_packed_f32",
        "verify_mamba2_fused_h128",
        "verify_mamba2_fused_h128_f32",
        "verify_mamba2_fused_h128_table64",
        "verify_mamba2_fused_h128_table16",
        "verify_mamba2_fused_h128_table16_f32",
        "verify_mamba2_fused_h128_packed",
        "verify_mamba2_fused_h128_packed_f32",
        "verify_mamba2_commit",
        "verify_mamba2_commit_h128",
        "prefill_mamba2_prepare_h128",
        "prefill_mamba2_scan_h128",
        "prefill_mamba2_gate_h128",
        "prefill_mamba2_gate_h128_f32",
        "prefill_mamba2_gate_sums_h128",
        "prefill_mamba2_gate_sums_h128_f32",
    };
    int failures = 0;
    for (const char *entry : names) {
      id<MTLFunction> function =
          [library newFunctionWithName:[NSString stringWithUTF8String:entry]];
      if (!function) {
        printf("missing  %s\n", entry);
        ++failures;
        continue;
      }
      error = nil;
      id<MTLComputePipelineState> pipeline =
          [device newComputePipelineStateWithFunction:function error:&error];
      if (!pipeline) {
        printf("FAIL     %s: %s\n", entry,
               error.localizedDescription.UTF8String);
        ++failures;
      } else {
        printf("ok       %s\n", entry);
      }
    }
    printf("failures: %d\n", failures);
    return failures ? 1 : 0;
  }
}
