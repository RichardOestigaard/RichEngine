#pragma once

// The C ABI of librichengine_coreai.dylib (ops/CoreAiShim.swift): the
// Swift-only CoreAI framework's encode() path for ops::CoreAiFfn. Loaded
// with dlopen/dlsym so that a Mac without the framework simply leaves the
// split unavailable.
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void *coreai_ffn_ref;

// An id<MTLBuffer> and a byte offset into it.
typedef struct {
  void *buffer;
  uint64_t offset;
} coreai_ffn_buf;

// One evaluation: rows x hidden fp16 activations to rows x hidden fp16,
// the layer's int8 weights and fp16 scales, and the shared-event values
// bracketing it.
typedef struct {
  coreai_ffn_buf x, y;
  coreai_ffn_buf gate_q, gate_s, up_q, up_s, down_q, down_s;
  uint64_t wait, signal;
  uint32_t rows;
} coreai_ffn_job;

// Loads function `name` of the .aimodel at `path`, specialized for the
// Neural Engine, with a ComputeStream over a new command queue of `device`
// (an id<MTLDevice>). Every job waits for `event` (an id<MTLSharedEvent>) to
// reach `wait` and raises it to `signal`, in queue order against the
// caller's own Metal commands. The model must be the FFN of `hidden`
// hidden and `ane` intermediate channels with runtime int8 weight inputs.
// Returns NULL on failure with a UTF-8 reason at `error` (owned by the
// library).
coreai_ffn_ref coreai_ffn_open(const char *path, const char *name, void *device, void *event, uint64_t hidden,
                               uint64_t ane, const char **error);

// Encodes the job on the function's stream: the event wait, the
// evaluation, the signal, in the stream's queue order. `report(ctx, ok)`
// runs once the work completes; a job that cannot be encoded reports at
// once with ok 0. Returns 0 only when `ffn` is invalid.
int coreai_ffn_start(coreai_ffn_ref ffn, const coreai_ffn_job *job, void (*report)(void *ctx, int ok), void *ctx);

void coreai_ffn_close(coreai_ffn_ref ffn);

#ifdef __cplusplus
}
#endif
