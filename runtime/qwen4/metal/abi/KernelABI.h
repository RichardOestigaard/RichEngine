#pragma once

// Shader prelude. The ABI headers below also compile as host C++; their
// shared parameter layouts use fixed-width scalars and explicit padding.

#include "qwen4/metal/abi/AneFfn.h"
#include "qwen4/metal/abi/DraftAttention.h"
#include "qwen4/metal/abi/Embedding.h"
#include "qwen4/metal/abi/ExecutionGeometry.h"
#include "qwen4/metal/abi/GDN.h"
#include "qwen4/metal/abi/MoE.h"
#include "qwen4/metal/abi/PagedAttention.h"
#include "qwen4/metal/abi/RoPE.h"
#include "qwen4/metal/abi/RowCopy.h"
#include "qwen4/metal/abi/Sampling.h"
#include "qwen4/metal/abi/Vision.h"
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>

using namespace metal;
using namespace mpp::tensor_ops;
