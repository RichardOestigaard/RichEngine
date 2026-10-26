#pragma once

#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/activation.h"
#include "metal/kernels/common/rms_inverse.h"

// The pending variant: the retained rows' convolution inputs sit on the
// lane's tape between the carried state and this step's packed rows, so the
// window's sources are state for the last three, the tape for the pending
// prefix and the packed rows after (the commit folded into the scan).
inline bfloat gdn_conv_silu_pending(device const bfloat *packed,
                                    device const bfloat *conv_state_in,
                                    device const bfloat *tape_conv,
                                    uint pending,
                                    device const bfloat *conv_weights,
                                    uint packed_width, uint conv_dim,
                                    uint token, uint channel) {
  float value = 0.0f;
  for (uint tap = 0; tap < 4; ++tap) {
    // The window stays three slots: the pending rows' tail slides into it —
    // position p reads the (p + pending)-th token of the chained
    // [state | pending] prefix, or the packed rows once past it.
    const uint position = token + tap;
    bfloat input;
    if (position >= 3)
      input = packed[(position - 3) * packed_width + channel];
    else if (position + pending >= 3)
      input = tape_conv[(position + pending - 3) * conv_dim + channel];
    else
      input = conv_state_in[(position + pending) * conv_dim + channel];
    value += float(input) * float(conv_weights[channel * 4 + tap]);
  }
  value = float(bfloat(value));
  return bfloat(richengine_silu(value));
}

// Four-tap causal convolution of one channel at one token of the command,
// reading the three preceding tokens from the carried state, rounded to bf16
// and gated by SiLU.
inline bfloat gdn_conv_silu(device const bfloat *packed,
                            device const bfloat *conv_state_in,
                            device const bfloat *conv_weights,
                            uint packed_width, uint conv_dim, uint token,
                            uint channel) {
  return gdn_conv_silu_pending(packed, conv_state_in, nullptr, 0,
                               conv_weights, packed_width, conv_dim,
                               token, channel);
}

// Row `row` of the carried state after consumed_tokens: the last three inputs
// seen, still taken from the incoming state when fewer were consumed.
// The pending variant: sources past the state's three rows come from the
// pending rows' tape conv inputs first, then this step's packed rows. With
// pending rows consumed, the carry's stored window ends at the last pending
// input — pass consumed_tokens = pending for the committed carry.
inline bfloat gdn_conv_carry_pending(device const bfloat *packed,
                                     device const bfloat *conv_state_in,
                                     device const bfloat *tape_conv,
                                     uint pending, uint packed_width,
                                     uint conv_dim, uint consumed_tokens,
                                     uint row, uint channel) {
  uint source = consumed_tokens + row;
  return source < 3            ? conv_state_in[source * conv_dim + channel]
         : source < 3 + pending ? tape_conv[(source - 3) * conv_dim + channel]
                    : packed[(source - 3 - pending) * packed_width + channel];
}

// Row `row` of the carried state after consumed_tokens.
inline bfloat gdn_conv_carry(device const bfloat *packed,
                             device const bfloat *conv_state_in,
                             uint packed_width, uint conv_dim,
                             uint consumed_tokens, uint row, uint channel) {
  return gdn_conv_carry_pending(packed, conv_state_in, nullptr, 0,
                                packed_width, conv_dim, consumed_tokens,
                                row, channel);
}

// The row `hops` parents above `row` in one lane's node table (tree verify):
// the convolution and carry of a branch row read their predecessors along
// the node's path, not the adjacent DFS rows.
inline uint gdn_tree_ancestor(device const uint *nodes, uint row, uint hops) {
  for (uint i = 0; i < hops; ++i)
    row = RICHENGINE_TREE_NODE_PARENT(nodes[row]);
  return row;
}

// The tree counterpart of gdn_conv_silu: the node's four taps cover path
// positions depth-3..depth, taken from the carried state below zero and
// from the ancestor row above it.
inline bfloat gdn_conv_silu_tree(device const bfloat *packed,
                                 device const bfloat *conv_state_in,
                                 device const bfloat *conv_weights,
                                 uint packed_width, uint conv_dim, uint row,
                                 uint channel, device const uint *nodes) {
  const uint depth = RICHENGINE_TREE_NODE_DEPTH(nodes[row]);
  float value = 0.0f;
  for (uint tap = 0; tap < 4; ++tap) {
    const int position = int(depth) + int(tap);
    const bfloat input =
        position < 3
            ? conv_state_in[uint(position) * conv_dim + channel]
            : packed[ulong(gdn_tree_ancestor(nodes, row, 3 - tap)) *
                         packed_width +
                     channel];
    value += float(input) * float(conv_weights[channel * 4 + tap]);
  }
  value = float(bfloat(value));
  return bfloat(richengine_silu(value));
}

// The tree counterpart of gdn_conv_carry: the consumed prefix is the
// retained path's DFS rows, so path[p] names the row at path position p.
inline bfloat gdn_conv_carry_tree(device const bfloat *packed,
                                  device const bfloat *conv_state_in,
                                  uint packed_width, uint conv_dim,
                                  device const uint *path, uint consumed,
                                  uint row, uint channel) {
  const uint source = consumed + row;
  return source < 3
             ? conv_state_in[source * conv_dim + channel]
             : packed[ulong(path[source - 3]) * packed_width + channel];
}

// The gates of one (token, value head): beta = sigmoid(b) and
// decay = exp(a_scale * softplus(bf16(a + dt_bias))), the softplus rounded to
// bf16 as the reference does.
struct GdnGates {
  bfloat beta;
  float decay;
};

inline GdnGates gdn_gates(device const bfloat *packed_row,
                          device const bfloat *dt_bias,
                          device const float *a_scale, uint b_offset,
                          uint a_offset, uint head) {
  float b = float(packed_row[b_offset + head]);
  GdnGates gates;
  gates.beta = bfloat(richengine_sigmoid(b));
  bfloat x = bfloat(float(packed_row[a_offset + head]) + float(dt_bias[head]));
  float xf = float(x);
  bfloat softplus =
      bfloat(max(xf, 0.0f) +
             fast::log2(1.0f + fast::exp2(-1.44269504089f * abs(xf))) *
                 0.69314718056f);
  gates.decay = fast::exp(a_scale[head] * float(softplus));
  return gates;
}

// The position of value head `head` among the GDN output's head blocks: the
// head itself, or with `tiled` llama.cpp's GGUF order, which puts value head
// j of every key head next to each other (GDNGatePrefillParams).
template <uint KeyHeads, uint ValueHeads>
inline uint gdn_output_head(uint head, bool tiled) {
  constexpr uint HeadsPerKey = ValueHeads / KeyHeads;
  return tiled ? (head % HeadsPerKey) * KeyHeads + head / HeadsPerKey : head;
}

// Gated RMSNorm of the recurrent row of one task (token, value head), one
// thread per dimension, stored at the head's output position. Prefill
// dispatches one task per threadgroup; decode runs gdn_decode_gate, which
// reproduces these rows bitwise. The norm weights are read in their stored
// type W: bfloat in the packed formats, float for a GGUF's F32 norms.
template <uint KeyHeads, uint ValueHeads, uint HeadDim, uint ConvDim,
          uint PackedWidth, class W>
inline void
gdn_gate_phase(device const bfloat *recurrent, device const bfloat *packed,
               device const W *norm_weight, device bfloat *hidden, uint task,
               bool tiled, threadgroup float *scratch, uint thread_index,
               uint lane, uint simd_group) {
  constexpr uint Simdgroups = 4, ZOffset = ConvDim;
  static_assert(HeadDim == Simdgroups * 32, "one thread per dimension");
  uint token = task / ValueHeads;
  uint head = task % ValueHeads;
  ulong base = ulong(task) * HeadDim;
  ulong hidden_base =
      (ulong(token) * ValueHeads +
       gdn_output_head<KeyHeads, ValueHeads>(head, tiled)) *
      HeadDim;
  float value = float(recurrent[base + thread_index]);
  const float inverse = rms_inverse_of_sums<Simdgroups>(
      value * value, HeadDim, scratch, thread_index, lane, simd_group);
  bfloat normalized =
      bfloat(value * inverse * float(norm_weight[thread_index]));
  float gate = float(packed[token * PackedWidth + ZOffset + head * HeadDim +
                            thread_index]);
  float silu = richengine_silu(gate);
  hidden[hidden_base + thread_index] = bfloat(float(normalized) * silu);
}
