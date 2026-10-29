#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#include "metal/abi/ExecutionGeometry.h"
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// The uniforms one lane draws a decode cycle with, in [0, 1): the first
// token after a prompt draws RICHENGINE_UNIFORM_INITIAL; the draft's sampled
// proposal at position p draws RICHENGINE_UNIFORM_PROPOSALS + p; acceptance tests
// draft token p against RICHENGINE_UNIFORM_ACCEPTANCE + p; and a sampled verify
// row draws its correction, or the bonus token after the whole draft, with
// RICHENGINE_UNIFORM_CORRECTION. Lane l's uniforms start at
// l * RICHENGINE_SAMPLING_UNIFORMS.
#define RICHENGINE_UNIFORM_INITIAL 0u
#define RICHENGINE_UNIFORM_PROPOSALS 1u
#define RICHENGINE_UNIFORM_ACCEPTANCE                                          \
  (RICHENGINE_UNIFORM_PROPOSALS + RICHENGINE_DRAFT_PROPOSAL_TOKENS)
#define RICHENGINE_UNIFORM_CORRECTION                                          \
  (RICHENGINE_UNIFORM_ACCEPTANCE + RICHENGINE_DRAFT_PROPOSAL_TOKENS)
#define RICHENGINE_SAMPLING_UNIFORMS (RICHENGINE_UNIFORM_CORRECTION + 1u)

// One target-policy dispatch over lanes of RICHENGINE_TARGET_VERIFY_ROWS logits
// rows, RICHENGINE_TARGET_VERIFY_ROWS + 1 constraint-mask rows and
// RICHENGINE_SAMPLING_UNIFORMS uniforms each. Selected row s is row s % rows of
// lane s / rows: its logits row is lane * RICHENGINE_TARGET_VERIFY_ROWS +
// logits_row + s % rows, its mask row lane * (RICHENGINE_TARGET_VERIFY_ROWS + 1) +
// mask_row + s % rows, and its draw takes uniform
// lane * RICHENGINE_SAMPLING_UNIFORMS + uniform; workspaces and output tokens are
// indexed by s. Rows below drafted_rows follow draft token s % rows (verify
// input row s % rows + 1).
// The fused vocabulary-head argmax (decode_head_argmax_q4): one 32 x 128
// logits tile per threadgroup, row maxima/index partials, no logits writes.
// Emitted only on all-greedy unconstrained chain-verify steps — the
// constraint mask is unused (no lane carries one), dead-row partials are
// skipped by the tile reducer like the sharded argmax's.
struct HeadArgmaxParams {
  uint32_t output_size;
  uint32_t input_size;
  uint32_t rows;
  uint32_t exclude_stop_mask;
  uint32_t stop_token_0;
  uint32_t stop_token_1;
  uint32_t live_rows[RICHENGINE_MAXIMUM_BATCH_WIDTH];
  // The head's packed output rows: output_size padded to its tile. A
  // vocabulary that is not a tile multiple pads the last tile's rows; the
  // pad tokens' argmax stays out by the output_size bound.
  uint32_t output_stride;
  // Per-lane logits writeback (the *_logits variants,
  // RICHENGINE_HEAD_PER_LANE): rows of a flagged lane are stored to the
  // logits buffer beside the argmax partials, so the batch keeps the fused
  // head while sampled, penalized and constrained lanes still read their
  // rows. Zero writes nothing — the all-greedy behavior.
  uint32_t logits_lane_mask;
};

static_assert(sizeof(HeadArgmaxParams) == 48,
              "Head argmax parameters are 48 bytes on both sides");

struct TargetSamplingParams {
  uint32_t vocabulary;
  uint32_t mask_words;
  uint32_t rows;
  uint32_t logits_row;
  uint32_t mask_row;
  uint32_t uniform;
  uint32_t drafted_rows;
  uint32_t top_k[RICHENGINE_MAXIMUM_BATCH_WIDTH];
  float temperature[RICHENGINE_MAXIMUM_BATCH_WIDTH];
  float top_p[RICHENGINE_MAXIMUM_BATCH_WIDTH];
  float min_p[RICHENGINE_MAXIMUM_BATCH_WIDTH];
  // Lanes that sample; the others take the argmax.
  uint32_t sampling_mask;
  uint32_t constrained_mask;
  // Lanes that ignore end-of-sequence: they never select a stop token.
  uint32_t exclude_stop_mask;
  uint32_t stop_token_0;
  uint32_t stop_token_1;
  // Adaptive proposal budgets (RICHENGINE_ADAPTIVE_PROPOSALS): the lane's live
  // selected rows. A row s with s % rows >= live_rows[lane] is dead — the
  // selection kernels skip it. Unadapted dispatches fill every lane with
  // rows.
  uint32_t live_rows[RICHENGINE_MAXIMUM_BATCH_WIDTH];
  // The packed lane stride of the logits and mask rows: a chain batch's
  // verify stride (draft.verifyRows()), RICHENGINE_TREE_VERIFY_NODES for a
  // tree lane.
  uint32_t lane_stride;
  // A row's logits span in the logits buffer: vocabulary padded to the
  // head's section rows when the vocabulary is not a tile multiple.
  uint32_t logits_stride;
  // Constrained lanes whose admission reads the GPU-DFA next-state rows
  // (DecodeTensor::DfaNextStates) instead of the bitmask: the row for
  // selected row s is mask_row + s % rows of the lane's
  // RICHENGINE_TARGET_VERIFY_ROWS + 1 rows of vocabulary int16s, and a token
  // is admitted iff its entry is not -1.
  uint32_t dfa_mask;
  // Lanes the fused head's argmax already answered
  // (RICHENGINE_HEAD_PER_LANE): the sharded argmax and reduce skip their
  // rows, since their logits rows were never written. Zero runs every
  // greedy lane — the unfused behavior.
  uint32_t fused_mask;
};

static_assert(sizeof(TargetSamplingParams) == 144,
              "Target sampling parameters are 144 bytes on both sides");

// One shard's share of a sampled row's softmax denominator: the largest
// logit it admits, the sum of exp((logit - maximum) / temperature) over its
// admitted tokens, and how many it admits.
struct TargetShardMass {
  float maximum;
  float sum;
  uint32_t admitted;
};

static_assert(sizeof(TargetShardMass) == 12,
              "Target shard masses are 12 bytes on both sides");

// A sampled row's selection over the whole vocabulary. The search records
// the row's largest admitted logit and where its min-p/top-k/top-p
// distribution ends in the order of the logits (the key and id of its last
// token, metal/kernels/decode/sampling.metal). The draw writes a drafted
// row's target probability of its draft token; the token it draws goes to
// the output tokens: for a verify row with a draft token, the correction
// acceptance takes if it rejects that token (a draw from the residual
// distribution); otherwise a draw from the row's distribution.
struct TargetVocabularyRow {
  float maximum;
  uint32_t end_key;
  uint32_t end_last;
  float draft_probability;
};

static_assert(sizeof(TargetVocabularyRow) == 16,
              "Target vocabulary rows are 16 bytes on both sides");

// One range of the vocabulary in such a row's draw, which one simdgroup of
// the row's groups sums: the kept weight of its tokens other than the
// draft's candidates, and one past the last of those with weight.
struct TargetVocabularyRange {
  float rest;
  uint32_t after;
};

static_assert(sizeof(TargetVocabularyRange) == 8,
              "Target vocabulary ranges are 8 bytes on both sides");

// A penalized request's word for each vocabulary token, in its state lane's
// row of the penalty table (ops::Sampling::rebuildPenaltyWords): the
// prompt bit marks a prompt token, and the count is how often the target
// selected it.
#define RICHENGINE_PENALTY_PROMPT_BIT 0x80000000u
#define RICHENGINE_PENALTY_COUNT_MASK 0x7fffffffu

// The penalized lanes of one penalty dispatch. Each entry names the lane of
// its logits and the penalty table row it reads; rows penalizes that many
// rows of the lane's RICHENGINE_TARGET_VERIFY_ROWS, from row_offset.
// repetition_inverse is 1 / repetition, saturated to the largest float.
struct SamplingPenaltyParams {
  uint32_t vocabulary;
  uint32_t rows;
  uint32_t row_offset;
  uint32_t entries;
  uint32_t logits_lane[RICHENGINE_MAXIMUM_BATCH_WIDTH];
  uint32_t table_row[RICHENGINE_MAXIMUM_BATCH_WIDTH];
  float repetition[RICHENGINE_MAXIMUM_BATCH_WIDTH];
  float repetition_inverse[RICHENGINE_MAXIMUM_BATCH_WIDTH];
  float presence[RICHENGINE_MAXIMUM_BATCH_WIDTH];
  float frequency[RICHENGINE_MAXIMUM_BATCH_WIDTH];
  // The packed lane stride of the logits and verify-input rows each entry
  // addresses: the batch's verify stride (draft.verifyRows()).
  uint32_t verify_stride;
  // A row's logits span in the logits buffer (vocabulary padded to the
  // head's section rows); the words table strides vocabulary.
  uint32_t logits_stride;
};

static_assert(sizeof(SamplingPenaltyParams) == 120,
              "Sampling penalty parameters are 120 bytes on both sides");

// The batched selector and acceptance kernels' grids cover exactly the
// dispatch's lanes: per-lane arrays hold those lanes, and entries past them
// are zero and unread. tree_mask marks the lanes whose draft_select_tree
// emits sibling leaves beyond the chain (greedy, unconstrained lanes of a
// tree-capable draft); sampled lanes always get a linear node table.
struct SelectorBatchParams {
  uint32_t anchor[RICHENGINE_MAXIMUM_BATCH_WIDTH];
  float temperature[RICHENGINE_MAXIMUM_BATCH_WIDTH];
  uint32_t lanes;
  uint32_t sampling_mask;
  uint32_t vocabulary;
  uint32_t tree_mask;
  // The lanes' packed draft query-row stride: the draft's trained-block
  // depth (draft.queryRows()), at most RICHENGINE_DRAFT_QUERY_ROWS.
  uint32_t query_rows;
};

static_assert(sizeof(SelectorBatchParams) == 52,
              "Draft selector parameters are 52 bytes on both sides");

// The rows of its block each lane of a decode cycle verifies (its verify
// rows, draft_verify_rows), chosen for the whole batch by expected accepted
// tokens per unit of the cycle's cost (model::VerifyBudget). The selector
// writes each lane's confidence at each position: the largest probability
// among the position's candidates (the softmax at temperature 1 of their
// scores), which does not depend on the candidate the lane draws there.
// Once the proposals before it are accepted, a proposal is accepted with
// probability p, where logit(p) = slope * logit(confidence) +
// intercept[position]; a lane's survival at a position, the product of p up
// to it, is the probability that its proposals up to it are all accepted,
// and verifying one more row adds that much to the lane's expected tokens.
// Every lane verifies its anchor; the rows past it go to the largest
// survivals over all lanes, as many as maximize the expected tokens over
// cost[the rows verified in all], none past a lane's maximum (its live
// rows). With every_row, each lane verifies its maximum. The verify's MoE
// steps route the verify rows alone (moe_group_routes_verified), and
// acceptance retains at most that many tokens.
// conf_min, when nonzero, additionally caps a lane's rows at its
// confidence prefix: the first position whose confidence head sigmoid
// (draft_conf_score, RICHENGINE_DRAFT_CONF_MIN) falls under conf_min cuts
// the proposal suffix there — the DSpark confidence-scheduled prune.
struct VerifyRowsParams {
  float slope;
  float intercept[RICHENGINE_DRAFT_PROPOSAL_TOKENS];
  float cost[RICHENGINE_DRAFT_QUERY_ROWS * RICHENGINE_MAXIMUM_BATCH_WIDTH + 1];
  uint32_t lanes;
  uint32_t every_row;
  uint32_t maximum[RICHENGINE_MAXIMUM_BATCH_WIDTH];
  float conf_min;
};

static_assert(sizeof(VerifyRowsParams) ==
                  16 + 4 * RICHENGINE_DRAFT_PROPOSAL_TOKENS +
                      4 * (RICHENGINE_DRAFT_QUERY_ROWS *
                               RICHENGINE_MAXIMUM_BATCH_WIDTH +
                           1) +
                      4 * RICHENGINE_MAXIMUM_BATCH_WIDTH,
              "Verify rows parameters differ across the ABI");

// draft_conf_score's payload: the DSpark confidence head's operand
// geometry — the lanes, the draft's final-hidden width, the Markov rank
// whose W1 predecessor feature the head appends, the proposal table's
// token bound and the draft block's packed query-row stride.
struct DraftConfScoreParams {
  uint32_t lanes;
  uint32_t hidden_size;
  uint32_t rank;
  uint32_t vocabulary;
  uint32_t query_rows;
};

static_assert(sizeof(DraftConfScoreParams) == 20,
              "Draft confidence score parameters are 20 bytes on both sides");

// The draft top-16 shard kernels' scalar payload (draft_select_top16_sharded,
// dspark_select_top16_sharded): the token bound and the logits row stride,
// which differ on a padded-vocabulary target.
struct DraftTop16Params {
  uint32_t vocabulary;
  uint32_t logits_stride;
};

static_assert(sizeof(DraftTop16Params) == 8,
              "Draft top-16 parameters are 8 bytes on both sides");

// One node of a lane's verify tree (draft_select_tree): the DFS row of its
// parent (0xFF for the anchor), its depth from the anchor, and its proposal
// position (0xFF for the anchor). Row 0 is the anchor, rows 1..7 the chain,
// rows 8.. the sibling leaves. A node's row also indexes its verify input
// token (TreeTokens), its logits row and its attention/GDN row.
#define RICHENGINE_TREE_NODE_PARENT(node) ((node)&0xffu)
#define RICHENGINE_TREE_NODE_DEPTH(node) (((node) >> 8) & 0xffu)
#define RICHENGINE_TREE_NODE_POSITION(node) (((node) >> 16) & 0xffu)
#define RICHENGINE_TREE_NODE_NONE 0xffu

struct AcceptBatchParams {
  uint32_t remaining[RICHENGINE_MAXIMUM_BATCH_WIDTH];
  uint32_t stop_token_0;
  uint32_t stop_token_1;
  uint32_t sampling_mask;
  // Adaptive proposal budgets: the most draft tokens the lane may accept
  // this step (RICHENGINE_DRAFT_PROPOSAL_TOKENS when unadapted). A lane's
  // retained count stays within proposals + 1 live rows.
  uint32_t proposals[RICHENGINE_MAXIMUM_BATCH_WIDTH];
  // Candidate-table stride: the draft ids and proposal probabilities of one
  // position (RICHENGINE_DRAFT_CANDIDATES for merged selectors, the
  // shards-times-candidates pool for DSpark).
  uint32_t candidate_stride;
  // The packed lane stride of the target rows: the batch's verify stride
  // (draft.verifyRows()).
  uint32_t verify_stride;
  // Greedy GPU-DFA lanes whose accepted prefix is capped by the truncation
  // pass's per-lane row (buffer 9): the row emits the masked correction,
  // so the prefix must stop there even when the corrected token equals the
  // proposal.
  uint32_t trunc_mask;
};

static_assert(sizeof(AcceptBatchParams) == 56,
              "Batched acceptance parameters are 56 bytes on both sides");

// verify_input_tree_tokens parameters: one thread per (lane, node) fills the
// node's verify input token, its (t, h, w) rope position (the lane's base
// triple plus the node's depth) and its ancestor bitmask.
struct VerifyTreeInputParams {
  uint32_t vocabulary;
  uint32_t mask_token;
  uint32_t base[RICHENGINE_MAXIMUM_BATCH_WIDTH][3];
};

static_assert(sizeof(VerifyTreeInputParams) == 56,
              "Verify tree input parameters are 56 bytes on both sides");

// decode_accept_tree walks one lane's tree: it shares the remaining/limits
// fields of AcceptBatchParams and adds nothing else.
struct TreeAcceptBatchParams {
  uint32_t remaining[RICHENGINE_MAXIMUM_BATCH_WIDTH];
  uint32_t stop_token_0;
  uint32_t stop_token_1;
  uint32_t lanes;
};

static_assert(sizeof(TreeAcceptBatchParams) == 28,
              "Batched tree acceptance parameters are 28 bytes on both sides");

// tree_leaf_patch splices ANE-produced leaf alternates into a lane's comb
// rows: the kernel runs while a predictor job may still be in flight, so it
// applies only when the flag's serial matches the one the encode snapshotted.
struct TreeLeafPatchParams {
  uint32_t expected;
  uint32_t lanes;
};

static_assert(sizeof(TreeLeafPatchParams) == 8,
              "Tree leaf patch parameters are 8 bytes on both sides");
