#pragma once

// C/Metal ABI constants shared by host and shader compilation. Startup
// static assertions check the corresponding model and operator contracts.
#define RICHENGINE_DRAFT_QUERY_ROWS 16u
#define RICHENGINE_DRAFT_PROPOSAL_TOKENS 15u
#define RICHENGINE_TARGET_VERIFY_ROWS 16u
// Nodes of one lane's verify tree in tree-verify mode (docs/TREE_VERIFY_DESIGN.md):
// the anchor plus the seven-node chain plus the seven second-best sibling
// leaves emitted by draft_select_tree (rows 8..14, one per proposal
// position; row 15 is spare). Verify-row tensors stay strided by
// RICHENGINE_TARGET_VERIFY_ROWS in chain mode; the tree layout occupies rows
// 0..tree_counts[lane] of a lane's RICHENGINE_TREE_VERIFY_NODES-stride region.
#define RICHENGINE_TREE_VERIFY_NODES 16u
#define RICHENGINE_MAXIMUM_CONTEXT_TOKENS 262144u
// Verify rows are stored to page slots ahead of acceptance; a tree lane
// occupies TREE_VERIFY_NODES - 1 slots before its path is compacted, so the
// scratch covers the larger of the chain and tree footprints.
#define RICHENGINE_SPECULATIVE_SCRATCH_TOKENS                                  \
  (RICHENGINE_TREE_VERIFY_NODES - 1u > RICHENGINE_TARGET_VERIFY_ROWS           \
       ? RICHENGINE_TREE_VERIFY_NODES - 1u                                     \
       : RICHENGINE_TARGET_VERIFY_ROWS)
#define RICHENGINE_MAXIMUM_PHYSICAL_KV_TOKENS                                  \
  (RICHENGINE_MAXIMUM_CONTEXT_TOKENS + RICHENGINE_SPECULATIVE_SCRATCH_TOKENS)
#define RICHENGINE_MAXIMUM_BATCH_WIDTH 4u
#define RICHENGINE_PREFILL_TOKEN_BUDGET 2048u
// Physical slots per KV head of the draft KV ring: the widest declared
// draft sliding_window the runtime serves. A draft's own declared window
// bounds how much of the ring its attention reads (DraftAttentionBatchParams).
#define RICHENGINE_DRAFT_SLIDING_WINDOW 4096u
#define RICHENGINE_TARGET_KV_BLOCK_TOKENS 32u
// Rows per KV head (and per query group) of one lane's verify chunk staging:
// one KV block, which holds the lane's RICHENGINE_TARGET_VERIFY_ROWS rows.
#define RICHENGINE_VERIFY_CHUNK_STRIDE RICHENGINE_TARGET_KV_BLOCK_TOKENS
#define RICHENGINE_PREFILL_ATTENTION_TILE_ROWS 8u
#define RICHENGINE_PREFILL_ATTENTION_MAXIMUM_SPLITS 32u
// Draft attention deals the live ring tiles of one (lane, KV head)
// round-robin to this many groups, the last of which also attends the eight
// current rows. RICHENGINE_DRAFT_SPLITS overrides the count at process
// start (Tuning.hpp): the kernels read it through the Metal function
// constant at this index, the host through the same once-read snapshot —
// the workspace, the dispatch's z extent and the kernel loop bounds can
// never disagree. The override clamps to DRAFT_SPLITS_MAXIMUM.
#define RICHENGINE_DRAFT_ATTENTION_SPLITS 4u
#define RICHENGINE_DRAFT_SPLITS_FUNCTION_CONSTANT 0u
#define RICHENGINE_DRAFT_SPLITS_MAXIMUM 16u
#define RICHENGINE_VERIFY_ATTENTION_MAXIMUM_SPLITS 128u
#define RICHENGINE_TARGET_SAMPLING_SHARDS 16u
// Threads of each group that selects a sampled row over the whole
// vocabulary (decode_sample_vocabulary*); a bracket of at most this many
// tokens is ordered in threadgroup memory, one token per thread.
#define RICHENGINE_TARGET_VOCABULARY_THREADS 1024u
// Groups that share such a row's draw, each over its own slice of the
// vocabulary, so the eight rows of a lane spread over the GPU's cores.
#define RICHENGINE_TARGET_VOCABULARY_GROUPS 8u
// The ranges of the vocabulary such a row's draw sums: one per simdgroup of
// the row's groups.
#define RICHENGINE_TARGET_VOCABULARY_RANGES                                    \
  (RICHENGINE_TARGET_VOCABULARY_GROUPS * (RICHENGINE_TARGET_VOCABULARY_THREADS / 32u))
#define RICHENGINE_DRAFT_SAMPLING_SHARDS 8u
// Candidates the draft selector keeps per proposal position; acceptance and
// the sampled draw read them.
#define RICHENGINE_DRAFT_CANDIDATES 16u
// The DSpark candidate pool: every slot the shard partials emit, scored by
// the Markov edge table rather than merged down to the top candidates.
#define RICHENGINE_DSPARK_POOL                                               \
  (RICHENGINE_DRAFT_SAMPLING_SHARDS * RICHENGINE_DRAFT_CANDIDATES)
// The rank of the draft selector's codebooks.
#define RICHENGINE_DRAFT_SELECTOR_RANK 256u
// Rows of one value head's recurrent state a prefill GDN scan threadgroup
// carries through the chunk: four simdgroups whose lanes each own sixteen key
// columns of one row. The thread count follows from the rows: 128 columns /
// 16 per lane = 8 lanes per row, times the 16 rows; a static_assert in
// prefill/gdn.metal ties the two literals together.
#define RICHENGINE_GDN_SCAN_STATE_ROWS 16u
#define RICHENGINE_GDN_SCAN_THREADS 128u
// Plain norms of at most RICHENGINE_STAGED_NORM_ROWS rows of at most
// RICHENGINE_STAGED_NORM_WIDTH columns run norm_rms_staged, whose 1024-thread
// groups hold a row in threadgroup memory: the region where it measured
// faster than norm_rms (shared/normalization.metal), which covers every
// decode norm of a 2048-wide model and its short prefill chunks.
#define RICHENGINE_STAGED_NORM_WIDTH 2048u
#define RICHENGINE_STAGED_NORM_ROWS 64u
#define RICHENGINE_STAGED_NORM_THREADS 1024u
