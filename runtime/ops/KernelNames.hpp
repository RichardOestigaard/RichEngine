#pragma once

// The Metal pipeline names the host dispatch code encodes, one constant per
// kernel (CommandGraph::add takes std::string, so add sites wrap these in
// std::string(); helpers taking const char * take .data()). Names that end
// in "_" or feed suffixes (e.g. "_vh32", "_m32", a format tag) are stems the
// ops assemble into the compiled instantiations.
#include <string_view>

namespace richengine::ops {

// Sampling (decode/, verify/ kernels).
inline constexpr std::string_view kDecodeSamplePenalize = "decode_sample_penalize";
inline constexpr std::string_view kDecodeSamplePenalizeVerify = "decode_sample_penalize_verify";
inline constexpr std::string_view kDecodeHeadArgmaxReduceTiles = "decode_head_argmax_reduce_tiles";
inline constexpr std::string_view kDecodeHeadArgmaxReduceTilesGguf = "decode_head_argmax_reduce_tiles_gguf";
inline constexpr std::string_view kDecodeSampleArgmaxSharded = "decode_sample_argmax_sharded";
inline constexpr std::string_view kDecodeSampleArgmaxReduce = "decode_sample_argmax_reduce";
inline constexpr std::string_view kDecodeSampleMassSharded = "decode_sample_mass_sharded";
inline constexpr std::string_view kDecodeSampleVocabularySearch = "decode_sample_vocabulary_search";
inline constexpr std::string_view kDecodeSampleVocabularyDraw = "decode_sample_vocabulary_draw";
inline constexpr std::string_view kDecodeAcceptDflash = "decode_accept_dflash";
inline constexpr std::string_view kDecodeAcceptTree = "decode_accept_tree";
inline constexpr std::string_view kDecodeLogitSoftcap = "decode_logit_softcap";
inline constexpr std::string_view kDecodeHeadArgmaxQ4 = "decode_head_argmax_q4";
inline constexpr std::string_view kTreeLeafPatch = "tree_leaf_patch";

// Draft attention heads and stages.
inline constexpr std::string_view kDraftConv = "draft_conv";
inline constexpr std::string_view kDraftConvH2048 = "draft_conv_h2048";
inline constexpr std::string_view kDraftResidualAdd = "draft_residual_add";
inline constexpr std::string_view kDraftAttentionQkv = "draft_attention_qkv";
inline constexpr std::string_view kDraftAttentionQkvQ16k2 = "draft_attention_qkv_q16k2";
inline constexpr std::string_view kDraftAttentionQkvQ32k8d64i = "draft_attention_qkv_q32k8d64i";
inline constexpr std::string_view kDraftAttentionBf16Split = "draft_attention_bf16_split";
inline constexpr std::string_view kDraftAttentionBf16SplitQ16k2 = "draft_attention_bf16_split_q16k2";
inline constexpr std::string_view kDraftAttentionBf16SplitQ32k8d64i = "draft_attention_bf16_split_q32k8d64i";
inline constexpr std::string_view kDraftAttentionBf16Reduce = "draft_attention_bf16_reduce";
inline constexpr std::string_view kDraftAttentionBf16ReduceQ16k2 = "draft_attention_bf16_reduce_q16k2";
inline constexpr std::string_view kDraftAttentionBf16ReduceQ32k8d64i = "draft_attention_bf16_reduce_q32k8d64i";
inline constexpr std::string_view kDraftAttentionReorder = "draft_attention_reorder";
inline constexpr std::string_view kDraftAttentionReorderQ16k2 = "draft_attention_reorder_q16k2";
inline constexpr std::string_view kDraftAttentionReorderQ32k8d64i = "draft_attention_reorder_q32k8d64i";
inline constexpr std::string_view kDraftContextKvCommit = "draft_context_kv_commit";
inline constexpr std::string_view kDraftContextKvCommitQ16k2 = "draft_context_kv_commit_q16k2";
inline constexpr std::string_view kDraftContextKvCommitQ32k8d64i = "draft_context_kv_commit_q32k8d64i";
inline constexpr std::string_view kPrefillDraftContextKv = "prefill_draft_context_kv";
inline constexpr std::string_view kPrefillDraftContextKvQ16k2 = "prefill_draft_context_kv_q16k2";
inline constexpr std::string_view kPrefillDraftContextKvQ32k8d64i = "prefill_draft_context_kv_q32k8d64i";

// Draft candidate selection.
inline constexpr std::string_view kDraftSelectTop16Sharded = "draft_select_top16_sharded";
inline constexpr std::string_view kDraftSelectEdges = "draft_select_edges";
inline constexpr std::string_view kDraftSelectDflash = "draft_select_dflash";
inline constexpr std::string_view kDraftSelectTree = "draft_select_tree";
inline constexpr std::string_view kDraftSelectPlain = "draft_select_plain";
inline constexpr std::string_view kDraftSelectPlainTree = "draft_select_plain_tree";
inline constexpr std::string_view kDraftSelectDspark = "draft_select_dspark";
inline constexpr std::string_view kDraftSelectDsparkTree = "draft_select_dspark_tree";
inline constexpr std::string_view kDsparkSelectTop16Sharded = "dspark_select_top16_sharded";
inline constexpr std::string_view kDsparkSelectEdges = "dspark_select_edges";
inline constexpr std::string_view kDflashSelectPoolEdges = "dflash_select_pool_edges";

// Row movement and tables.
inline constexpr std::string_view kCopyRowsBf16 = "copy_rows_bf16";
inline constexpr std::string_view kRopeBuildTables = "rope_build_tables";
inline constexpr std::string_view kGegluMultiply = "geglu_multiply";

// Gated delta net.
inline constexpr std::string_view kPrefillGdnPrepare = "prefill_gdn_prepare";
inline constexpr std::string_view kPrefillGdnPrepareVh32 = "prefill_gdn_prepare_vh32";
inline constexpr std::string_view kPrefillGdnScan = "prefill_gdn_scan";
inline constexpr std::string_view kPrefillGdnScanVh32 = "prefill_gdn_scan_vh32";
inline constexpr std::string_view kPrefillGdnGate = "prefill_gdn_gate";
inline constexpr std::string_view kPrefillGdnGateSums = "prefill_gdn_gate_sums";
inline constexpr std::string_view kPrefillGdnGateVh32 = "prefill_gdn_gate_vh32";
inline constexpr std::string_view kPrefillGdnGateSumsVh32 = "prefill_gdn_gate_sums_vh32";
// Stems: the chunked kernels take a "_c<factor>" suffix.
inline constexpr std::string_view kGdnChunkedPrep = "gdn_chunked_prep";
inline constexpr std::string_view kGdnChunkedScan = "gdn_chunked_scan";
// Stems: the fused verifies take a table suffix then an optional "_vh32".
inline constexpr std::string_view kVerifyGdnFused = "verify_gdn_fused";
inline constexpr std::string_view kVerifyGdnFusedVh32 = "verify_gdn_fused_vh32";
inline constexpr std::string_view kVerifyTreeGdnFused = "verify_tree_gdn_fused";
inline constexpr std::string_view kVerifyGdnCommit = "verify_gdn_commit";
inline constexpr std::string_view kVerifyGdnCommitVh32 = "verify_gdn_commit_vh32";
inline constexpr std::string_view kVerifyGdnCommitTree = "verify_gdn_commit_tree";
inline constexpr std::string_view kVerifyGdnCommitTreeVh32 = "verify_gdn_commit_tree_vh32";

// RMS norms (normKernel() appends the epsilon/width tags).
inline constexpr std::string_view kNormRms = "norm_rms";
inline constexpr std::string_view kNormRmsStaged = "norm_rms_staged";
inline constexpr std::string_view kPrefillNormRmsSums32 = "prefill_norm_rms_sums32";

// LFM short convolutions.
inline constexpr std::string_view kPrefillLfmConv = "prefill_lfm_conv";
inline constexpr std::string_view kPrefillLfmConvV4 = "prefill_lfm_conv_v4";
inline constexpr std::string_view kVerifyLfmConv = "verify_lfm_conv";
inline constexpr std::string_view kVerifyLfmConvV4 = "verify_lfm_conv_v4";
inline constexpr std::string_view kCommitLfmConv = "commit_lfm_conv";
inline constexpr std::string_view kCommitLfmConvV4 = "commit_lfm_conv_v4";

// Vision tower and merger.
inline constexpr std::string_view kVisionLayerNorm = "vision_layer_norm";
inline constexpr std::string_view kVisionPatchify = "vision_patchify";
inline constexpr std::string_view kVisionPreparePositions = "vision_prepare_positions";
inline constexpr std::string_view kVisionQkvPrepare = "vision_qkv_prepare";
inline constexpr std::string_view kVisionAttention = "vision_attention";
inline constexpr std::string_view kVisionAttentionPack = "vision_attention_pack";
inline constexpr std::string_view kVisionGemmM64n128 = "vision_gemm_m64n128";
inline constexpr std::string_view kVisionGemmM64n128Residual = "vision_gemm_m64n128_residual";
inline constexpr std::string_view kVisionGemmM64n128GeluTanh = "vision_gemm_m64n128_gelu_tanh";
inline constexpr std::string_view kVisionGemmM32n256 = "vision_gemm_m32n256";
inline constexpr std::string_view kVisionGemmM32n256GeluErf = "vision_gemm_m32n256_gelu_erf";

// Paged attention stems (the format tag, split/store/gate role and layout
// suffix are appended by the PagedAttention helpers).
inline constexpr std::string_view kPrefillAttention = "prefill_attention";
inline constexpr std::string_view kPrefillAttentionQkv = "prefill_attention_qkv";
inline constexpr std::string_view kPrefillAttentionReduce = "prefill_attention_reduce";
inline constexpr std::string_view kVerifyAttention = "verify_attention";
inline constexpr std::string_view kVerifyAttentionQkv = "verify_attention_qkv";
inline constexpr std::string_view kVerifyAttentionReduce = "verify_attention_reduce";
inline constexpr std::string_view kVerifyAttentionReduceGate = "verify_attention_reduce_gate";
inline constexpr std::string_view kVerifyAttentionReduceGather = "verify_attention_reduce_gather";
inline constexpr std::string_view kVerifyAttentionGate = "verify_attention_gate";
inline constexpr std::string_view kVerifyTreeAttention = "verify_tree_attention";
inline constexpr std::string_view kVerifyTreeAttentionReduce = "verify_tree_attention_reduce";
inline constexpr std::string_view kVerifyTreeAttentionReduceGate = "verify_tree_attention_reduce_gate";
inline constexpr std::string_view kPrefillAttentionReduceCanvasGemmaH256 = "prefill_attention_reduce_canvas_gemma_h256";
inline constexpr std::string_view kPrefillAttentionReduceCanvasGemmaHd512 = "prefill_attention_reduce_canvas_gemma_hd512";

// Named instantiations of the paged attention family used by tests.
inline constexpr std::string_view kPrefillAttentionQ8Store = "prefill_attention_q8_store";
inline constexpr std::string_view kPrefillAttentionQ8Split = "prefill_attention_q8_split";
inline constexpr std::string_view kPrefillAttentionInt4Split = "prefill_attention_int4_split";
inline constexpr std::string_view kPrefillAttentionBf16Split = "prefill_attention_bf16_split";
inline constexpr std::string_view kPrefillAttentionFp8Split = "prefill_attention_fp8_split";
inline constexpr std::string_view kPrefillAttentionReduceGemmaHd512 = "prefill_attention_reduce_gemma_hd512";
inline constexpr std::string_view kPrefillAttentionQ8SplitCanvasSwaHd512 = "prefill_attention_q8_split_canvas_swa_hd512";
inline constexpr std::string_view kPrefillAttentionQ8SplitCanvasSwaHd512M2 = "prefill_attention_q8_split_canvas_swa_hd512_m2";
inline constexpr std::string_view kVerifyAttentionQ8Store = "verify_attention_q8_store";
inline constexpr std::string_view kVerifyAttentionQ8Split = "verify_attention_q8_split";
inline constexpr std::string_view kVerifyAttentionInt4Split = "verify_attention_int4_split";
inline constexpr std::string_view kVerifyAttentionBf16Split = "verify_attention_bf16_split";
inline constexpr std::string_view kVerifyAttentionFp8Split = "verify_attention_fp8_split";
inline constexpr std::string_view kVerifyAttentionReduceGemmaHd512 = "verify_attention_reduce_gemma_hd512";
inline constexpr std::string_view kVerifyAttentionQ8SplitSwaHd512 = "verify_attention_q8_split_swa_hd512";
inline constexpr std::string_view kVerifyAttentionQ8SplitSwaHd512M2 = "verify_attention_q8_split_swa_hd512_m2";

// Embeddings (stems take the format or hidden-width suffix).
inline constexpr std::string_view kGgufEmbedRotatedPq20 = "gguf_embed_rotated_pq20";
inline constexpr std::string_view kGgufEmbed = "gguf_embed_";
inline constexpr std::string_view kEmbeddingQ4H = "embedding_q4_h";
inline constexpr std::string_view kEmbeddingQ4tH = "embedding_q4t_h";
inline constexpr std::string_view kEmbeddingQ4ScaledH = "embedding_q4_scaled_h";
inline constexpr std::string_view kVerifyInputTokens = "verify_input_tokens";
inline constexpr std::string_view kVerifyInputTreeTokens = "verify_input_tree_tokens";
inline constexpr std::string_view kTreeCaptureGather = "tree_capture_gather";

// GGUF linear tiles (stems take format, row count and epilogue tags).
inline constexpr std::string_view kGgufRotate = "gguf_rotate";
inline constexpr std::string_view kGgufPackHalf = "gguf_pack_half";
inline constexpr std::string_view kDecodeLinearGgufPrepare = "decode_linear_gguf_prepare";
inline constexpr std::string_view kGgufDecode = "gguf_decode_";
inline constexpr std::string_view kGgufDecodeSg = "gguf_decode_sg_";
inline constexpr std::string_view kGgufDecodeFusedM = "gguf_decode_fused_m";
inline constexpr std::string_view kGgufDecodeSgFused = "gguf_decode_sg_fused";
inline constexpr std::string_view kGgufPrefill = "gguf_prefill_";
inline constexpr std::string_view kGgufFloat = "gguf_float_";
inline constexpr std::string_view kGgufFloatNa = "gguf_float_na_";
inline constexpr std::string_view kGgufDecodeQ4kM8AF32 = "gguf_decode_q4k_m8_a_f32";
inline constexpr std::string_view kGgufPrefillQ5kR = "gguf_prefill_q5k_r";

// Affine Q4/int8 linear tiles.
inline constexpr std::string_view kPrefillLinearI8Quant = "prefill_linear_i8_quant";
inline constexpr std::string_view kPrefillLinearI8N256 = "prefill_linear_i8_n256";
inline constexpr std::string_view kPrefillLinearI8N256Residual = "prefill_linear_i8_n256_residual";
inline constexpr std::string_view kPrefillLinearI8N256UpSiluSums = "prefill_linear_i8_n256_up_silu_sums";
inline constexpr std::string_view kDecodeLinearQ4Prepare = "decode_linear_q4_prepare";
inline constexpr std::string_view kPrefillLinearQ4Sums32 = "prefill_linear_q4_sums32";
inline constexpr std::string_view kPrefillLinearQ4N128 = "prefill_linear_q4_n128";
inline constexpr std::string_view kPrefillLinearQ4N256 = "prefill_linear_q4_n256";
inline constexpr std::string_view kPrefillLinearQ4N128Sg4 = "prefill_linear_q4_n128_sg4";
inline constexpr std::string_view kPrefillLinearQ4N128Residual = "prefill_linear_q4_n128_residual";
inline constexpr std::string_view kPrefillLinearQ4N256Residual = "prefill_linear_q4_n256_residual";
inline constexpr std::string_view kPrefillLinearQ4N128ResidualSg4 = "prefill_linear_q4_n128_residual_sg4";
inline constexpr std::string_view kPrefillLinearQ4N128UpSiluSumsSg4 = "prefill_linear_q4_n128_up_silu_sums_sg4";
inline constexpr std::string_view kPrefillLinearQ4N256UpSiluSums = "prefill_linear_q4_n256_up_silu_sums";
inline constexpr std::string_view kDecodeLinearQ4Sg = "decode_linear_q4_sg";
inline constexpr std::string_view kDecodeLinearQ4SgGateUp = "decode_linear_q4_sg_gate_up";
inline constexpr std::string_view kDecodeLinearQ4SgResidual = "decode_linear_q4_sg_residual";
inline constexpr std::string_view kDecodeLinearQ4N128 = "decode_linear_q4_n128";
inline constexpr std::string_view kDecodeLinearQ4N128M16 = "decode_linear_q4_n128_m16";
inline constexpr std::string_view kDecodeLinearQ4N128M24 = "decode_linear_q4_n128_m24";
inline constexpr std::string_view kDecodeLinearQ4N128M32 = "decode_linear_q4_n128_m32";
inline constexpr std::string_view kDecodeLinearQ4N128M48 = "decode_linear_q4_n128_m48";
inline constexpr std::string_view kDecodeLinearQ4N128M64 = "decode_linear_q4_n128_m64";
inline constexpr std::string_view kDecodeLinearQ4N128Paired = "decode_linear_q4_n128_paired";
inline constexpr std::string_view kDecodeLinearQ4N128M24Sg4 = "decode_linear_q4_n128_m24_sg4";
inline constexpr std::string_view kDecodeLinearQ4N128Residual = "decode_linear_q4_n128_residual";
inline constexpr std::string_view kDecodeLinearQ4N128ResidualM16 = "decode_linear_q4_n128_residual_m16";
inline constexpr std::string_view kDecodeLinearQ4N128ResidualM24 = "decode_linear_q4_n128_residual_m24";
inline constexpr std::string_view kDecodeLinearQ4N128ResidualM32 = "decode_linear_q4_n128_residual_m32";
inline constexpr std::string_view kDecodeLinearQ4N128ResidualM48 = "decode_linear_q4_n128_residual_m48";
inline constexpr std::string_view kDecodeLinearQ4N128ResidualM64 = "decode_linear_q4_n128_residual_m64";
inline constexpr std::string_view kDecodeLinearQ4N128ResidualPaired = "decode_linear_q4_n128_residual_paired";
inline constexpr std::string_view kDecodeLinearQ4N128ResidualM24Sg4 = "decode_linear_q4_n128_residual_m24_sg4";
inline constexpr std::string_view kDecodeLinearQ4N128Split = "decode_linear_q4_n128_split";
inline constexpr std::string_view kDecodeLinearQ4N128SplitM16 = "decode_linear_q4_n128_split_m16";
inline constexpr std::string_view kDecodeLinearQ4N128SplitM24 = "decode_linear_q4_n128_split_m24";
inline constexpr std::string_view kDecodeLinearQ4N128SplitM32 = "decode_linear_q4_n128_split_m32";
inline constexpr std::string_view kDecodeLinearQ4N128SplitM48 = "decode_linear_q4_n128_split_m48";
inline constexpr std::string_view kDecodeLinearQ4N128SplitM64 = "decode_linear_q4_n128_split_m64";
inline constexpr std::string_view kDecodeLinearQ4N128SplitResidual = "decode_linear_q4_n128_split_residual";
inline constexpr std::string_view kDecodeLinearQ4N128SplitResidualM16 = "decode_linear_q4_n128_split_residual_m16";
inline constexpr std::string_view kDecodeLinearQ4N128SplitResidualM24 = "decode_linear_q4_n128_split_residual_m24";
inline constexpr std::string_view kDecodeLinearQ4N128SplitResidualM32 = "decode_linear_q4_n128_split_residual_m32";
inline constexpr std::string_view kDecodeLinearQ4N128SplitResidualM48 = "decode_linear_q4_n128_split_residual_m48";
inline constexpr std::string_view kDecodeLinearQ4N128SplitResidualM64 = "decode_linear_q4_n128_split_residual_m64";
inline constexpr std::string_view kDecodeLinearQ4N128SplitUpSilu = "decode_linear_q4_n128_split_up_silu";
inline constexpr std::string_view kDecodeLinearQ4N128SplitUpSiluM16 = "decode_linear_q4_n128_split_up_silu_m16";
inline constexpr std::string_view kDecodeLinearQ4N128SplitUpSiluM24 = "decode_linear_q4_n128_split_up_silu_m24";
inline constexpr std::string_view kDecodeLinearQ4N128SplitUpSiluM32 = "decode_linear_q4_n128_split_up_silu_m32";
inline constexpr std::string_view kDecodeLinearQ4N128SplitUpSiluM48 = "decode_linear_q4_n128_split_up_silu_m48";
inline constexpr std::string_view kDecodeLinearQ4N128SplitUpSiluM64 = "decode_linear_q4_n128_split_up_silu_m64";
inline constexpr std::string_view kDecodeLinearQ4N256 = "decode_linear_q4_n256";
inline constexpr std::string_view kDecodeLinearQ4N256M16 = "decode_linear_q4_n256_m16";
inline constexpr std::string_view kDecodeLinearQ4N256M24 = "decode_linear_q4_n256_m24";
inline constexpr std::string_view kDecodeLinearQ4N256M32 = "decode_linear_q4_n256_m32";
inline constexpr std::string_view kDecodeLinearQ4N256M48 = "decode_linear_q4_n256_m48";
inline constexpr std::string_view kDecodeLinearQ4N256M64 = "decode_linear_q4_n256_m64";
inline constexpr std::string_view kDecodeLinearQ4N256M64Sg16 = "decode_linear_q4_n256_m64_sg16";
inline constexpr std::string_view kDecodeLinearQ4N256GateUp = "decode_linear_q4_n256_gate_up";
inline constexpr std::string_view kDecodeLinearQ4N256GateUpM16 = "decode_linear_q4_n256_gate_up_m16";
inline constexpr std::string_view kDecodeLinearQ4N256PairedSg4 = "decode_linear_q4_n256_paired_sg4";
inline constexpr std::string_view kDecodeLinearQ4N256UpSiluM24 = "decode_linear_q4_n256_up_silu_m24";
inline constexpr std::string_view kDecodeLinearQ4N256UpSiluM32 = "decode_linear_q4_n256_up_silu_m32";
inline constexpr std::string_view kDecodeLinearQ4N256UpSiluM48 = "decode_linear_q4_n256_up_silu_m48";
inline constexpr std::string_view kDecodeLinearQ4N256UpSiluM64 = "decode_linear_q4_n256_up_silu_m64";
inline constexpr std::string_view kDecodeLinearQ4N256UpSiluM64Sg16 = "decode_linear_q4_n256_up_silu_m64_sg16";

// Mixture of experts.
inline constexpr std::string_view kMoePrepareTable16 = "moe_prepare_table16";
inline constexpr std::string_view kMoeRouteSelectF32 = "moe_route_select_f32";
inline constexpr std::string_view kMoeRouteSelectSigmoid = "moe_route_select_sigmoid";
inline constexpr std::string_view kMoeRouteScoresQ8M8 = "moe_route_scores_q8_m8";
inline constexpr std::string_view kMoeRouteScoresQ8M32 = "moe_route_scores_q8_m32";
inline constexpr std::string_view kMoeRouteSelectQ8 = "moe_route_select_q8";
inline constexpr std::string_view kMoeRouteCap = "moe_route_cap";
inline constexpr std::string_view kMoeRouteGroupSigmoid = "moe_route_group_sigmoid";
inline constexpr std::string_view kMoeGroupRoutes = "moe_group_routes";
inline constexpr std::string_view kMoeGatherTable16 = "moe_gather_table16";
inline constexpr std::string_view kMoeGatherPacked = "moe_gather_packed";
inline constexpr std::string_view kMoeGatherRows = "moe_gather_rows";
inline constexpr std::string_view kMoeCombine = "moe_combine";
inline constexpr std::string_view kMoeRouteScoresGemma = "moe_route_scores_gemma";
inline constexpr std::string_view kMoeRouteScoresGemmaVec = "moe_route_scores_gemma_vec";
inline constexpr std::string_view kMoeRouteSelectGemma = "moe_route_select_gemma";
// Stems: "_m<rows>" or a register-tile suffix then "_a"/"_g" and the packed
// "_p"/native "_n" tag.
inline constexpr std::string_view kMoeExpertGgufM = "moe_expert_gguf_m";
inline constexpr std::string_view kMoeExpertGgufSg = "moe_expert_gguf_sg";
inline constexpr std::string_view kMoeExpertGateUpQ4M8 = "moe_expert_gate_up_q4_m8";
inline constexpr std::string_view kMoeExpertDownQ4M8 = "moe_expert_down_q4_m8";
inline constexpr std::string_view kMoeExpertGateUpQ4M8N128Sg4 = "moe_expert_gate_up_q4_m8_n128_sg4";
inline constexpr std::string_view kMoeExpertDownQ4M8N256Sg4 = "moe_expert_down_q4_m8_n256_sg4";
inline constexpr std::string_view kMoeExpertGateUpQ4M8Gelu = "moe_expert_gate_up_q4_m8_gelu";
inline constexpr std::string_view kMoeExpertGateUpQ4M8GeluN128Sg4 = "moe_expert_gate_up_q4_m8_gelu_n128_sg4";
inline constexpr std::string_view kPrefillMoeExpertQ4N256 = "prefill_moe_expert_q4_n256";
inline constexpr std::string_view kPrefillMoeExpertQ4N256M16 = "prefill_moe_expert_q4_n256_m16";
inline constexpr std::string_view kPrefillMoeExpertQ4N256M32 = "prefill_moe_expert_q4_n256_m32";
inline constexpr std::string_view kPrefillMoeExpertQ4N256Indirect = "prefill_moe_expert_q4_n256_indirect";
inline constexpr std::string_view kPrefillMoeExpertQ4N256IndirectM16 = "prefill_moe_expert_q4_n256_indirect_m16";
inline constexpr std::string_view kPrefillMoeExpertQ4N256IndirectM32 = "prefill_moe_expert_q4_n256_indirect_m32";
inline constexpr std::string_view kPrefillMoeExpertQ4N256UpSiluIndirect = "prefill_moe_expert_q4_n256_up_silu_indirect";
inline constexpr std::string_view kPrefillMoeExpertQ4N256UpSiluIndirectM16 = "prefill_moe_expert_q4_n256_up_silu_indirect_m16";
inline constexpr std::string_view kPrefillMoeExpertQ4N256UpSiluIndirectM32 = "prefill_moe_expert_q4_n256_up_silu_indirect_m32";
inline constexpr std::string_view kPrefillMoeExpertQ4N256UpGeluIndirectM16 = "prefill_moe_expert_q4_n256_up_gelu_indirect_m16";
inline constexpr std::string_view kPrefillMoeExpertQ4N256UpGeluIndirectM32 = "prefill_moe_expert_q4_n256_up_gelu_indirect_m32";

// ANE FFN split support (stems take the plane suffix and rotation tag).
inline constexpr std::string_view kAneFfnRotate = "ane_ffn_rotate";
inline constexpr std::string_view kAneFfnPack = "ane_ffn_pack";
inline constexpr std::string_view kAneFfnJoin = "ane_ffn_join";
inline constexpr std::string_view kAneFfnRowScale = "ane_ffn_row_scale";
inline constexpr std::string_view kAneFfnWeights = "ane_ffn_weights";

// Canvas (diffusion decoding).
inline constexpr std::string_view kCanvasUniformNoise = "canvas_uniform_noise";
inline constexpr std::string_view kCanvasLogitsScale = "canvas_logits_scale";
inline constexpr std::string_view kCanvasLogitSoftcap = "canvas_logit_softcap";
inline constexpr std::string_view kCanvasRowStats = "canvas_row_stats";
inline constexpr std::string_view kCanvasRowStatsFused = "canvas_row_stats_fused";
inline constexpr std::string_view kCanvasEntropyAccept = "canvas_entropy_accept";
inline constexpr std::string_view kCanvasSoftEmbedHistogram = "canvas_soft_embed_histogram";
inline constexpr std::string_view kCanvasSoftEmbedExact = "canvas_soft_embed_exact";
inline constexpr std::string_view kCanvasSoftEmbedTopk = "canvas_soft_embed_topk";
inline constexpr std::string_view kCanvasSelfCondition = "canvas_self_condition";

// Test/benchmark kernels and the deliberately-unresolved pipeline name.
inline constexpr std::string_view kTestCopyU32 = "test_copy_u32";
inline constexpr std::string_view kTestFillU32 = "test_fill_u32";
inline constexpr std::string_view kTestAddU32 = "test_add_u32";
inline constexpr std::string_view kAddressedWriteU32 = "addressed_write_u32";
inline constexpr std::string_view kAddressedCheckU32 = "addressed_check_u32";
inline constexpr std::string_view kDoesNotExist = "does_not_exist";
inline constexpr std::string_view kResidencyKick = "residency_kick";
inline constexpr std::string_view kGgufTestDequant = "gguf_test_dequant_";
inline constexpr std::string_view kGgufTestDequantMxfp4n = "gguf_test_dequant_mxfp4n";
inline constexpr std::string_view kMppAttentionThreadgroup = "mpp_attention_threadgroup";
inline constexpr std::string_view kMppAttentionSimdgroup = "mpp_attention_simdgroup";
inline constexpr std::string_view kRichengineFp8QuantizeKvPage = "richengine_fp8_quantize_kv_page";
inline constexpr std::string_view kRichengineFp8DequantizeKvPage = "richengine_fp8_dequantize_kv_page";
inline constexpr std::string_view kRichengineFp8GatherLogicalKvPage = "richengine_fp8_gather_logical_kv_page";
inline constexpr std::string_view kRichengineQ8QuantizeKvPage = "richengine_q8_quantize_kv_page";
inline constexpr std::string_view kRichengineQ8DequantizeKvPage = "richengine_q8_dequantize_kv_page";
inline constexpr std::string_view kRichengineQ8GatherLogicalKvPage = "richengine_q8_gather_logical_kv_page";
inline constexpr std::string_view kProbeInt4Qk = "probe_int4_qk";
inline constexpr std::string_view kProbeInt4PvNn = "probe_int4_pv_nn";
inline constexpr std::string_view kBenchMoeN128M32 = "bench_moe_n128_m32";
inline constexpr std::string_view kBenchMoeN64M32 = "bench_moe_n64_m32";
inline constexpr std::string_view kBenchMoeN128GateIndirectM32 = "bench_moe_n128_gate_indirect_m32";
inline constexpr std::string_view kBenchMoeN64GateIndirectM32 = "bench_moe_n64_gate_indirect_m32";
inline constexpr std::string_view kBenchMoeN128UpGeluIndirectM32 = "bench_moe_n128_up_gelu_indirect_m32";
inline constexpr std::string_view kBenchMoeN64UpGeluIndirectM32 = "bench_moe_n64_up_gelu_indirect_m32";
inline constexpr std::string_view kBenchMoeN256Sg4UpGeluIndirectM32 = "bench_moe_n256_sg4_up_gelu_indirect_m32";
inline constexpr std::string_view kBenchMoeN128Sg4UpGeluIndirectM32 = "bench_moe_n128_sg4_up_gelu_indirect_m32";
inline constexpr std::string_view kBenchMoeN256UpGeluIndirectM16 = "bench_moe_n256_up_gelu_indirect_m16";
inline constexpr std::string_view kBenchMoeN256UpGeluPipeM32 = "bench_moe_n256_up_gelu_pipe_m32";
inline constexpr std::string_view kBenchMoeUpGeluKsplit2M32 = "bench_moe_up_gelu_ksplit2_m32";
inline constexpr std::string_view kBenchMoeUpGeluKsplit4M32 = "bench_moe_up_gelu_ksplit4_m32";
inline constexpr std::string_view kBenchDenseQ4N256Pipe = "bench_dense_q4_n256_pipe";
inline constexpr std::string_view kBenchDenseQ4N256Sg4 = "bench_dense_q4_n256_sg4";
inline constexpr std::string_view kBenchDenseQ4N256M64Sg4 = "bench_dense_q4_n256_m64_sg4";

} // namespace richengine::ops
