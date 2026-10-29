# LFM3-Omni-2B implementation reference

Porting notes for a custom C++ inference engine.

**Notation:** `<|name|>` in this document denotes the literal special token formed by
`<`, `|`, name, `|`, `>` (e.g. `<|audio_start|>` is the token spelled `audio_start`
between `<|` and `|>` markers). The HF files in `hf/` contain the real strings.

Two reference implementations are mirrored:

- `llamacpp/` - the ggml port (llama.cpp master: `tools/mtmd/*`, `src/models/lfm3.cpp`,
  `conversion/lfm3.py` copied as `llamacpp/lfm3.py`). Tensor names cited as
  `v.blk.N.*`, `a.blk.N.*`, `a.gen.*` are GGUF tensor names.
- `hf/` - the HF remote code shipped in LiquidAI/LFM3-Omni-2B. Names cited as
  `model.vision_tower.*` etc. are the safetensors names.

Citations are `file:line` of the copies in this directory.

## 0. Model inventory (hf/config.json)

| part | source | dims |
|---|---|---|
| language model | `text_config` | 24 layers, hidden 2048, 16 q / 4 kv heads, head_dim 128, ffn 7680, vocab 128000, tied embeddings, conv_L_cache 7, sliding_window 256, use_sandwich_norm |
| vision tower | `vision_config` | 14 layers, hidden 1024, 16 q / 8 kv heads, head_dim 64, ffn 2560, patch 16, temporal_patch 2, spatial merge 2, axial rope theta 1e6 |
| audio encoder | `audio_encoder_config` | 128 mel, 3x stride-2 conv downsample (8x in time), 17-layer LFM2-style stack, hidden 768, 8 q / 4 kv heads, head_dim 96, ffn 2560, conv_L_cache 3, sliding_window 32 tokens, rope theta 1e6 |
| depth decoder | `depth_config` | 3 full_attention layers, hidden 1024, 32 q / 8 kv heads, head_dim 32, ffn 2816, no sandwich norm, bidirectional (MaskGIT) |
| codec decoder | `audio_tokenizer/config.json` | FSQ levels [8,7,6,6], n_q 12, codebook size prod(levels)=2016 (+1 end code -> 2017 ids), 8-layer LFM3 decoder hidden 512 (conv at 0,1,3,5,7; sliding attn at 2,4,6; window 30), STFT n_fft 1280, hop 320, win 1280, frame_stack 6, 24 kHz out |

Special tokens (hf/tokenizer.json added_tokens, hf/config.json):

| id | token | role |
|---|---|---|
| 124893 | `<|pad|>` | pad_token_id |
| 124894 | `<|startoftext|>` | bos_token_id |
| 124895 | `<|endoftext|>` | extra stop (live ASR) |
| 124899 | `<|im_start|>` | turn open (chat template) |
| 124900 | `<|im_end|>` | eos_token_id, turn close |
| 124901/124902 | think open / think close | empty think block opens omni TTS turns |
| 125009 | `<|vision_start|>` | wraps image/video/audio spans |
| 125010 | `<|vision_end|>` | closes those spans |
| 125012 | `<|audio_start|>` | audio_start_token_id; enters AUDIO mode |
| 125013 | `<|text_start|>` | interleaved text block marker |
| 125014 | `<|text_end|>` | text_end_token_id; closes interleaved text block / whole turn outside interleave; also in eos_token_id list |
| 125015 | `<|mixed_start|>` | mixed_start_token_id; enters INTERLEAVED mode |
| 125016 | `<|mixed_end|>` | live-ASR final-block marker |
| 125029 | `<|video|>` | video_token_id placeholder |
| 125030 | `<|image|>` | image_token_id placeholder |
| 125031 | `<|audio|>` | audio_token_id placeholder (encoder input) |
| 125032 | `<|audio_out|>` | audio_out_token_id placeholder (speech frame, in or out) |

Generation-relevant config (hf/config.json, hf/generation_config.json):

- `num_codebooks` = 12, `codebook_size` = 2017 (ids 0..2016; id 2016 = `audio_eos_code`, the end-of-audio frame).
- `maskgit_schedule` = [4,4,4]: 3 depth-decoder passes, 4 codebooks revealed per pass, random order.
- `interleaved_n_text` = 6, `interleaved_n_audio` = 12.
- Default sampling: text greedy; audio temperature 0.4, repetition_penalty 1.2, window 64 frames (generation_config.json:15-18; llama.cpp defaults match, `llamacpp/clip.cpp` LFM3A_GEN hparams ~line 2178).

## 1. Vision tower (Lfm3VlVisionModel)

HF: `hf/modeling_lfm3_vl.py:460-537`. ggml: `llamacpp/lfm3vl.cpp` (whole file).

### 1.1 Patch embed

- HF: `patchifier` is `nn.Linear(in_channels*temporal_patch*patch^2 -> hidden, bias=False)`,
  i.e. 3*2*16*16=1536 -> 1024 (`modeling_lfm3_vl.py:472-473`). The image processor
  pre-orders pixels as merge-block-major flattened patches (`image_processing_lfm3_vl.py:173-197`).
- ggml: two Conv2D kernels `v.patch_embd.weight` and `v.patch_embd.weight.1` (stride 16,
  no pad), one per temporal slice; `build_inp_temporal_merge()` (`llamacpp/clip.cpp:588-616`)
  returns `conv0(frame0)+conv1(frame1)` for a 2-frame batch, and `conv0(img)+conv1(img)`
  for a still image (the temporal pair is the same image twice, matching the HF
  expand in `image_processing_lfm3_vl.py:190`). Conversion splits the linear weight
  into the two kernels (`llamacpp/lfm3.py:234-245`).
- Then `v.embedding_norm.weight` (RMSNorm, eps 1e-5) is applied (`lfm3vl.cpp:31`,
  HF `embedding_norm` at `modeling_lfm3_vl.py:474,499`).

### 1.2 Encoder layers

`layer_types` (config.json:204-219): indices 0,1 = conv; 2 = attn; repeating
conv/conv/attn? No: `[c,c,a,c,a,c,a,c,a,c,a,c,a,c]` - conv at 0,1,3,5,7,9,11,13 (8
layers), full_attention at 2,4,6,8,10,12 (6 layers). ggml detects conv layers by
`layer.conv2d_in_proj_w != nullptr` (`lfm3vl.cpp:46`, `clip.cpp:2889`).

Every layer, pre-norm residual structure (`lfm3vl.cpp:43-146`, HF `modeling_lfm3_vl.py:366-386`):

```
h = RMSNorm(x, ln1)                    # operator_norm, eps 1e-5
y = conv_block(h) or attn_block(h)     # below
x = x + y
h = RMSNorm(x, ln2)                    # ffn_norm
x = x + FFN(h)                         # w2(silu(w1(x))*w3(x)), no bias
```

FFN tensor names: HF `feed_forward.{w1,w3,w2}` -> GGUF `v.blk.N.ffn_{gate,up,down}`.

Conv block (HF `Lfm3VlVisionConv2d` `modeling_lfm3_vl.py:308-351`; ggml `lfm3vl.cpp:53-90`):

```
bcx = in_proj(h)                # 1024 -> 3072, split into [B, C, X] thirds
gated = B * X                   # elementwise
gated -> 2D grid (H_p x W_p raster order)
conv = depthwise 3x3 conv2d(gated, padding=1, groups=1024)   # conv2d_spatial
out = out_proj(C * conv)        # 1024 -> 1024
```

GGUF names: `v.blk.N.conv2d_in_proj.weight` (3072x1024), `v.blk.N.conv2d_spatial.weight`
(depthwise 3x3, stored F32), `v.blk.N.conv2d_out_proj.weight` (`lfm3.py:250-258`,
`clip.cpp:2690-2692`).

Attention block (HF `Lfm3VlVisionAttention` `modeling_lfm3_vl.py:195-305`; ggml `lfm3vl.cpp:91-125`):

- GQA: q_proj 1024->16*64, k/v_proj 1024->8*64, all bias-free; out_proj 1024->1024.
- Per-head RMSNorm on q and k BEFORE rope (`q_layernorm`, `k_layernorm`, dim 64, eps 1e-5).
- Axial 2D RoPE, theta 1e6 (`modeling_lfm3_vl.py:86-134`): inv_freq has head_dim/4=16
  freqs computed over spatial_dim=head_dim/2=32; each patch's (h,w) coords each rotate
  half the pairs; final cos/sin = cat(freq_h, freq_w) repeated twice over the 64-dim
  head. In ggml this is `ggml_rope_multi(..., n_rot=d_head/2, mrope_sections={d_head/4,d_head/4,0,0}, GGML_ROPE_TYPE_VISION, theta=1e6)` (`lfm3vl.cpp:16,111-118`),
  positions supplied as 4 i32 channels [y, x, y, x] in raster order (`clip.cpp:5663-5681`).
- Attention is BIDIRECTIONAL (is_causal=False, no mask) per frame; frames attend
  separately (`modeling_lfm3_vl.py:207,260-301`). ggml `build_attn` called with no mask.
- scale = head_dim^-0.5 = 1/8.

No final/post layernorm after the 14 layers (`lfm3vl.cpp:148`).

### 1.3 Projector (mm.*)

HF `Lfm3OmniMultiModalProjector` (`modeling_lfm3_omni.py:349-371`), ggml `lfm3vl.cpp:150-178`:

1. LayerNorm(hidden 1024, eps=`projector_layer_norm_eps`=1e-6, WITH bias) per patch -
   `mm.input_norm.{weight,bias}`.
2. 2x2 spatial merge: group each 2x2 block of patches into one vector of 4096. HF does
   it implicitly: pixels were pre-grouped merge-block-major by the image processor, so
   `reshape(-1, 4096)` suffices (`modeling_lfm3_vl.py:533`). ggml keeps raster order
   inside the encoder and performs the interleave at the projector
   (`lfm3vl.cpp:156-171`): reshape `[C*2, Wp/2, Hp]` -> `[C*2, Wp/2, 2, B*Hp/2]` ->
   permute -> contiguous `[C, Hp*Wp]` merge-block order, then reshape
   `[4096, n_pos/4]`. Either way the result is the same set of 4096-vectors in
   block-raster order.
3. `linear_1` 4096->4096 (bias), GELU (HF `"gelu"` = exact/erf; ggml `FFN_GELU_ERF` at
   `lfm3vl.cpp:173-178`), `linear_2` 4096->2048 (bias) -> LM embedding space.
   GGUF: `mm.0.{weight,bias}` = linear_1, `mm.2.{weight,bias}` = linear_2 (`lfm3.py:261-263`).

### 1.4 LM tokens per image

`n_tokens = (Hp/2)*(Wp/2) = grid_h*grid_w/4` per frame (`modeling_lfm3_vl.py:617`,
ggml `clip_n_output_tokens_x/y` `clip.cpp:4680,4713`). Video: sum over frames
(t*h*w/4 per grid_thw row).

Image sizing (mtmd `llamacpp/mtmd-image.cpp:826-871`, HF `image_processing_lfm3_vl.py:50-75`):
smart_resize to multiples of patch*merge=32 px, clamped to [65536, 2097152] pixels
(images; video frames [4096, 25165824] with temporal rounding to 2). Normalize
(x/255 - 0.5)/0.5, bicubic resize (`processor_config.json`).

**Ordering caveat for the port:** HF `pixel_values` arrive pre-grouped 2x2-merge-block
order and the conv block un-interleaves to raster internally (`modeling_lfm3_vl.py:334-340`).
llama.cpp feeds plain raster order and interleaves at the projector instead. Pick one
convention; the raster one is simpler (no re-order before the conv layers).

## 2. Audio front end + encoder (Lfm3OmniEncoder)

### 2.1 Log-mel front end

HF `Lfm3OmniFeatureExtractor` (`hf/feature_extraction_lfm3_omni.py:32-199`), ggml
`mtmd_audio_preprocessor_lfm3a` (`llamacpp/mtmd-audio.cpp:1059-1111`):

- 16 kHz mono input. STFT: n_fft=400, win=400 (periodic Hann), hop=160, center=False
  (no padding; right-pad the signal so the last frame is whole: `pad_length`
  `feature_extraction_lfm3_omni.py:73-76`, ggml `mtmd-audio.cpp:1079-1084`).
- Power spectrum -> slaney mel filterbank, 128 bins, fmin 0, fmax 8000, slaney area
  norm (`feature_extraction_lfm3_omni.py:59-67`; ggml `fill_mel_filterbank_matrix`
  `mtmd-audio.cpp:35-119`).
- `log10(mel)` clamped to floor -12, then `(x+4)/4` (`feature_extraction_lfm3_omni.py:94-95`;
  ggml `mtmd-audio.cpp:1093,1105-1107`, floor 1e-12).
- Output `[n_mel=128, n_frames]`; one LM token per 8 mel frames
  (`num_encoder_tokens` `feature_extraction_lfm3_omni.py:78-79`; ggml `samples_per_row
  = 8 * hop = 1280` `mtmd.cpp:1899`). So 1 token = 80 ms of audio.

### 2.2 Conv downsampler (a.conv1d.*)

HF `Lfm3OmniDownsampler` (`modeling_lfm3_omni.py:374-408`), ggml `conv_down` in
`llamacpp/lfm3a.cpp:45-73`. Input `[1, n_mel=128, T]` treated as a 2D image
(time x freq), three stages of k=3, s=2, p=1 (p=1 only on the right/bottom in the
ggml streaming form: `ggml_pad_ext(x, 0,1,...)` + conv with pad (0,1) `lfm3a.cpp:51-53`):

1. `conv_in`: Conv2d 1->256, SiLU. GGUF `a.conv1d.0.{weight,bias}`.
2. `conv_mid`: depthwise Conv2d 256->256 (groups=256) then `lin_mid` 1x1 Conv2d 256->256, SiLU.
   GGUF `a.conv1d.2.*`, `a.conv1d.3.*`.
3. `conv_end`: depthwise Conv2d 256->256 then `lin_end` 1x1 Conv2d 256->256, SiLU.
   GGUF `a.conv1d.5.*`, `a.conv1d.6.*`.
4. flatten freq*ch (128/8 * 256 = 4096) -> `lin_out` Conv1d 4096->768 k=1.
   GGUF `a.pre_encode.out.{weight,bias}` (`lfm3.py:520-527`, index map at `lfm3.py:314`).

Time is downsampled 8x: `n_pos = ceil(T/8)` (`output_lengths` `modeling_lfm3_omni.py:407-408`).

### 2.3 Encoder trunk (a.blk.N.*)

17 layers, type pattern `[c,c,a]*5 + [a,a]` - conv at 0,1,3,4,6,7,9,10,12,13,
attention at 2,5,8,11,14,15,16 (config.json:37-55). Same pre-norm residual skeleton
as the vision tower, all RMSNorm eps 1e-5 (`lfm3a.cpp:93-169`).

Attention layers (full_attention type but the shared mask is a sliding-window causal
mask of width `sliding_window=32` tokens, `modeling_lfm3_omni.py:425-438`; ggml
`build_inp_attn_mask(prefix + n_pos, n_pos)` `lfm3a.cpp:89`):

- q/k/v proj, GQA 8 q / 4 kv heads, head_dim 96; per-head q/k RMSNorm.
- NEOX rope over the FULL head_dim (n_rot = d_head), theta 1e6 (`lfm3a.cpp:134-137`).
- No per-head sigmoid gate here (unlike the LM): `build_attn(layer.o_w, ...)`
  `lfm3a.cpp:150`.
- ggml names: `a.blk.N.ln1`, `a.blk.N.attn_{q,k,v,q_norm,k_norm,out}`,
  `a.blk.N.ln2`, `a.blk.N.ffn_{gate,up,down}` (tensor-mapping excerpt
  `llamacpp/gguf/tensor-mapping-lfm-excerpt.txt:222-293`).

Short-conv layers (`lfm3a.cpp:99-121`; same algorithm as the LM, section 3.2):

```
bcx = conv_pw1(h)              # 768 -> 2304, split [B, C, X]
bx   = B * X                   # transposed to [n_pos, 768]
conv = causal depthwise conv1d over [state | bx], kernel width conv_L_cache=3 (a.blk.N.conv_dw)
y    = conv_pw2(C * conv)      # conv_pw1/conv_dw/conv_pw2 names in clip-impl.h:363-366
```

Final norm `a.post_ln.weight` (encoder.lfm.lfm.embedding_norm), then projector
(`lfm3a.cpp:171-177`, HF `Lfm3OmniProjector` `modeling_lfm3_omni.py:498-509`):

```
out = linear_2(gelu(linear_1(LayerNorm(h, eps 1e-5))))   # 768 -> 2048 -> GELU -> 2048
```

GGUF: `mm.a.mlp.0.*` (layer_norm), `mm.a.mlp.1.*` (linear_1), `mm.a.mlp.3.*`
(linear_2) - the HF renames map `audio_projector.{layer_norm,linear_1,linear_2}` onto
`mlp.model.{0,1,3}` (`llamacpp/lfm3.py:589-591`, `constants.py` A_MMPROJ = `mm.a.mlp.{bid}`).

### 2.4 Streaming encoder state

`list_lfm3a_state_slots` (`lfm3a.cpp:5-27`): per call the graph consumes a chunk of
mel rows and reads/writes named state tensors - `pos`, `ds0/ds1/ds2` (last input
column of each downsample conv), `conv_i` (d_conv taps per conv layer), `k_i`/`v_i`
(last 31 positions of the sliding window per attention layer). A zero state encodes a
whole clip identically to the offline path; HF equivalent `Lfm3OmniEncoderStreamingCache`
/`stream()` `modeling_lfm3_omni.py:104-125,441-495` (7 frames lookahead per token;
`is_final` flushes the tail). llama.cpp streaming driver: `mtmd_audio_stream_lfm3a`
(`mtmd.cpp:3454-3485,3797-4010`).

## 3. Language model (Lfm3Model, lfm3)

HF `hf/modeling_lfm3.py`. ggml `llamacpp/lfm3.cpp` (graph), `llamacpp/lfm-base.cpp`
(shortconv + ffn helpers), memory in `llamacpp/llama-memory-hybrid-iswa.h`.

### 3.1 Layer pattern

`layer_types` length 24 (config.json:142-167): conv at 0,1,3,5,7,9,11,13,15,17,19,21,23
(13 layers); full_attention at 2,6,10,14,18,22 (6); sliding_attention at 4,8,12,16,20
(5). Detected in gguf by `n_head_kv(il)==0` marking conv layers (`lfm3.cpp:8-10`) and
the sliding-window pattern array (`lfm3.cpp:13`).

Per layer (`lfm3.cpp:159-203`, HF `Lfm3DecoderLayer` `modeling_lfm3.py:445-499`):

```
h = RMSNorm(x, operator_norm)
y = attn(h) or shortconv(h)
y = RMSNorm(y, post_operator_norm)        # use_sandwich_norm=true in this checkpoint
x = x + y
h = RMSNorm(x, ffn_norm)
f = w2(silu(w1(h)) * w3(h))               # ffn_gate=w1, ffn_up=w3, ffn_down=w2
f = RMSNorm(f, post_ffn_norm)             # sandwich norm
x = x + f
```

Final `model.embedding_norm` (RMSNorm) then `lm_head` = tied `embed_tokens`
(`lfm3.cpp:36-41,206-213`). All norms RMSNorm eps 1e-5 (`norm_eps`).

GGUF names: `blk.N.attn_norm` (operator_norm), `blk.N.attn_post_norm`
(post_operator_norm), `blk.N.ffn_norm`, `blk.N.ffn_post_norm`, `blk.N.attn_{q,k,v,output}.weight`,
`blk.N.attn_{q,k}_norm`, `blk.N.attn_gate` (wqkv_gate), `blk.N.ffn_{gate,up,down}`,
`blk.N.shortconv.{in_proj,out_proj}`, `blk.N.shortconv.conv` - mapping table
`llamacpp/gguf/tensor-mapping-lfm-excerpt.txt` + `lfm3.cpp:45-88`.

### 3.2 Short-conv operator (conv layers)

HF `Lfm3ShortConv` (`modeling_lfm3.py:331-442`), ggml `build_lfm_shortconv`
(`lfm-base.cpp:9-112`):

```
bcx = in_proj(h)                     # 2048 -> 6144, split [B, C, X]
bx   = B * X                         # (transpose to [n_tokens, emb])
bx   = concat(conv_state, bx)        # conv_state = last d_conv=6 inputs, per sequence
conv = depthwise causal conv1d(bx, kernel=L=7, no bias)
y    = out_proj(C * conv)
```

- State is the last `L-1=6` vectors of `bx` per layer (`lfm-base.cpp:44-72`); decode
  step keeps a rolling window (mamba-style conv cache). For non-causal (d1 decision
  graph) the window is centered (`lfm-base.cpp:48-56`) - not needed for omni.
- `ggml_ssm_conv` semantics: `out[t] = sum_k w[k] * bx[t - (L-1) + k]` per channel.

### 3.3 Attention operator

HF `Lfm3Attention` (`modeling_lfm3.py:213-273`), ggml `build_attn_block`
(`lfm3.cpp:101-145`):

- q_proj 2048->2048 (16x128), k/v_proj 2048->512 (4x128), all bias-free.
- Per-head RMSNorm on q,k (head_dim 128) BEFORE rope - `attn_q_norm`, `attn_k_norm`.
- RoPE NEOX, in fp32: full_attention layers: theta 1e6, PARTIAL rotary factor 0.5
  (n_rot=64 of 128; the unrotated tail is encoded as rope_freqs = [1]*32 + [1e30]*32
  generated at conversion, `lfm3.py:101-112`, applied via `ggml_rope_ext` with
  `rope_factors` `lfm3.cpp:113-125`). sliding_attention layers: theta 1e4, full
  rotation (config.json:175-185; `hparams.rope_freq_base_train_swa` `lfm3.cpp:20`).
- Causal attention; sliding layers additionally limited to window 256 (`swa_type =
  LLAMA_SWA_TYPE_STANDARD` `lfm3.cpp:17`).
- scale = 1/sqrt(128).
- **Per-head sigmoid output gate:** `gate = sigmoid(wqkv_gate(h))` shape [n_head=16]
  per token; applied to the attention output per head before `o_proj`
  (`lfm3.cpp:132-139`, HF `modeling_lfm3.py:270-272`). GGUF `blk.N.attn_gate.weight`
  is [2048 x 16].

### 3.4 Memory layout

`llama_memory_hybrid_iswa` (`llamacpp/llama-memory-hybrid-iswa.h`): recurrent conv
state (d_conv * n_embd per conv layer, plus rollback slots for the n_rs_seq feature),
full KV for full_attention layers, sliding-window KV (256) for sliding layers. The
`decision=true` graph variant (`lfm3.cpp:91-96,152-154`) is a no-cache single-batch
prefill used by the d1 models - ignore for omni.

### 3.5 Embedding injection

The omni model scatters computed embeddings over placeholder token positions
(`modeling_lfm3_omni.py:685-717`):

- `<|image|>`/`<|video|>` positions <- projected vision features (section 1).
- `<|audio|>` positions <- projected encoder features (section 2).
- `<|audio_out|>` positions <- `get_audio_out_embeds(codes)`: sum over the 12
  codebook rows of `codebook_embedding` (a [12*2017, 2048] table; codes get
  `+ codebook*2017` offsets, `modeling_lfm3_omni.py:576-597`). GGUF
  `a.gen.code.out_embd` (`lfm3.py:552-555`; used identically as the feedback
  embedding in generation, section 4).

## 4. Speech-code head: depth decoder (MaskGIT, a.gen.code.*)

HF `Lfm3OmniDepthDecoder` (`modeling_lfm3_omni.py:215-282`), ggml `build_gen_code`
(`llamacpp/lfm3a-gen.cpp:113-170`).

Purpose: given the LM hidden state at the position preceding a speech frame, predict
the 12 codebook codes of that frame. Non-autoregressive: codebooks revealed in a
random order over `maskgit_schedule` passes.

### 4.1 GEN_CODE graph (one depthformer pass)

Inputs: `h_state` [2048] (LM hidden state), `inp_codes` [12] i32 (known codes,
-1 = masked), `inp_mask` [12] f32 (1 = revealed).

```
proj   = depth_linear(h_state)                      # 2048 -> 12*1024 -> [1024 x 12] slots
slot_i = proj_i + ( mask_i ? embd_i[code_i] : mask_embedding )   # per-codebook embedding table
cur    = 3x bidirectional LFM3 layer (full attention, NO mask, NO state)
cur    = RMSNorm(cur, a.gen.code.output_norm)       # trunk final norm
cur    = RMSNorm(cur, a.gen.code.head_norm)         # per-codebook norm: weight is [1024 x 12]
logits = embd[cur]                                  # tied: logits_i = embd_i @ head_norm(cur_i) -> [2017 x 12]
```

Details:

- Attention layers are the LFM3 kind WITH the per-head sigmoid gate
  (`a.gen.code.blk.N.attn_gate`, `lfm3a-gen.cpp:83-87`) but WITHOUT sandwich norms
  (depth_config `use_sandwich_norm` false) and no sliding window (all
  `full_attention`, `is_causal=False` `modeling_lfm3_omni.py:274-281`). Rope NEOX
  theta 1e6, full head_dim=32 rotation, positions = arange(12).
- The per-codebook embedding tables `depth_embeddings.{i}.embedding` (each
  [2017, 1024]) are merged into ONE 3D tensor `a.gen.code.embd` [1024 x 2017 x 12]
  at conversion (`lfm3.py:558-568`); each table doubles as its own output head
  (HF checks `to_logits.weight == embedding.weight`, `lfm3.py:569-572`). Code rows
  are addressed as `code + codebook*2017` (`lfm3a-gen.cpp:118-119,145`).
- `a.gen.code.head_norm` merges the 12 `embedding_norm` weights into [1024 x 12] so
  each slot is normed by its own codebook's norm (`lfm3.py:565-568`,
  `lfm3a-gen.cpp:162`).
- The feedback embedding `fb = sum_i out_embd[code_i + i*2017]` [2048] is produced
  by the same graph from `a.gen.code.out_embd` (`lfm3a-gen.cpp:124-127`) and re-fed
  into the LM as the `<|audio_out|>` slot's embedding for the next decode step.
  When all 12 codes are known (`all_known`), the graph returns only `fb`
  (`lfm3a-gen.cpp:128-130`, dispatch `clip.cpp:1204-1219`).

### 4.2 Sampling one frame

HF `sample_audio_frame` (`modeling_lfm3_omni.py:1063-1100`), ggml `sample_frame`
(`mtmd-helper-gen.cpp:1530-1580`):

1. Shuffle a permutation of the 12 codebooks (mt19937 in llama.cpp; torch.randperm in HF).
2. For each pick of 4 codebooks in the shuffled order (schedule [4,4,4]): run
   GEN_CODE; for each picked codebook apply per-codebook repetition penalty then
   sample (temp 0.4, no top-k by default) or argmax; write the code, mark revealed.
3. If ANY code equals the end code 2016: the frame is the end-of-audio frame; all
   codes are set to 2016 and generation leaves audio mode (`mtmd-helper-gen.cpp:1561-1569`,
   HF `modeling_lfm3_omni.py:1734-1744`).
4. Run GEN_CODE once more with all codes known to get `fb`, embed it as the next
   LM input embedding (position incremented like a text token).
5. Repetition penalty: for each codebook, over the last `audio_repetition_window`
   (64) frames, unique codes get `l /= penalty` if `l > 0` else `l *= penalty`
   (CTRL-style, `mtmd-helper-gen.cpp:1582-1597`).

## 5. Codes -> waveform: liquid codec decoder (a.gen.wav.*)

HF `modeling_liquid_codec.py` (decoder part), ggml `build_gen_wav`
(`llamacpp/lfm3a-gen.cpp:172-251`) + host-side ISTFT (`mtmd.cpp:2358-2418`,
`mtmd-audio.cpp:1539-1640`).

Only the DECODER half of LiquidCodecModel ships in the GGUF; the STFT encoder is not
needed at inference (but is needed if you also port prompt-side `audio_out` clip
encoding - see 7.3).

### 5.1 GEN_WAV graph (one call per window of frames)

Inputs: `inp_codes` [12 * n_frames] frame-major codes, plus the streaming state
slots (`pos`, `conv_i`, `k_i`, `v_i` - `lfm3a-gen.cpp:12-29`).

1. **FSQ decode:** `quant.first.codebook` is a precomputed [2016, 4] table of the
   FSQ level values `((idx // basis) % levels - half) / half`, built at conversion
   (`lfm3.py:481-487`); `get_rows` gives [4 x 12*n_frames] -> reshape [48, n_frames].
2. `quant.first.out_proj` (48 -> 256, weight+bias = `quantizer.project_out`) then
   `tfm.in_proj` (256 -> 512, = `from_quantizer`) (`lfm3a-gen.cpp:194-198`,
   `lfm3.py:489-494`).
3. **Nearest-exact upsample x6:** every codec frame becomes 6 decoder positions
   (75 Hz STFT frames, `frame_stack=6`; `lfm3a-gen.cpp:200-202`, HF
   `modeling_liquid_codec.py:222-224`).
4. **8-layer LFM3 decoder**, hidden 512: conv at 0,1,3,5,7; sliding_attention at
   2,4,6 (window 30 positions, i.e. `prefix=29`); WITH sandwich norms
   (`post_operator_norm`, `post_ffn_norm` - `use_sandwich_norm=True` in
   `lfm_config`, `configuration_liquid_codec.py:104-131`); attention has q/k RMSNorm,
   per-head sigmoid gate (`a.gen.wav.tfm.blk.N.attn_gate`), NEOX rope theta 1e6 full
   head_dim=64; conv layers are the standard LFM short-conv (conv_L_cache=7 taps).
   Layer tensor names: `a.gen.wav.tfm.blk.N.{ln1,attn_post_norm,ln2,ffn_post_norm,
   attn_{q,k,v,q_norm,k_norm,out,gate},conv_{pw1,dw,pw2},ffn_{gate,up,down}}`
   (map `lfm3.py:331-349`).
5. `a.gen.wav.tfm.output_norm` (RMSNorm) then `tfm.out_proj` 512 -> 2*641
   (`decoder.lin`, `lfm3a-gen.cpp:238-240`): [log_magnitude | angle] per STFT frame.
6. State out: `pos += n_pos`, conv taps and last-29 k/v per layer are written back
   (`lfm3a-gen.cpp:204-236`).

### 5.2 Host-side ISTFT

`mtmd_gen_wav_lfm3a` (`mtmd.cpp:2358-2418`) + `mtmd_audio_streaming_istft`
(`mtmd-audio.cpp:1542-1640`), matching HF `LiquidCodecISTFT`
(`modeling_liquid_codec.py:192-207`):

- Per STFT column: `mag = exp(log_mag)`, complex bin = `mag*(cos a, sin a)`,
  irfft(1280) -> * periodic Hann(1280), overlap-add hop 320, normalize by the
  window-square envelope, drop the first (1280-320)/2 = 480 samples once
  (`padding_to_remove`, `mtmd-audio.cpp:1548,1603-1605`).
- On flush (end of audio segment or `n_codes==0` call): drain the overlap buffer and
  drop the last 480 samples (`mtmd.cpp:2369-2373`). HF equivalent: trim of
  `(win-hop)/2` both sides, `modeling_liquid_codec.py:324,342-357`.
- Output: 24 kHz float PCM; each codec frame yields 6*320 = 1920 samples.

## 6. Generation state machine

HF `_generate` (`modeling_lfm3_omni.py:1457-1818`), ggml `lfm3a_gen_audio_pipeline`
(`mtmd-helper-gen.cpp:1273-1718`).

Modes: TEXT, AUDIO, INTERLEAVED. The last prompt token picks the start mode:
`<|audio_start|>`/`<|audio_out|>` -> AUDIO, `<|mixed_start|>` -> INTERLEAVED, else
TEXT (`modeling_lfm3_omni.py:1595-1603`, `mtmd-helper-gen.cpp:1332-1335`).

Per step:

- **TEXT slot:** sample/argmax from `lm_head(hidden)` (lm_head = tied embed_tokens).
  `<|audio_start|>` switches to AUDIO; `<|mixed_start|>` enters INTERLEAVED with
  counters reset (6 text / 12 audio); `<|text_end|>` inside INTERLEAVED ends the
  text half (`text_done`, remaining audio runs to the end-of-audio frame), outside
  it ends the turn; `<|im_end|>` always ends the turn. Suppressed in text mode:
  all placeholder ids, plus `<|mixed_start|>`/`<|audio_start|>` when TEXT mode was
  forced (`modeling_lfm3_omni.py:1585-1593`).
- **AUDIO slot:** `sample_audio_frame` (section 4.2) from the current LM hidden
  state; a non-end frame is appended to the audio history, decoded to PCM through
  GEN_WAV (streaming or at the end), and its feedback embedding `fb` is the next LM
  input embedding. An end-of-audio frame flushes the codec tail and returns to TEXT.
- **INTERLEAVED bookkeeping:** every slot decrements `slots_left`; at 0 the block
  flips text<->audio and reloads 6 or 12 (`mtmd-helper-gen.cpp:1452-1456`,
  `modeling_lfm3_omni.py:1782-1786`).
- Text repetition penalty window 32 tokens (`text_repetition_window`); audio window
  64 frames.
- Live ASR variant: `transcribe_live` (`modeling_lfm3_omni.py:1189-1317`) feeds
  encoder tokens inside the assistant turn in chunks of 12, then greedily decodes
  text until `<|text_end|>`; llama.cpp equivalent `step_live` (`mtmd.cpp:3994-4010`).

## 7. Tensor name map (HF -> GGUF -> your engine)

Authoritative: `llamacpp/lfm3.py` (`_audio_renames` `:586-595`, `_lfm3vl_tensors`
`:233-266`, `_codec_tensors` `:465-503`, `_depthformer_tensor` `:536-573`) and
`llamacpp/gguf/tensor-mapping-lfm-excerpt.txt`.

### 7.1 Text model (prefix `model.language_model.` stripped by `lfm.` removal `lfm3.py:34`)

| HF | GGUF |
|---|---|
| embed_tokens.weight | token_embd.weight |
| layers.N.operator_norm | blk.N.attn_norm |
| layers.N.post_operator_norm | blk.N.attn_post_norm |
| layers.N.self_attn.{q,k,v}_proj | blk.N.attn_{q,k,v} |
| layers.N.self_attn.{q_norm,k_norm} | blk.N.attn_{q,k}_norm |
| layers.N.self_attn.gate_proj | blk.N.attn_gate |
| layers.N.self_attn.o_proj | blk.N.attn_output |
| layers.N.conv.{in_proj,out_proj} | blk.N.shortconv.{in_proj,out_proj} |
| layers.N.conv.conv (squeeze dim1) | blk.N.shortconv.conv [7 x 2048, depthwise taps] |
| layers.N.ffn_norm / post_ffn_norm | blk.N.ffn_norm / blk.N.ffn_post_norm |
| layers.N.feed_forward.{w1,w3,w2} | blk.N.ffn_{gate,up,down} |
| embedding_norm | output_norm |
| lm_head | tied to token_embd (output.weight optional) |
| rope_freqs (synthetic) | blk.N.rope_freqs, full-attn layers only, [1]*32+[1e30]*32 |

### 7.2 Vision (mmproj file)

| HF | GGUF |
|---|---|
| vision_tower.patchifier.weight | v.patch_embd.weight + v.patch_embd.weight.1 (temporal split) |
| vision_tower.embedding_norm | v.embedding_norm.weight |
| layers.N.operator_norm / ffn_norm | v.blk.N.ln1 / v.blk.N.ln2 |
| self_attn.{q,k,v,out}_proj | v.blk.N.attn_{q,k,v,out} |
| self_attn.{q,k}_layernorm | v.blk.N.attn_{q,k}_norm |
| conv.{in_proj,spatial_conv,out_proj} | v.blk.N.conv2d_{in_proj,spatial,out_proj} |
| feed_forward.{w1,w3,w2} | v.blk.N.ffn_{gate,up,down} |
| multi_modal_projector.{layer_norm,linear_1,linear_2} | mm.input_norm, mm.0, mm.2 |

### 7.3 Audio encoder + projector + speech heads

| HF | GGUF |
|---|---|
| audio_encoder.downsample.{conv_in,conv_mid,lin_mid,conv_end,lin_end} | a.conv1d.{0,2,3,5,6} |
| audio_encoder.downsample.lin_out | a.pre_encode.out |
| audio_encoder.lfm.layers.N.* | a.blk.N.* (ln1, attn_*, conv_{pw1,dw,pw2}, ln2, ffn_*) |
| audio_encoder.lfm.embedding_norm | a.post_ln |
| audio_projector.{layer_norm,linear_1,linear_2} | mm.a.mlp.{0,1,3} |
| codebook_embedding.weight [12*2017, 2048] | a.gen.code.out_embd |
| depth_decoder.depth_linear | a.gen.code.proj_in |
| depth_decoder.mask_embedding | a.gen.code.mask_embd |
| depth_decoder.model.embedding_norm | a.gen.code.output_norm |
| depth_decoder.model.layers.N.* | a.gen.code.blk.N.* (+ attn_gate) |
| depth_decoder.depth_embeddings.{i}.embedding | a.gen.code.embd (merged [1024,2017,12]) |
| depth_decoder.depth_embeddings.{i}.embedding_norm | a.gen.code.head_norm (merged [1024,12]) |
| audio_tokenizer decoder.lfm.layers.N.* | a.gen.wav.tfm.blk.N.* |
| audio_tokenizer decoder.lfm.embedding_norm | a.gen.wav.tfm.output_norm |
| audio_tokenizer decoder.lin | a.gen.wav.tfm.out_proj |
| audio_tokenizer from_quantizer | a.gen.wav.tfm.in_proj |
| audio_tokenizer quantizer.project_out | a.gen.wav.quant.first.out_proj |
| FSQ level table (synthetic [2016,4]) | a.gen.wav.quant.first.codebook |

Not exported to GGUF (port only if needed): codec ENCODER (stft, frontend,
encoder lfm, to_quantizer, quantizer.project_in) - used only when the prompt
carries `audio_out` clips that must be encoded to codes (`processing_lfm3_omni.py:284-306`).

## 8. Porting checklist / gotchas

1. **Three LFM variants, one block builder.** LM (sandwich norms + sigmoid gate +
   mixed conv/full/sliding + partial rope), vision (no gate, 2D axial rope, depthwise
   2D conv, bidirectional), audio encoder (no gate, NEOX rope, 1D short-conv,
   sliding causal), depth decoder (gate, no sandwich, bidirectional), codec decoder
   (gate + sandwich + 1D short-conv + sliding causal). `build_lfm_layer` /
   `build_lfm_attn` / `build_lfm_conv` in `lfm3a-gen.cpp:32-111` is the cleanest
   single parametrized reference.
2. RoPE: LM full-attn = NEOX theta 1e6 on first 64 of 128 dims (zero-tail partial);
   LM sliding = NEOX theta 1e4 full 128; audio enc / depth / codec = NEOX theta 1e6
   full head_dim; vision = axial 2D (head_dim/4 per axis, theta 1e6, no scaling).
3. Norms: everything RMSNorm eps 1e-5 EXCEPT projector input LayerNorms (vision
   eps 1e-6 with bias; audio projector eps 1e-5 with bias). The depth-decoder head
   norm applies per codebook column. HF note: `Lfm3OmniRMSNorm` multiplies weight
   INSIDE fp32 (`(x*w).to(dtype)` vs `w*x.to(dtype)`) - matters for bf16 parity,
   `modeling_lfm3_omni.py:206-212`.
4. Patch order: keep raster order through the vision encoder (llama.cpp way); do
   the 2x2 interleave at the projector. HF merges blocks before the tower instead.
5. Conv order conventions: `in_proj` output splits `[B, C, X]`; gate is `B*X` into
   the conv and `C*conv` out (NOT SiLU gated; B and C are linear gates).
6. Attention gates: sigmoid per-head scalar gate on attn out (LM, depth, codec) -
   multiply attn output [head_dim, head, pos] by gate [1, head, pos] BEFORE o_proj.
7. `<|audio_out|>` embedding = row-summed 12-codebook table (same tensor used as
   depth-decoder feedback embedding). In llama.cpp speech frames in the PROMPT go
   through `get_audio_out_embeds` too.
8. Codec end-of-audio: code 2016 in ANY codebook terminates speech; emit all-2016
   frame for the feedback embedding only when continuing (`mtmd-helper-gen.cpp:1561-1569`).
9. Codec decode is streaming: SWA-30 over 75 Hz positions, conv taps, and ISTFT
   overlap state carry across calls; `n_codes==0` flushes the ISTFT tail.
10. Mel pipeline: center=False STFT, right-pad to whole frames; log10 floor -12,
    then (x+4)/4; 8 mel frames per LM token.
11. Quantization notes from conversion (`lfm3.py:225-231,453-459`): patch_embd F16,
    conv2d_spatial and conv_dw kernels F32, all a.conv1d and a.gen.wav.quant.* F32,
    rest of a.gen.wav.* F16 (polar STFT output degrades under quantization).
12. Chat template: `hf/chat_template.jinja` - media spans are
    `<|vision_start|><|image|><|vision_end|>` (and same wrapper around
    `<|audio|>`); assistant turns open with an empty think block for omni TTS
    (`llamacpp/lfm3.py:54-64` tts template, `_asr_template` `:39-43`). Tool calls
    use `tool_call`/`function=`/`parameter=` tags (template lines ~42-58).

## 9. Asked questions

### b. Audio-in encoder forward pass

Compact restatement of section 2, walking one clip through both implementations.

1. **Log-mel front end.** HF `Lfm3OmniFeatureExtractor` defaults: `feature_size=128`,
   `sampling_rate=16000`, `n_fft=400`, `hop_length=160`, `win_length=400`,
   `log_floor=-12`, `normalization_offset=4.0`, `normalization_divisor=4.0`,
   `encoder_stride=8` (`hf/feature_extraction_lfm3_omni.py:37-58`). `_log_mel` does
   `stft(center=False, periodic Hann 400)` -> power -> slaney mel bank (128 bins,
   0..8000 Hz, `feature_extraction_lfm3_omni.py:59-67`) -> `log10` clamped at -12
   -> `(x+4)/4` (`:81-95`). `num_encoder_tokens = ceil(n_frames/8)`
   (`:78-79`): 8 mel frames x 10 ms = **one feature per 80 ms** of audio. Signal is
   right-padded so the last frame is whole (`pad_length`, `:73-76`).
   llama.cpp: `mtmd_audio_preprocessor_lfm3a` builds the same filterbank/window
   from hparams fixed at `16000/400/400/160` (`llamacpp/clip.cpp:2300-2305`),
   right-pads identically (`llamacpp/mtmd-audio.cpp:1079-1084`), applies
   `mel_floor=1e-12` ("log10 floor of -12", `mtmd-audio.cpp:1093`) and the same
   `(v+4)/4` (`mtmd-audio.cpp:1105-1107`). The converter refuses any other
   front end (`llamacpp/lfm3.py:365-367,405-408`).

2. **`a.conv1d.*` downsampler** (8x in time; input `[128 mel, T]` as a 2D
   time x freq plane). HF `Lfm3OmniDownsampler`
   (`hf/modeling_lfm3_omni.py:374-404`): all kernels are `k=3, s=2, p=1` conv2d
   except the two 1x1s -
   `conv_in` 1->256, SiLU;
   `conv_mid` depthwise 256->256 (groups=256) + `lin_mid` 1x1 256->256, SiLU;
   `conv_end` depthwise 256->256 + `lin_end` 1x1 256->256, SiLU;
   then `lin_out` Conv1d `(128/8)*256=4096 -> 768`, k=1.
   llama.cpp loads `a.conv1d.{0,2,3,5,6}` = {conv_in, conv_mid, lin_mid, conv_end,
   lin_end} (`llamacpp/clip.cpp:3958-3961`; name `a.conv1d.%d` in
   `llamacpp/clip-impl.h:214`; index map `llamacpp/lfm3.py:314`) and `lin_out` as
   `a.pre_encode.out` (`clip-impl.h:354`, `clip.cpp:3963-3964`). The ggml graph
   reproduces the stack as 3 streaming conv stages + SiLU after each
   (`llamacpp/lfm3a.cpp:61-73`) - stride-2 convs use the state-cached left column
   + right zero-pad (`lfm3a.cpp:45-55`), the 1x1s are `ggml_conv_2d_direct`
   (`:65-66,:70-71`) - then flatten + `pre_encode_out` linear (`:75-81`).
   `n_pos = ceil(ceil(ceil(T/2)/2)/2)` (`clip.cpp:4977-4979`; HF `output_lengths`
   `modeling_lfm3_omni.py:407-408`).

3. **17 `a.blk.*` layers** (`config.json:37-55` `layer_types`): conv at
   0,1,3,4,6,7,9,10,12,13 (10 layers); `full_attention` at 2,5,8,11,14,15,16
   (7 - every third, then three in a row at the tail). Per layer:
   `RMSNorm(ln1)` -> operator -> `+residual` -> `RMSNorm(ln2)` -> gated SiLU FFN
   (`ffn_{gate,up,down}`, ffn 2560, no bias) -> `+residual`
   (`lfm3a.cpp:93-169`). ggml decides conv-vs-attn by which tensor triples are
   present and errors if both/neither (`clip.cpp:3976-3988`).

   Conv layers (conv_L_cache = 3, `config.json:20`):
   `conv_pw1` (768->2304, no bias) -> split `[B,C,X]` -> `B*X` -> causal depthwise
   conv1d width 3 via `ggml_ssm_conv` (state = last 2 taps per layer,
   `lfm3a.cpp:99-121`) -> `C*conv` -> `conv_pw2` (768->768). GGUF
   `a.blk.N.conv_{pw1,dw,pw2}` (`clip-impl.h:363-366`; HF `conv.in_proj`,
   `conv.conv` squeezed, `conv.out_proj` - `lfm3.py:525-526,538-539`, mapping
   excerpt `gguf/tensor-mapping-lfm-excerpt.txt` lines ~2843-2878). HF reference
   `Lfm3ShortConv` (`modeling_lfm3.py:332-356,371-424`).

   Attention layers (8 q / 4 kv heads, head_dim 96, `config.json:59-61`):
   `attn_{q,k,v,out}` bias-free projections -> reshape [96, heads, pos] ->
   **per-head q/k RMSNorm** `attn_q_norm`/`attn_k_norm` before rope
   (`lfm3a.cpp:127-132`; HF `q_layernorm`/`k_layernorm`,
   `modeling_lfm3.py:229-230,245-246`) -> NEOX rope over the full 96-dim head,
   theta 1e6 (`lfm3a.cpp:134-137`, `clip.cpp:2299`) -> causal attention over
   `[31 cached + n_pos]` keys -> `attn_out`. **Sliding window 32 tokens**
   (`config.json:11`; GGUF audio window `lfm3a.cpp:33` prefix=31; the shared
   mask is built once `lfm3a.cpp:89`); HF wraps the same window around
   `full_attention` layers via `create_sliding_window_causal_mask`
   (`modeling_lfm3_omni.py:425-438`). No sigmoid output gate and no
   post-operator sandwich norm here (unlike the LM trunk).

4. **`a.post_ln` + `mm.a.mlp` projector.** Final RMSNorm `a.post_ln.weight`
   = HF `audio_encoder.lfm.embedding_norm` (`lfm3a.cpp:171`; required at load
   `clip.cpp:3972-3975`; rename `lfm3.py:587-588` + `encoder.lfm.lfm.` prefix).
   Then `Lfm3OmniProjector` (`modeling_lfm3_omni.py:498-509`) = LayerNorm(768,
   eps 1e-5, with bias) -> Linear 768->2048 -> GELU -> Linear 2048->2048. ggml:
   `mm.a.mlp.0` (LayerNorm weight+bias), `mm.a.mlp.1` (linear_1), `mm.a.mlp.3`
   (linear_2), `FFN_GELU_ERF` (`lfm3a.cpp:174-177`, `clip.cpp:3966-3971`,
   `clip-impl.h:217`; HF->GGUF renames `lfm3.py:589-591`).

5. **Injection.** HF: the processor expands each audio placeholder into
   `num_encoder_tokens` copies of the audio placeholder token
   (`hf/processing_lfm3_omni.py:466-473`, `num_encoder_tokens` is
   `ceil(frames/8)` in `hf/feature_extraction_lfm3_omni.py:78-79`) and wraps the
   run in the vision span tokens (`_wrap`, `hf/processing_lfm3_omni.py:491-492`;
   defaults `media_start`/`media_end` with `vision_start`/`vision_end` fallback,
   `:181-211`). Inside `forward`, `get_audio_features` returns the projected
   2048-dim rows and `inputs_embeds.masked_scatter` writes them over the
   `audio_token_id` = 125031 positions (`hf/config.json:80`,
   `hf/modeling_lfm3_omni.py:701-707`; count check `get_placeholder_mask`
   `:627-636`). llama.cpp equivalent: each audio bitmap is emitted as
   `aud_beg` + an audio token chunk + `aud_end` (`llamacpp/mtmd.cpp:1789-1866`),
   where LFM3-Omni reuses the image span markers as `aud_beg`/`aud_end`
   (`mtmd.cpp:1121-1128`: `aud_beg = img_beg` = vision_start, `aud_end = img_end`
   = vision_end, set for LFM3VL at `mtmd.cpp:813-815`). The chunk's `n_tokens`
   come from `clip_n_output_tokens` = `ceil(T/8)` (`clip.cpp:4977-4979`) and the
   encoded+projected rows replace the chunk's embedding slots at eval, matching
   the HF masked_scatter.
