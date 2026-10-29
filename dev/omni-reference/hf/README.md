---
library_name: transformers
license: other
license_name: lfm1.0
license_link: LICENSE
pipeline_tag: any-to-any
tags:
- liquid
- lfm3
- lfm3-omni
- omni
- audio
- vision
- speech-to-speech
- edge
- conversational
- custom_code
language:
- en
- ar
- zh
- fr
- de
- it
- ja
- ko
- pt
- es
- vi
- th
- id
- hi
- ru
- pl
- da
- el
- fi
- he
- nl
- sv
- tr
---

<div align="center">
  <img
    src="https://huggingface.co/LiquidAI/LFM3-1.7B/resolve/main/assets/liquid-ai-banner.webp"
    alt="Liquid AI"
    style="width: 100%; max-width: 100%; height: auto; display: block; margin: 0;"
  />
  <div align="center" style="display: flex; justify-content: center; align-items: center; gap: 1.25em; margin: 8px 0 1.7rem; padding: 0; line-height: 1.3;">
    <a href="https://console.liquid.ai/dashboard" style="display: inline-flex; align-items: center; gap: 6px;"><img src="https://huggingface.co/LiquidAI/LFM3-1.7B/resolve/main/assets/icons/api.svg" width="16" height="16" alt="" aria-hidden="true" style="display: block; flex: 0 0 16px; width: 16px; height: 16px; margin: 0;" /><strong>API</strong></a>
    <a href="https://playground.liquid.ai/talk" style="display: inline-flex; align-items: center; gap: 6px;"><img src="https://huggingface.co/LiquidAI/LFM3-1.7B/resolve/main/assets/icons/try.svg" width="16" height="16" alt="" aria-hidden="true" style="display: block; flex: 0 0 16px; width: 16px; height: 16px; margin: 0;" /><strong>Try</strong></a>
    <a href="https://docs.liquid.ai/lfm/getting-started/welcome" style="display: inline-flex; align-items: center; gap: 6px;"><img src="https://huggingface.co/LiquidAI/LFM3-1.7B/resolve/main/assets/icons/docs.svg" width="16" height="16" alt="" aria-hidden="true" style="display: block; flex: 0 0 16px; width: 16px; height: 16px; margin: 0;" /><strong>Docs</strong></a>
    <a href="https://www.liquid.ai/blog/lfm3" style="display: inline-flex; align-items: center; gap: 6px;"><img src="https://huggingface.co/LiquidAI/LFM3-1.7B/resolve/main/assets/icons/blog.svg" width="16" height="16" alt="" aria-hidden="true" style="display: block; flex: 0 0 16px; width: 16px; height: 16px; margin: 0;" /><strong>Blog</strong></a>
  </div>
</div>

# LFM3-Omni-2B

LFM3-Omni-2B is [Liquid AI](https://www.liquid.ai/)'s first **omni-modal** model in the LFM3 family: a single end-to-end model that **understands text, images, video and audio**, and **answers in text and speech**. It is built for **on-device** use: one model replaces a separate ASR model, vision-language model, LLM and TTS model.

- **Any input, text or speech output:** ask about a photo by voice, transcribe or translate speech, chat out loud, or ask questions about a video and its soundtrack together.
- **Watches and listens together:** for videos with sound, the model sees and hears them at the same time, so it can answer questions that need both.
- **Small enough for devices:** at 2.2B parameters, it is designed to run efficiently on phones, laptops and other edge hardware.

Find more information about the LFM3 family in [our release blog post](https://www.liquid.ai/blog/lfm3).

## Model details
<hr style="margin: -0.7em 0 1.2em 0; border: 0; border-top: 1px solid rgba(127, 127, 127, 0.3);" />

The [LFM3 family](https://huggingface.co/collections/LiquidAI/lfm3) includes base, text, vision, audio, omni, and decision models.

<img src="https://huggingface.co/LiquidAI/LFM3-Omni-2B/resolve/main/assets/lfm3-omni-architecture-light.png" alt="LFM3-Omni architecture: a spoken question passes through the LFM3 audio encoder and audio connector, an image through the LFM3 vision encoder and vision connector, and a text prompt through the tokenizer and embedding, all into the LFM3-1.7B Liquid Foundation Model. Its output stream of text and audio tokens drives the LM head for text and the depthformer and Liquid Audio Codec for 24 kHz speech." width="2400" height="1996" style="width: 100%; max-width: 2400px; height: auto;" />

<table style="width: 100%; display: table; border: 0; border-radius: 0; border-collapse: collapse;">
<thead style="border: 0;">
<tr style="border: 0;">
<th scope="col" style="text-align: left; padding: 0.55em 0.75em; border: 0; border-bottom: 1px solid rgba(127, 127, 127, 0.55);">Property</th>
<th scope="col" style="text-align: left; padding: 0.55em 0.75em; border: 0; border-bottom: 1px solid rgba(127, 127, 127, 0.55);">Specification</th>
</tr>
</thead>
<tbody style="border: 0;">
<tr style="border: 0;">
<td style="border: 0; padding: 0.55em 0.75em;"><strong>Checkpoint</strong></td>
<td style="border: 0; padding: 0.55em 0.75em;">Post-trained omni-modal model</td>
</tr>
<tr style="border: 0;">
<td style="border: 0; padding: 0.55em 0.75em;"><strong>Parameters</strong></td>
<td style="border: 0; padding: 0.55em 0.75em;">2.17B total </td>
</tr>
<tr style="border: 0;">
<td style="border: 0; padding: 0.55em 0.75em;"><strong>Architecture</strong></td>
<td style="border: 0; padding: 0.55em 0.75em;">LFM3 backbone, LFM3 vision encoder, LFM3 causal audio encoder, LiquidCodec audio tokenizer. </td>
</tr>
<tr style="border: 0;">
<td style="border: 0; padding: 0.55em 0.75em;"><strong>Inputs</strong></td>
<td style="border: 0; padding: 0.55em 0.75em;">Text, images, video with or without its soundtrack, and 16 kHz audio. Images and video are processed at native resolution; each 2×2 block of patches becomes one language token. Video is sampled at 2 fps.</td>
</tr>
<tr style="border: 0;">
<td style="border: 0; padding: 0.55em 0.75em;"><strong>Outputs</strong></td>
<td style="border: 0; padding: 0.55em 0.75em;">Text, 24 kHz speech, or interleaved text and speech</td>
</tr>
<tr style="border: 0;">
<td style="border: 0; padding: 0.55em 0.75em;"><strong>Context length</strong></td>
<td style="border: 0; padding: 0.55em 0.75em;">32K tokens (input + output)</td>
</tr>
<tr style="border: 0;">
<td style="border: 0; padding: 0.55em 0.75em;"><strong>Vocabulary</strong></td>
<td style="border: 0; padding: 0.55em 0.75em;">128,000 text tokens</td>
</tr>
<tr style="border: 0;">
<td style="border: 0; padding: 0.55em 0.75em;"><strong>Text languages</strong></td>
<td style="border: 0; padding: 0.55em 0.75em;">English, Arabic, Chinese, French, German, Italian, Japanese, Korean, Portuguese, Spanish, Vietnamese, Thai, Indonesian, Hindi, Russian, Polish</td>
</tr>
<tr style="border: 0;">
<td style="border: 0; padding: 0.55em 0.75em;"><strong>Speech languages and locales</strong></td>
<td style="border: 0; padding: 0.55em 0.75em;">Arabic, Danish, German, Greek, British English, American English, Finnish, French, Hebrew, Italian, Japanese, Korean, Dutch, Polish, Portuguese, Spanish, Swedish, Turkish, Chinese</td>
</tr>
<tr style="border: 0;">
<td style="border: 0; padding: 0.55em 0.75em;"><strong>License</strong></td>
<td style="border: 0; padding: 0.55em 0.75em;"><a href="LICENSE">LFM Open License v1.0</a></td>
</tr>
</tbody>
</table>

This is a **non-reasoning** model. `enable_thinking=True` is not supported.

## Inference
<hr style="margin: -0.7em 0 1.2em 0; border: 0; border-top: 1px solid rgba(127, 127, 127, 0.3);" />

Choose a checkpoint variant for your inference framework and hardware.

<table style="width: 100%; display: table; border: 0; border-radius: 0; border-collapse: collapse;">
<thead style="border: 0;">
<tr style="border: 0;">
<th scope="col" style="text-align: left; padding: 0.55em 0.75em; border: 0; border-bottom: 1px solid rgba(127, 127, 127, 0.55);">Checkpoint</th>
<th scope="col" style="text-align: left; padding: 0.55em 0.75em; border: 0; border-bottom: 1px solid rgba(127, 127, 127, 0.55);">Framework</th>
<th scope="col" style="text-align: left; padding: 0.55em 0.75em; border: 0; border-bottom: 1px solid rgba(127, 127, 127, 0.55);">Description</th>
</tr>
</thead>
<tbody style="border: 0;">
<tr style="border: 0;">
<td style="border: 0; padding: 0.55em 0.75em;"><a href="https://huggingface.co/LiquidAI/LFM3-Omni-2B">BF16</a></td>
<td style="border: 0; padding: 0.55em 0.75em;"><a href="https://docs.liquid.ai/deployment/gpu-inference/sglang">SGLang</a>, <a href="https://docs.liquid.ai/deployment/gpu-inference/transformers">Transformers</a></td>
<td style="border: 0; padding: 0.55em 0.75em;">BF16 weights for GPU inference. SGLang serves text output.</td>
</tr>
<tr style="border: 0;">
<td style="border: 0; padding: 0.55em 0.75em;"><a href="https://huggingface.co/LiquidAI/LFM3-Omni-2B-GGUF">GGUF</a></td>
<td style="border: 0; padding: 0.55em 0.75em;"><a href="https://docs.liquid.ai/deployment/on-device/llama-cpp">llama.cpp</a></td>
<td style="border: 0; padding: 0.55em 0.75em;">Local inference on CPUs and GPUs, including a realtime speech server.</td>
</tr>
</tbody>
</table>

### Sampling parameters

- **Text output:** greedy decoding works well.
- **Speech or interleaved output:** use `audio_temperature=0.4`, `audio_repetition_penalty=1.2`

### Generation modes (optional)

To force a generation mode, pass the optional `generation_mode` argument to `model.generate`. 

<table style="width: 100%; display: table; border: 0; border-radius: 0; border-collapse: collapse;">
<thead style="border: 0;">
<tr style="border: 0;">
<th scope="col" style="text-align: left; padding: 0.55em 0.75em; border: 0; border-bottom: 1px solid rgba(127, 127, 127, 0.55);">Mode</th>
<th scope="col" style="text-align: left; padding: 0.55em 0.75em; border: 0; border-bottom: 1px solid rgba(127, 127, 127, 0.55);"><code>generation_mode</code> (optional)</th>
<th scope="col" style="text-align: left; padding: 0.55em 0.75em; border: 0; border-bottom: 1px solid rgba(127, 127, 127, 0.55);">Opening token</th>
</tr>
</thead>
<tbody style="border: 0;">
<tr style="border: 0;">
<td style="border: 0; padding: 0.55em 0.75em;">Text</td>
<td style="border: 0; padding: 0.55em 0.75em;"><code>"text"</code></td>
<td style="border: 0; padding: 0.55em 0.75em;">none</td>
</tr>
<tr style="border: 0;">
<td style="border: 0; padding: 0.55em 0.75em;">Speech</td>
<td style="border: 0; padding: 0.55em 0.75em;"><code>"audio"</code></td>
<td style="border: 0; padding: 0.55em 0.75em;"><code>&lt;|audio_start|&gt;</code></td>
</tr>
<tr style="border: 0;">
<td style="border: 0; padding: 0.55em 0.75em;">Text and speech, interleaved</td>
<td style="border: 0; padding: 0.55em 0.75em;"><code>"interleaved"</code></td>
<td style="border: 0; padding: 0.55em 0.75em;"><code>&lt;|mixed_start|&gt;</code></td>
</tr>
</tbody>
</table>

By default, the model selects the output modality based on the system prompt and input modality, typically matching the input modality. To encourage a specific output modality, append the corresponding instructions to the end of your system prompt.

<table style="width: 100%; display: table; border: 0; border-radius: 0; border-collapse: collapse;">
<thead style="border: 0;">
<tr style="border: 0;">
<th scope="col" style="text-align: left; padding: 0.55em 0.75em; border: 0; border-bottom: 1px solid rgba(127, 127, 127, 0.55);">Mode</th>
<th scope="col" style="text-align: left; padding: 0.55em 0.75em; border: 0; border-bottom: 1px solid rgba(127, 127, 127, 0.55);">System Prompt</th>
</tr>
</thead>
<tbody style="border: 0;">
<tr style="border: 0;">
<td style="border: 0; padding: 0.55em 0.75em;">Text</td>
<td style="border: 0; padding: 0.55em 0.75em;"><code>Respond in text.</code></td>
</tr>
<tr style="border: 0;">
<td style="border: 0; padding: 0.55em 0.75em;">Text and speech, interleaved</td>
<td style="border: 0; padding: 0.55em 0.75em;"><code>Respond in interleaved text and audio.</code></td>
</tr>
</tbody>
</table>



### Transformers quick start

Install the dependencies, then load the model and processor with the checkpoint's custom code. The codec implementation is included in this repository, and the processor loads its weights from `audio_tokenizer/`.

```bash
pip install "transformers>=5.20.2" torch torchvision torchcodec accelerate pillow librosa soundfile
```

```python
import torch
import soundfile as sf

from transformers import AutoModelForMultimodalLM, AutoProcessor

model_id = "LiquidAI/LFM3-Omni-2B"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

model = AutoModelForMultimodalLM.from_pretrained(
    model_id, dtype=dtype, device_map=device, trust_remote_code=True
).eval()
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
codec = processor.audio_tokenizer.to(device=device).eval()
```

<details>
<summary><strong>Chat template</strong></summary>

LFM3-Omni uses a ChatML-like format, which `processor.apply_chat_template()` renders for you, generated from the checkpoint's [chat template](https://huggingface.co/LiquidAI/LFM3-Omni-2B/blob/main/chat_template.jinja):

```text
<|startoftext|><|im_start|>system
Respond with text.<|im_end|>
<|im_start|>user
Name a famous bridge in San Francisco.<|im_end|>
<|im_start|>assistant
<think></think>The Golden Gate Bridge.<|im_end|>
<|im_start|>user
When did it open?<|im_end|>
<|im_start|>assistant
<think></think>
```

The system turn is optional, and every assistant turn opens with an empty `<think></think>`. Each image, video or audio clip is wrapped in `<|vision_start|>` / `<|vision_end|>`, which the template adds. Here is an image and a spoken question:

```text
<|startoftext|><|im_start|>user
<|vision_start|><|image|><|vision_end|><|vision_start|><|audio|><|vision_end|>Answer in one sentence.<|im_end|>
<|im_start|>assistant
<think></think>
```

</details>

### Usage examples

The examples below reuse the model, codec, and processor loaded above. Replace `audio.wav`, `question.wav`, `recording.wav`, `photo.jpg`, or `clip.mp4` with your own files. Audio files are resampled to 16 kHz mono on load.

<details>
<summary><strong>Speech recognition (ASR)</strong></summary>

To use the model for transcription, use the system prompt `Perform ASR in <language>.`, where `<language>` is the English name of the language.

Below, we transcribe a `.wav` file containing English speech.

```python
conversation = [
    {"role": "system", "content": "Perform ASR in English."},
    {"role": "user", "content": [{"type": "audio", "audio": "audio.wav"}]},
]

inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=1024, generation_mode="text")
text = processor.decode(
    outputs.sequences[0, inputs["input_ids"].shape[1]:outputs.sequence_lengths[0]],
    skip_special_tokens=True,
)
print(text)
```

</details>

<details>
<summary><strong>Speech synthesis (TTS)</strong></summary>

For speech synthesis, use the system prompt `Perform TTS in <language> using a <gender> voice.`.
Use one of the speech languages listed in [Model details](#model-details) and a `female` or `male` voice. For English, say `American English` or `British English`. Pass `generation_mode="audio"` so the answer opens with `<|audio_start|>`.

```python
conversation = [
    {"role": "system", "content": "Perform TTS in American English using a female voice."},
    {"role": "user", "content": "The quick brown fox jumps over the lazy dog"},
]
inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(device)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        generation_mode="audio",
        audio_temperature=0.4,
        audio_repetition_penalty=1.2,
    )
    waveform = codec.decode(
        outputs.audio_codes[:1, :, :outputs.audio_lengths[0]]
    ).audio_values[0]  # 24 kHz

sf.write("tts.wav", waveform.squeeze(0).float().cpu().numpy(), 24_000)
```

</details>

<details>
<summary><strong>Voice chat: speech and text responses</strong></summary>

In this mode, the model will respond to the user in both speech and text.
The recommended system prompt is `Respond with interleaved text and audio.`.
User input can be either text or audio, and we pass `generation_mode="interleaved"` so the answer opens with `<|mixed_start|>`.

The assistant responds in a conversational style with interleaved text and speech. This example saves the complete response.

```python
conversation = [
    {"role": "system", "content": "Respond with interleaved text and audio."},
    {"role": "user", "content": [{"type": "audio", "audio": "question.wav"}]},
]
inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(device)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        generation_mode="interleaved",
        audio_temperature=0.4,
        audio_repetition_penalty=1.2,
    )

    text = processor.decode(
        outputs.sequences[0, inputs["input_ids"].shape[1]:outputs.sequence_lengths[0]],
        skip_special_tokens=True,
    )
    waveform = codec.decode(
        outputs.audio_codes[:1, :, :outputs.audio_lengths[0]]
    ).audio_values[0]  # 24 kHz
print(text)
sf.write("reply.wav", waveform.squeeze(0).float().cpu().numpy(), 24_000)
```

</details>

<details>
<summary><strong>Speech-to-text chat</strong></summary>

For a text-only answer, no specific system prompt is needed. `Respond with text.` makes it explicit. Pass `generation_mode="text"`. LFM3-Omni-2B is a non-reasoning model: every answer opens with an empty `<think></think>`.

```python
conversation = [
    {"role": "user", "content": [{"type": "audio", "audio": "question.wav"}]},
]
inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=1024, generation_mode="text")
    text = processor.decode(
        outputs.sequences[0, inputs["input_ids"].shape[1]:outputs.sequence_lengths[0]],
        skip_special_tokens=True,
    )
print(text)
```

</details>

<details>
<summary><strong>Audio understanding</strong></summary>

Include the recording and a question in the user turn. Questions can address speech, music, background sounds, or events. No specific system prompt is required. To ground answers in the recording, use `Answer questions about the provided audio clip. Base every answer only on what can be heard.`.

```python
conversation = [
    {
        "role": "system",
        "content": "Answer questions about the provided audio clip. Base every answer only on what can be heard.",
    },
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio": "recording.wav"},
            {"type": "text", "text": "What is happening in this recording?"},
        ],
    },
]
inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=1024, generation_mode="text")

print(processor.decode(
    outputs.sequences[0, inputs["input_ids"].shape[1]:outputs.sequence_lengths[0]],
    skip_special_tokens=True,
))
```

For multiple-choice questions, include the choices in the text part and end it with `Please choose the answer from the options.`

</details>

<details>
<summary><strong>Translation</strong></summary>

Use a system prompt that specifies the source language, target language, and output format:

- **Speech to text:** `Please translate the following spoken {source} into {target} text.`

Replace `{source}` and `{target}` with the English names of the source and target languages. For text input, set the user message's `content` to the text to translate. Translation to or from English generally works best.

</details>

<details>
<summary><strong>Image understanding</strong></summary>

Include the image and a question in the user turn. To ground answers in the image, use the system prompt `Answer questions about the provided image. Base every answer only on what can be seen.`. For captions, use `Describe the provided image.`. Images are processed at their native resolution, up to 2,048 tokens per image.

```python
conversation = [
    {
        "role": "system",
        "content": "Answer questions about the provided image. Base every answer only on what can be seen.",
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "photo.jpg"},
            {"type": "text", "text": "What is written on the sign?"},
        ],
    },
]
inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=1024, generation_mode="text")

print(processor.decode(
    outputs.sequences[0, inputs["input_ids"].shape[1]:outputs.sequence_lengths[0]],
    skip_special_tokens=True,
))
```

</details>

<details>
<summary><strong>Spoken questions about an image</strong></summary>

The question can be asked by voice, with the recording in place of the text. Add `Respond with text.` to the system prompt for a text answer, or follow the voice chat example with `Respond with interleaved text and audio.` for a spoken one.

```python
conversation = [
    {
        "role": "system",
        "content": "Answer questions about the provided image. Base every answer only on what can be seen. Respond with text.",
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "photo.jpg"},
            {"type": "audio", "audio": "question.wav"},
        ],
    },
]
inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=1024, generation_mode="text")

print(processor.decode(
    outputs.sequences[0, inputs["input_ids"].shape[1]:outputs.sequence_lengths[0]],
    skip_special_tokens=True,
))
```

</details>

<details>
<summary><strong>Video understanding</strong></summary>

Include the video and a question in the user turn. No specific system prompt is required. Video is sampled at 2 frames per second (4 to 384 frames), and each pair of frames is prefixed with its timestamp.

```python
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "video", "video": "clip.mp4"},
            {"type": "text", "text": "What happens in this video?"},
        ],
    },
]
inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=1024, generation_mode="text")

print(processor.decode(
    outputs.sequences[0, inputs["input_ids"].shape[1]:outputs.sequence_lengths[0]],
    skip_special_tokens=True,
))
```

</details>

<details>
<summary><strong>Video with its soundtrack</strong></summary>

To let the model see and hear a video, use the system prompt `Answer questions about the video using what you see and hear.` and pass two flags:

- `load_audio_from_video=True` reads the audio track out of the video file.
- `use_audio_in_video=True` interleaves that audio with the frames, so each pair of frames is followed by the sound that goes with it.

Without them, the model gets the frames only.

```python
conversation = [
    {"role": "system", "content": "Answer questions about the video using what you see and hear."},
    {
        "role": "user",
        "content": [
            {"type": "video", "video": "clip.mp4"},
            {"type": "text", "text": "What happens here, and what can you hear?"},
        ],
    },
]
inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    load_audio_from_video=True,
    processor_kwargs={"use_audio_in_video": True},
).to(device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=1024, generation_mode="text")

print(processor.decode(
    outputs.sequences[0, inputs["input_ids"].shape[1]:outputs.sequence_lengths[0]],
    skip_special_tokens=True,
))
```

</details>

<details>
<summary><strong>Tool calling from spoken requests</strong></summary>

Pass function schemas through `tools=...` to `apply_chat_template()`. The chat template places them in the system prompt between `<tools>` and `</tools>`. Requests may be spoken or written. A tool call looks like this:

```
<tool_call>
<function=set_volume>
<parameter=volume>
30
</parameter>
</function>
</tool_call>
```

Values other than strings are written as JSON. Your application parses and runs the calls, then appends the assistant's `tool_calls` (with `arguments` as a dict) and a `tool` message with the results to the conversation. Generate again with the same `tools`: the model either makes more calls or answers.

```python
import json
import re

tools = [{
    "type": "function",
    "function": {
        "name": "set_volume",
        "description": "Set the device speaker volume.",
        "parameters": {
            "type": "object",
            "properties": {
                "volume": {"type": "integer", "description": "Volume from 0 to 100."}
            },
            "required": ["volume"],
        },
    },
}]
functions = {"set_volume": lambda volume: {"volume": volume}}  # your implementations

CALL_RE = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>", re.DOTALL)
PARAM_RE = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.DOTALL)


def parse_tool_calls(text):
    calls = []
    for name, body in CALL_RE.findall(text):
        arguments = {}
        for key, value in PARAM_RE.findall(body):
            try:
                arguments[key] = json.loads(value)
            except json.JSONDecodeError:
                arguments[key] = value  # plain string
        calls.append({"type": "function", "function": {"name": name, "arguments": arguments}})
    return calls


def respond(conversation):
    inputs = processor.apply_chat_template(
        conversation,
        tools=tools,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=1024, generation_mode="text")
    return processor.decode(
        outputs.sequences[0, inputs["input_ids"].shape[1]:outputs.sequence_lengths[0]],
        skip_special_tokens=True,
    )


conversation = [
    {"role": "user", "content": [{"type": "audio", "audio": "question.wav"}]},
]
text = respond(conversation)
while calls := parse_tool_calls(text):
    results = [functions[c["function"]["name"]](**c["function"]["arguments"]) for c in calls]
    conversation += [
        {"role": "assistant", "tool_calls": calls},
        {"role": "tool", "content": json.dumps(results)},
    ]
    text = respond(conversation)

print(text)
```

See the [Tool Use documentation](https://docs.liquid.ai/lfm/key-concepts/tool-use) for details.

</details>

## Fine-tuning
<hr style="margin: -0.7em 0 1.2em 0; border: 0; border-top: 1px solid rgba(127, 127, 127, 0.3);" />

LFM3-Omni-2B is designed for **easy customization** for a given domain, task, or application with Hugging Face Transformers.

<table style="width: 100%; display: table; border: 0; border-radius: 0; border-collapse: collapse;">
<thead style="border: 0;">
<tr style="border: 0;">
<th scope="col" style="text-align: left; padding: 0.55em 0.75em; border: 0; border-bottom: 1px solid rgba(127, 127, 127, 0.55);">Training</th>
<th scope="col" style="text-align: left; padding: 0.55em 0.75em; border: 0; border-bottom: 1px solid rgba(127, 127, 127, 0.55);">Use case</th>
<th scope="col" style="text-align: center; padding: 0.55em 0.75em; border: 0; border-bottom: 1px solid rgba(127, 127, 127, 0.55);">Notebook</th>
</tr>
</thead>
<tbody style="border: 0;">
<tr style="border: 0;">
<td style="border: 0; padding: 0.55em 0.75em;"><strong>SFT</strong></td>
<td style="border: 0; padding: 0.55em 0.75em;">Fine-tune on your own omni data (text, audio, images) with Transformers</td>
<td style="text-align: center; border: 0; padding: 0.55em 0.75em;"><a href="https://colab.research.google.com/github/Liquid4All/cookbook/blob/main/finetuning/notebooks/lfm3_omni_sft_with_transformers.ipynb" title="SFT with Transformers"><img src="https://colab.research.google.com/assets/colab-badge.svg" width="117" height="20" alt="Open in Colab" style="display: inline-block; width: 117px; height: 20px; margin: 0; vertical-align: middle;" /></a></td>
</tr>
<tr style="border: 0;">
<td style="border: 0; padding: 0.55em 0.75em;"><strong>SFT</strong></td>
<td style="border: 0; padding: 0.55em 0.75em;">Visual question answering</td>
<td style="text-align: center; border: 0; padding: 0.55em 0.75em;"><a href="https://colab.research.google.com/github/Liquid4All/cookbook/blob/main/finetuning/notebooks/lfm3_omni_vqa_sft_with_transformers.ipynb" title="VQA SFT with Transformers"><img src="https://colab.research.google.com/assets/colab-badge.svg" width="117" height="20" alt="Open in Colab" style="display: inline-block; width: 117px; height: 20px; margin: 0; vertical-align: middle;" /></a></td>
</tr>
</tbody>
</table>

## Performance
<hr style="margin: -0.7em 0 1.2em 0; border: 0; border-top: 1px solid rgba(127, 127, 127, 0.3);" />

<img src="https://huggingface.co/LiquidAI/LFM3-Omni-2B/resolve/main/assets/lfm3-omni-family-radar-light.png" alt="LFM3-Omni radar chart: one model scored across text (IFEval, GSM8K, MMLU-Pro, BFCL v4, RULER), vision (general, perception, documents, math and reasoning, grounding, embodied, GUI, video, multilingual), speech (English and multilingual ASR, TTS quality, VoiceBench, MMAR, voice tool calling) and omni tasks (image and audio, video and audio, spoken VQA, spoken embodied)." width="1320" height="1016" style="width: 100%; max-width: 1320px; height: auto;" />

### Omni

<div role="region" aria-label="Omni benchmarks" tabindex="0" style="width: 100%; max-width: 100%; overflow-x: auto; -webkit-overflow-scrolling: touch;">
<table style="width: 100%; min-width: 36rem; display: table; border: 0; border-radius: 0; border-collapse: collapse; font-size: calc(0.875em - 1pt);">
<thead style="border: 0;">
<tr style="border: 0;">
<th scope="col" style="text-align: left; padding: 0.55em 0.75em; border: 0; border-bottom: 1px solid rgba(127, 127, 127, 0.55);">Benchmark</th>
<th scope="col" style="text-align: center; background-color: rgba(108, 79, 224, 0.08); padding: 0.55em 0.75em; border: 0; border-bottom: 1px solid rgba(127, 127, 127, 0.55);"><strong>LFM3-Omni<br>2B</strong></th>
<th scope="col" style="text-align: center; padding: 0.55em 0.75em; border: 0; border-bottom: 1px solid rgba(127, 127, 127, 0.55);">Qwen2.5-Omni<br>3B</th>
<th scope="col" style="text-align: center; padding: 0.55em 0.75em; border: 0; border-bottom: 1px solid rgba(127, 127, 127, 0.55);">Qwen2.5-Omni<br>7B</th>
<th scope="col" style="text-align: center; padding: 0.55em 0.75em; border: 0; border-bottom: 1px solid rgba(127, 127, 127, 0.55);">Gemma 4<br>E2B</th>
<th scope="col" style="text-align: center; padding: 0.55em 0.75em; border: 0; border-bottom: 1px solid rgba(127, 127, 127, 0.55);">Gemma 4<br>E4B</th>
<th scope="col" style="text-align: center; padding: 0.55em 0.75em; border: 0; border-bottom: 1px solid rgba(127, 127, 127, 0.55);">Phi-4<br>multimodal</th>
</tr>
</thead>
<tbody style="border: 0;">
<tr style="border: 0;">
<th scope="row" style="text-align: left; font-weight: 400; border: 0; padding: 0.55em 0.75em;">Total parameters</th>
<td style="text-align: center; font-variant-numeric: tabular-nums; background-color: rgba(108, 79, 224, 0.08); border: 0; padding: 0.55em 0.75em;">2.2B</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">6B</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">10.7B</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">5B</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">8B</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">5.6B</td>
</tr>
</tbody>
<tbody style="border: 0;">
<tr style="border: 0;"><th scope="rowgroup" style="text-align: left; background-color: transparent; padding: 0.55em 0.75em; border: 0; font-weight: 600; padding-top: 1em; padding-bottom: 0.5em; border-top: 1px solid rgba(127, 127, 127, 0.25);">Video + audio understanding</th><td aria-hidden="true" style="border: 0; padding: 0.55em 0.75em; background-color: rgba(108, 79, 224, 0.08); border-top: 1px solid rgba(127, 127, 127, 0.25);"></td><td colspan="5" aria-hidden="true" style="border: 0; padding: 0.55em 0.75em; border-top: 1px solid rgba(127, 127, 127, 0.25);"></td></tr>
<tr style="border: 0;">
<th scope="row" style="text-align: left; font-weight: 400; border: 0; padding: 0.55em 0.75em;">Daily-Omni</th>
<td style="text-align: center; font-variant-numeric: tabular-nums; background-color: rgba(108, 79, 224, 0.08); border: 0; padding: 0.55em 0.75em;"><strong>70.09</strong></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">58.81</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><u>61.32</u></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">50.63</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">54.14</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">48.87</td>
</tr>
<tr style="border: 0;">
<th scope="row" style="text-align: left; font-weight: 400; border: 0; padding: 0.55em 0.75em;">WorldSense</th>
<td style="text-align: center; font-variant-numeric: tabular-nums; background-color: rgba(108, 79, 224, 0.08); border: 0; padding: 0.55em 0.75em;"><strong>47.10</strong></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">44.99</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><u>45.37</u></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">38.11</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">43.51</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">42.37</td>
</tr>
<tr style="border: 0;">
<th scope="row" style="text-align: left; font-weight: 400; border: 0; padding: 0.55em 0.75em;">AV-SpeakerBench</th>
<td style="text-align: center; font-variant-numeric: tabular-nums; background-color: rgba(108, 79, 224, 0.08); border: 0; padding: 0.55em 0.75em;"><strong>46.79</strong></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">40.44</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><u>43.24</u></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">32.32</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">36.36</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">37.95</td>
</tr>
</tbody>
<tbody style="border: 0;">
<tr style="border: 0;"><th scope="rowgroup" style="text-align: left; background-color: transparent; padding: 0.55em 0.75em; border: 0; font-weight: 600; padding-top: 1em; padding-bottom: 0.5em; border-top: 1px solid rgba(127, 127, 127, 0.25);">Image + audio understanding</th><td aria-hidden="true" style="border: 0; padding: 0.55em 0.75em; background-color: rgba(108, 79, 224, 0.08); border-top: 1px solid rgba(127, 127, 127, 0.25);"></td><td colspan="5" aria-hidden="true" style="border: 0; padding: 0.55em 0.75em; border-top: 1px solid rgba(127, 127, 127, 0.25);"></td></tr>
<tr style="border: 0;">
<th scope="row" style="text-align: left; font-weight: 400; border: 0; padding: 0.55em 0.75em;">OmniBench (overall)</th>
<td style="text-align: center; font-variant-numeric: tabular-nums; background-color: rgba(108, 79, 224, 0.08); border: 0; padding: 0.55em 0.75em;">46.32</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><u>47.11</u></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><strong>49.65</strong></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">16.81</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">20.67</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">38.97</td>
</tr>
<tr style="border: 0;">
<th scope="row" style="text-align: left; font-weight: 400; border: 0; padding: 0.55em 0.75em;">OmniBench (speech)</th>
<td style="text-align: center; font-variant-numeric: tabular-nums; background-color: rgba(108, 79, 224, 0.08); border: 0; padding: 0.55em 0.75em;"><strong>48.25</strong></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">45.65</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><u>47.99</u></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">16.08</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">21.27</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">38.91</td>
</tr>
</tbody>
<tbody style="border: 0;">
<tr style="border: 0;"><th scope="rowgroup" style="text-align: left; background-color: transparent; padding: 0.55em 0.75em; border: 0; font-weight: 600; padding-top: 1em; padding-bottom: 0.5em; border-top: 1px solid rgba(127, 127, 127, 0.25);">Spoken visual QA</th><td aria-hidden="true" style="border: 0; padding: 0.55em 0.75em; background-color: rgba(108, 79, 224, 0.08); border-top: 1px solid rgba(127, 127, 127, 0.25);"></td><td colspan="5" aria-hidden="true" style="border: 0; padding: 0.55em 0.75em; border-top: 1px solid rgba(127, 127, 127, 0.25);"></td></tr>
<tr style="border: 0;">
<th scope="row" style="text-align: left; font-weight: 400; border: 0; padding: 0.55em 0.75em;">RealWorldQA</th>
<td style="text-align: center; font-variant-numeric: tabular-nums; background-color: rgba(108, 79, 224, 0.08); border: 0; padding: 0.55em 0.75em;"><u>64.68</u></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">61.05</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><strong>69.93</strong></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">39.35</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">49.02</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">64.31</td>
</tr>
<tr style="border: 0;">
<th scope="row" style="text-align: left; font-weight: 400; border: 0; padding: 0.55em 0.75em;">MMStar</th>
<td style="text-align: center; font-variant-numeric: tabular-nums; background-color: rgba(108, 79, 224, 0.08); border: 0; padding: 0.55em 0.75em;">51.79</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">50.80</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><strong>57.27</strong></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">38.87</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">44.60</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><u>54.80</u></td>
</tr>
<tr style="border: 0;">
<th scope="row" style="text-align: left; font-weight: 400; border: 0; padding: 0.55em 0.75em;">POPE</th>
<td style="text-align: center; font-variant-numeric: tabular-nums; background-color: rgba(108, 79, 224, 0.08); border: 0; padding: 0.55em 0.75em;">83.78</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><u>85.34</u></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><strong>85.37</strong></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">74.20</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">77.82</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">83.00</td>
</tr>
<tr style="border: 0;">
<th scope="row" style="text-align: left; font-weight: 400; border: 0; padding: 0.55em 0.75em;">ERQA</th>
<td style="text-align: center; font-variant-numeric: tabular-nums; background-color: rgba(108, 79, 224, 0.08); border: 0; padding: 0.55em 0.75em;"><strong>39.43</strong></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">37.34</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><u>39.16</u></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">20.63</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">32.12</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">30.81</td>
</tr>
<tr style="border: 0;">
<th scope="row" style="text-align: left; font-weight: 400; border: 0; padding: 0.55em 0.75em;">EmbSpatial</th>
<td style="text-align: center; font-variant-numeric: tabular-nums; background-color: rgba(108, 79, 224, 0.08); border: 0; padding: 0.55em 0.75em;"><strong>67.59</strong></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">53.40</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><u>59.51</u></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">44.08</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">54.01</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">42.60</td>
</tr>
</tbody>
</table>
</div>

### Vision

<div role="region" aria-label="Vision benchmarks" tabindex="0" style="width: 100%; max-width: 100%; overflow-x: auto; -webkit-overflow-scrolling: touch;">
<table style="width: 100%; min-width: 36rem; display: table; border: 0; border-radius: 0; border-collapse: collapse; font-size: calc(0.875em - 1pt);">
<thead style="border: 0;">
<tr style="border: 0;">
<th scope="col" style="text-align: left; padding: 0.55em 0.75em; border: 0; border-bottom: 1px solid rgba(127, 127, 127, 0.55);">Benchmark</th>
<th scope="col" style="text-align: center; background-color: rgba(108, 79, 224, 0.08); padding: 0.55em 0.75em; border: 0; border-bottom: 1px solid rgba(127, 127, 127, 0.55);"><strong>LFM3-Omni<br>2B</strong></th>
<th scope="col" style="text-align: center; padding: 0.55em 0.75em; border: 0; border-bottom: 1px solid rgba(127, 127, 127, 0.55);">Qwen2.5-Omni<br>3B</th>
<th scope="col" style="text-align: center; padding: 0.55em 0.75em; border: 0; border-bottom: 1px solid rgba(127, 127, 127, 0.55);">Gemma 4<br>E2B</th>
<th scope="col" style="text-align: center; padding: 0.55em 0.75em; border: 0; border-bottom: 1px solid rgba(127, 127, 127, 0.55);">Gemma 4<br>E4B</th>
<th scope="col" style="text-align: center; padding: 0.55em 0.75em; border: 0; border-bottom: 1px solid rgba(127, 127, 127, 0.55);">Phi-4<br>multimodal</th>
<th scope="col" style="text-align: center; padding: 0.55em 0.75em; border: 0; border-bottom: 1px solid rgba(127, 127, 127, 0.55);">Cosmos3<br>Edge</th>
</tr>
</thead>
<tbody style="border: 0;">
<tr style="border: 0;">
<th scope="row" style="text-align: left; font-weight: 400; border: 0; padding: 0.55em 0.75em;">Total parameters</th>
<td style="text-align: center; font-variant-numeric: tabular-nums; background-color: rgba(108, 79, 224, 0.08); border: 0; padding: 0.55em 0.75em;">2.2B</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">6B</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">5B</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">8B</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">5.6B</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">4B</td>
</tr>
</tbody>
<tbody style="border: 0;">
<tr style="border: 0;"><th scope="rowgroup" style="text-align: left; background-color: transparent; padding: 0.55em 0.75em; border: 0; font-weight: 600; padding-top: 1em; padding-bottom: 0.5em; border-top: 1px solid rgba(127, 127, 127, 0.25);">General</th><td aria-hidden="true" style="border: 0; padding: 0.55em 0.75em; background-color: rgba(108, 79, 224, 0.08); border-top: 1px solid rgba(127, 127, 127, 0.25);"></td><td colspan="5" aria-hidden="true" style="border: 0; padding: 0.55em 0.75em; border-top: 1px solid rgba(127, 127, 127, 0.25);"></td></tr>
<tr style="border: 0;">
<th scope="row" style="text-align: left; font-weight: 400; border: 0; padding: 0.55em 0.75em;">MMStar</th>
<td style="text-align: center; font-variant-numeric: tabular-nums; background-color: rgba(108, 79, 224, 0.08); border: 0; padding: 0.55em 0.75em;">55.67</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">55.67</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">46.47</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">52.00</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><strong>59.33</strong></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><u>56.07</u></td>
</tr>
<tr style="border: 0;">
<th scope="row" style="text-align: left; font-weight: 400; border: 0; padding: 0.55em 0.75em;">MMMU (val)</th>
<td style="text-align: center; font-variant-numeric: tabular-nums; background-color: rgba(108, 79, 224, 0.08); border: 0; padding: 0.55em 0.75em;">42.44</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><u>48.00</u></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">37.81</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">43.14</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><strong>54.76</strong></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">37.14</td>
</tr>
<tr style="border: 0;">
<th scope="row" style="text-align: left; font-weight: 400; border: 0; padding: 0.55em 0.75em;">MMMU-Pro</th>
<td style="text-align: center; font-variant-numeric: tabular-nums; background-color: rgba(108, 79, 224, 0.08); border: 0; padding: 0.55em 0.75em;">27.28</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">24.80</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">24.34</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><strong>28.67</strong></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><u>27.63</u></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">13.76</td>
</tr>
<tr style="border: 0;">
<th scope="row" style="text-align: left; font-weight: 400; border: 0; padding: 0.55em 0.75em;">MMBench</th>
<td style="text-align: center; font-variant-numeric: tabular-nums; background-color: rgba(108, 79, 224, 0.08); border: 0; padding: 0.55em 0.75em;">73.68</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><u>79.30</u></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">67.61</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">73.54</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><strong>82.73</strong></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">76.63</td>
</tr>
<tr style="border: 0;">
<th scope="row" style="text-align: left; font-weight: 400; border: 0; padding: 0.55em 0.75em;">SEEDBench (image)</th>
<td style="text-align: center; font-variant-numeric: tabular-nums; background-color: rgba(108, 79, 224, 0.08); border: 0; padding: 0.55em 0.75em;"><u>75.07</u></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><strong>75.27</strong></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">72.53</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">74.78</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">72.24</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">74.73</td>
</tr>
<tr style="border: 0;">
<th scope="row" style="text-align: left; font-weight: 400; border: 0; padding: 0.55em 0.75em;">RealWorldQA</th>
<td style="text-align: center; font-variant-numeric: tabular-nums; background-color: rgba(108, 79, 224, 0.08); border: 0; padding: 0.55em 0.75em;"><u>66.54</u></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">63.14</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">48.63</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">53.07</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">64.84</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><strong>72.81</strong></td>
</tr>
<tr style="border: 0;">
<th scope="row" style="text-align: left; font-weight: 400; border: 0; padding: 0.55em 0.75em;">MathVista (mini)</th>
<td style="text-align: center; font-variant-numeric: tabular-nums; background-color: rgba(108, 79, 224, 0.08); border: 0; padding: 0.55em 0.75em;"><u>62.50</u></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">58.80</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">51.00</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">54.30</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><strong>65.40</strong></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">53.90</td>
</tr>
<tr style="border: 0;">
<th scope="row" style="text-align: left; font-weight: 400; border: 0; padding: 0.55em 0.75em;">POPE</th>
<td style="text-align: center; font-variant-numeric: tabular-nums; background-color: rgba(108, 79, 224, 0.08); border: 0; padding: 0.55em 0.75em;"><strong>89.34</strong></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><u>84.62</u></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">78.78</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">81.24</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">83.34</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">83.52</td>
</tr>
</tbody>
<tbody style="border: 0;">
<tr style="border: 0;"><th scope="rowgroup" style="text-align: left; background-color: transparent; padding: 0.55em 0.75em; border: 0; font-weight: 600; padding-top: 1em; padding-bottom: 0.5em; border-top: 1px solid rgba(127, 127, 127, 0.25);">Documents and OCR</th><td aria-hidden="true" style="border: 0; padding: 0.55em 0.75em; background-color: rgba(108, 79, 224, 0.08); border-top: 1px solid rgba(127, 127, 127, 0.25);"></td><td colspan="5" aria-hidden="true" style="border: 0; padding: 0.55em 0.75em; border-top: 1px solid rgba(127, 127, 127, 0.25);"></td></tr>
<tr style="border: 0;">
<th scope="row" style="text-align: left; font-weight: 400; border: 0; padding: 0.55em 0.75em;">DocVQA</th>
<td style="text-align: center; font-variant-numeric: tabular-nums; background-color: rgba(108, 79, 224, 0.08); border: 0; padding: 0.55em 0.75em;">92.47</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><strong>92.71</strong></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">73.49</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">77.91</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><u>92.60</u></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">84.43</td>
</tr>
<tr style="border: 0;">
<th scope="row" style="text-align: left; font-weight: 400; border: 0; padding: 0.55em 0.75em;">InfoVQA</th>
<td style="text-align: center; font-variant-numeric: tabular-nums; background-color: rgba(108, 79, 224, 0.08); border: 0; padding: 0.55em 0.75em;">64.08</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><strong>74.55</strong></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">38.72</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">48.30</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><u>65.98</u></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">60.94</td>
</tr>
<tr style="border: 0;">
<th scope="row" style="text-align: left; font-weight: 400; border: 0; padding: 0.55em 0.75em;">OCRBench</th>
<td style="text-align: center; font-variant-numeric: tabular-nums; background-color: rgba(108, 79, 224, 0.08); border: 0; padding: 0.55em 0.75em;"><u>806</u></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><strong>863</strong></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">731</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">761</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">795</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">751</td>
</tr>
</tbody>
<tbody style="border: 0;">
<tr style="border: 0;"><th scope="rowgroup" style="text-align: left; background-color: transparent; padding: 0.55em 0.75em; border: 0; font-weight: 600; padding-top: 1em; padding-bottom: 0.5em; border-top: 1px solid rgba(127, 127, 127, 0.25);">Grounding</th><td aria-hidden="true" style="border: 0; padding: 0.55em 0.75em; background-color: rgba(108, 79, 224, 0.08); border-top: 1px solid rgba(127, 127, 127, 0.25);"></td><td colspan="5" aria-hidden="true" style="border: 0; padding: 0.55em 0.75em; border-top: 1px solid rgba(127, 127, 127, 0.25);"></td></tr>
<tr style="border: 0;">
<th scope="row" style="text-align: left; font-weight: 400; border: 0; padding: 0.55em 0.75em;">RefCOCO</th>
<td style="text-align: center; font-variant-numeric: tabular-nums; background-color: rgba(108, 79, 224, 0.08); border: 0; padding: 0.55em 0.75em;">73.55</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><strong>78.76</strong></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">63.18</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">63.36</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">—<sup>*</sup></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><u>74.53</u></td>
</tr>
</tbody>
<tbody style="border: 0;">
<tr style="border: 0;"><th scope="rowgroup" style="text-align: left; background-color: transparent; padding: 0.55em 0.75em; border: 0; font-weight: 600; padding-top: 1em; padding-bottom: 0.5em; border-top: 1px solid rgba(127, 127, 127, 0.25);">Video</th><td aria-hidden="true" style="border: 0; padding: 0.55em 0.75em; background-color: rgba(108, 79, 224, 0.08); border-top: 1px solid rgba(127, 127, 127, 0.25);"></td><td colspan="5" aria-hidden="true" style="border: 0; padding: 0.55em 0.75em; border-top: 1px solid rgba(127, 127, 127, 0.25);"></td></tr>
<tr style="border: 0;">
<th scope="row" style="text-align: left; font-weight: 400; border: 0; padding: 0.55em 0.75em;">MVBench</th>
<td style="text-align: center; font-variant-numeric: tabular-nums; background-color: rgba(108, 79, 224, 0.08); border: 0; padding: 0.55em 0.75em;">58.03</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><u>66.43</u></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">42.05</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">47.10</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">52.60</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><strong>71.65</strong></td>
</tr>
<tr style="border: 0;">
<th scope="row" style="text-align: left; font-weight: 400; border: 0; padding: 0.55em 0.75em;">Video-MME (no subtitles)</th>
<td style="text-align: center; font-variant-numeric: tabular-nums; background-color: rgba(108, 79, 224, 0.08); border: 0; padding: 0.55em 0.75em;"><u>57.70</u></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><strong>62.59</strong></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">47.70</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">52.44</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">53.44</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">55.89</td>
</tr>
</tbody>
<tbody style="border: 0;">
<tr style="border: 0;"><th scope="rowgroup" style="text-align: left; background-color: transparent; padding: 0.55em 0.75em; border: 0; font-weight: 600; padding-top: 1em; padding-bottom: 0.5em; border-top: 1px solid rgba(127, 127, 127, 0.25);">Spatial and embodied</th><td aria-hidden="true" style="border: 0; padding: 0.55em 0.75em; background-color: rgba(108, 79, 224, 0.08); border-top: 1px solid rgba(127, 127, 127, 0.25);"></td><td colspan="5" aria-hidden="true" style="border: 0; padding: 0.55em 0.75em; border-top: 1px solid rgba(127, 127, 127, 0.25);"></td></tr>
<tr style="border: 0;">
<th scope="row" style="text-align: left; font-weight: 400; border: 0; padding: 0.55em 0.75em;">ERQA</th>
<td style="text-align: center; font-variant-numeric: tabular-nums; background-color: rgba(108, 79, 224, 0.08); border: 0; padding: 0.55em 0.75em;">38.50</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><u>40.25</u></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">35.25</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">34.50</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">35.75</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><strong>41.00</strong></td>
</tr>
<tr style="border: 0;">
<th scope="row" style="text-align: left; font-weight: 400; border: 0; padding: 0.55em 0.75em;">EmbSpatial</th>
<td style="text-align: center; font-variant-numeric: tabular-nums; background-color: rgba(108, 79, 224, 0.08); border: 0; padding: 0.55em 0.75em;"><u>70.55</u></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">65.58</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">47.64</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">62.28</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">57.83</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><strong>76.24</strong></td>
</tr>
<tr style="border: 0;">
<th scope="row" style="text-align: left; font-weight: 400; border: 0; padding: 0.55em 0.75em;">RoboSpatial</th>
<td style="text-align: center; font-variant-numeric: tabular-nums; background-color: rgba(108, 79, 224, 0.08); border: 0; padding: 0.55em 0.75em;">49.14</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><strong>52.00</strong></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">27.71</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">38.29</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">41.14</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><u>50.57</u></td>
</tr>
</tbody>
</table>
</div>


### Audio

<div role="region" aria-label="Audio benchmarks" tabindex="0" style="width: 100%; max-width: 100%; overflow-x: auto; -webkit-overflow-scrolling: touch;">
<table style="width: 100%; min-width: 36rem; display: table; border: 0; border-radius: 0; border-collapse: collapse; font-size: calc(0.875em - 1pt);">
<thead style="border: 0;">
<tr style="border: 0;">
<th scope="col" style="text-align: left; padding: 0.55em 0.75em; border: 0; border-bottom: 1px solid rgba(127, 127, 127, 0.55);">Benchmark</th>
<th scope="col" style="text-align: center; background-color: rgba(108, 79, 224, 0.08); padding: 0.55em 0.75em; border: 0; border-bottom: 1px solid rgba(127, 127, 127, 0.55);"><strong>LFM3-Omni<br>2B</strong></th>
<th scope="col" style="text-align: center; padding: 0.55em 0.75em; border: 0; border-bottom: 1px solid rgba(127, 127, 127, 0.55);">Qwen2.5-Omni<br>3B</th>
<th scope="col" style="text-align: center; padding: 0.55em 0.75em; border: 0; border-bottom: 1px solid rgba(127, 127, 127, 0.55);">Gemma 4<br>E2B</th>
<th scope="col" style="text-align: center; padding: 0.55em 0.75em; border: 0; border-bottom: 1px solid rgba(127, 127, 127, 0.55);">Gemma 4<br>E4B</th>
<th scope="col" style="text-align: center; padding: 0.55em 0.75em; border: 0; border-bottom: 1px solid rgba(127, 127, 127, 0.55);">Phi-4<br>multimodal</th>
</tr>
</thead>
<tbody style="border: 0;">
<tr style="border: 0;">
<th scope="row" style="text-align: left; font-weight: 400; border: 0; padding: 0.55em 0.75em;">Total parameters</th>
<td style="text-align: center; font-variant-numeric: tabular-nums; background-color: rgba(108, 79, 224, 0.08); border: 0; padding: 0.55em 0.75em;">2.2B</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">6B</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">5B</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">8B</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">5.6B</td>
</tr>
</tbody>
<tbody style="border: 0;">
<tr style="border: 0;">
<th scope="row" style="text-align: left; font-weight: 400; border: 0; padding: 0.55em 0.75em;">English ASR<br><small>WER ↓</small></th>
<td style="text-align: center; font-variant-numeric: tabular-nums; background-color: rgba(108, 79, 224, 0.08); border: 0; padding: 0.55em 0.75em;">6.36</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><u>6.30</u></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">8.26</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">7.82</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><strong>5.14</strong></td>
</tr>
<tr style="border: 0;">
<th scope="row" style="text-align: left; font-weight: 400; border: 0; padding: 0.55em 0.75em;">Multilingual ASR<br><small>WER / CER ↓</small></th>
<td style="text-align: center; font-variant-numeric: tabular-nums; background-color: rgba(108, 79, 224, 0.08); border: 0; padding: 0.55em 0.75em;"><u>6.72</u></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">7.41</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">8.08</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><strong>6.54</strong></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">—<sup>†</sup></td>
</tr>
<tr style="border: 0;">
<th scope="row" style="text-align: left; font-weight: 400; border: 0; padding: 0.55em 0.75em;">English TTS<br><small>UTMOSv2 ↑</small></th>
<td style="text-align: center; font-variant-numeric: tabular-nums; background-color: rgba(108, 79, 224, 0.08); border: 0; padding: 0.55em 0.75em;"><strong>3.80</strong></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><u>3.42</u></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">—</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">—</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">—</td>
</tr>
<tr style="border: 0;">
<th scope="row" style="text-align: left; font-weight: 400; border: 0; padding: 0.55em 0.75em;">VoiceBench ↑</th>
<td style="text-align: center; font-variant-numeric: tabular-nums; background-color: rgba(108, 79, 224, 0.08); border: 0; padding: 0.55em 0.75em;">64.73</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><u>66.37</u></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">63.28</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><strong>72.90</strong></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">64.97</td>
</tr>
<tr style="border: 0;">
<th scope="row" style="text-align: left; font-weight: 400; border: 0; padding: 0.55em 0.75em;">Tool calling<br><small>internal eval ↑</small></th>
<td style="text-align: center; font-variant-numeric: tabular-nums; background-color: rgba(108, 79, 224, 0.08); border: 0; padding: 0.55em 0.75em;">52.20</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">44.30</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><u>55.60</u></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><strong>72.40</strong></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">—</td>
</tr>
<tr style="border: 0;">
<th scope="row" style="text-align: left; font-weight: 400; border: 0; padding: 0.55em 0.75em;">Audio understanding<br><small>MMAR ↑</small></th>
<td style="text-align: center; font-variant-numeric: tabular-nums; background-color: rgba(108, 79, 224, 0.08); border: 0; padding: 0.55em 0.75em;"><u>51.40</u></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;"><strong>54.41</strong></td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">44.49</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">49.10</td>
<td style="text-align: center; font-variant-numeric: tabular-nums; border: 0; padding: 0.55em 0.75em;">44.69</td>
</tr>
</tbody>
</table>
</div>


<details>
<summary><small>Evaluation details</small></summary>

<p><small>All scores were measured by Liquid AI. Every model uses its own chat template, prompt format, and image and video preprocessing, with greedy decoding for text output. LFM3-Omni-2B speech output uses <code>audio_temperature=0.4</code> and <code>audio_repetition_penalty=1.2</code>.</small></p>

<p><small>
<strong>Scale</strong>: all scores are on a 0–100 scale except OCRBench, which is out of 1,000. DocVQA and InfoVQA use ANLS.<br>
<strong>Daily-Omni</strong>, <strong>WorldSense</strong>, and <strong>AV-SpeakerBench</strong>: video with its soundtrack, multiple-choice accuracy.<br>
<strong>OmniBench</strong>: an image and an audio clip with each question, scored with the official prompt and answer parser on all 1,142 questions. <strong>OmniBench (speech)</strong> is the subset whose audio is speech.<br>
<strong>Spoken visual QA</strong>: the image benchmarks with each question given only as speech.<br>
<strong>POPE</strong>: F1 over all examples in the adversarial, popular, and random subsets.<br>
<strong>MMMU-Pro</strong>: vision setting, with the question and options shown in the image.<br>
<strong>RefCOCO</strong>: boxes are read in each model's native coordinate format. <sup>*</sup>Phi-4-multimodal does not return bounding boxes in a parseable format (72% of answers fail to parse), so its score is not reported.<br>
<strong>MVBench</strong> and <strong>Video-MME</strong>: video frames without audio, sampled with each model's recommended settings. Video-MME is without subtitles.<br>
<strong>RoboSpatial</strong>: RoboSpatial-Home, scored with the official RoboSpatial-Eval rule, with points read in each model's native coordinate format.<br>
<strong>English ASR</strong>: average word error rate (WER) in percent on the ESB / Open ASR benchmark suite, lower is better.<br>
<strong>Multilingual ASR</strong>: mean error rate in percent on FLEURS across English, Spanish, Portuguese, German, French, Italian, Chinese, Japanese, and Korean, lower is better. WER is used for all languages except Chinese, Japanese, and Korean, which use character error rate (CER). <sup>†</sup>Phi-4-multimodal does not support Korean speech, so its score is not reported.<br>
<strong>English TTS</strong>: UTMOSv2 speech quality score on Seed-TTS-eval English, on a 1–5 scale, higher is better.<br>
<strong>VoiceBench</strong>: audio input, text output, mean of the 9 subsets on a 0–100 scale (the three 1–5 judge scores are multiplied by 20), higher is better.<br>
<strong>Tool calling</strong>: internal evaluation with audio input, percentage score, higher is better.<br>
<strong>Audio understanding</strong>: MMAR, percentage score, higher is better.
</small></p>

</details>

## Citation
<hr style="margin: -0.7em 0 1.2em 0; border: 0; border-top: 1px solid rgba(127, 127, 127, 0.3);" />

```bibtex
@article{liquidAI2026LFM3,
  author  = {Liquid AI},
  title   = {Introducing LFM3: Every Modality for Specialized Intelligence},
  journal = {Liquid AI Blog},
  year    = {2026},
  note    = {www.liquid.ai/blog/lfm3},
}
```

For questions and feedback, join [Discord](https://discord.com/invite/liquid-ai). For custom deployment and fine-tuning support, [contact Liquid AI](https://www.liquid.ai/connect?utm_source=huggingface&utm_medium=referral&utm_campaign=huggingface).
