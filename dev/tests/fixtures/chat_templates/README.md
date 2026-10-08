These unmodified upstream templates exercise the startup probe and the structural
patch for system messages after the first message (`server/chat_templates.py`,
tested by `dev/tests/engine/test_chat_templates.py`). They are test fixtures,
not templates distributed with the runtime.

- `qwen36.jinja`: [mlx-community/Qwen3.6-35B-A3B-4bit](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit/blob/38740b847e4cb78f352aba30aa41c76e08e6eb46/chat_template.jinja),
  revision `38740b847e4cb78f352aba30aa41c76e08e6eb46`.
- `qwen38.jinja`: [mlx-community/Qwen3.8-27B-4bit](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit/blob/3e6447f082e89cc7f0bc6e5441afd38dfce760ff/chat_template.jinja),
  revision `3e6447f082e89cc7f0bc6e5441afd38dfce760ff`.
- `qwen36_gguf.jinja`: embedded `tokenizer.chat_template` of
  `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` in `unsloth/Qwen3.6-35B-A3B-GGUF`,
  file SHA-256 `ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61`.
- `qwen38_gguf.jinja`: embedded `tokenizer.chat_template` of
  `Qwen3.8-27B-UD-Q4_K_M.gguf` in `unsloth/Qwen3.8-27B-GGUF`,
  file SHA-256 `322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482`.
- `gemma4.jinja`: [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it/blob/4d7ae4984b7db7de8f8457170b3f1a419ee76d52/chat_template.jinja),
  revision `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`,
  file SHA-256 `ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4`.
- `diffusiongemma.jinja`: [google/diffusiongemma-26B-A4B-it](https://huggingface.co/google/diffusiongemma-26B-A4B-it/blob/f7f5b7f5fa82ffc52addd066915886d497f5517b/chat_template.jinja),
  revision `f7f5b7f5fa82ffc52addd066915886d497f5517b`,
  file SHA-256 `9aeb7eac68ad87bba7567e9d4597ff203e5609f1b427d9e823437d0142cc61bf`.
  Same Gemma 4 family as `gemma4.jinja`, but the non-thinking generation
  prompt leaves the model turn bare: `gemma4.jinja` still appends an empty
  closed thought channel, DiffusionGemma does not. Both spellings are in
  the wild; `test_gemma4.py` covers the former, `test_diffusiongemma.py`
  the latter.
