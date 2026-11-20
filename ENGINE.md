# XeQ Runtime

XeQ Runtime is the OpenCL inference engine for XeFP4 models on Intel Arc GPUs. Batch 1; greedy decoding by default, sampling on request.

| Model | Files | Notes |
|---|---|---|
| Qwen3.5-9B | `q35_xe8g32e.bin/.json`, `q35_mtp_xe8.bin/.json`, `vocab_order.npy` | MTP speculative decoding (same tokens as plain decoding) |
| Gemma 4 26B-A4B-it | Speed `g4it_xe8.bin/.json` or Quality `g4it_xe8_ne8.bin/.json`; optional `vision/g4v_i8.bin/.json`, optional `assistant/` | image input with the vision encoder; MTP with the original assistant model |

## Requirements

- Windows 10/11 and an Intel Arc GPU with OpenCL. Tested on:
  - Arc A770 16 GB (Xe-HPG), driver 32.0.101.9150
  - Arc 140T (Xe-LPG+, no split DPAS), driver 32.0.101.9033
- Python 3.10 or later

```bash
pip install -r requirements.txt
```

## Model folder

Each model folder holds the XeFP4 `.bin`/`.json` pair; the original model's `tokenizer.json` and `chat_template.jinja` go next to it or in the parent folder:

```
Qwen3.5-9B-XeFP4/
  q35_xe8g32e.bin  q35_xe8g32e.json  q35_mtp_xe8.bin  q35_mtp_xe8.json  vocab_order.npy
  tokenizer.json  chat_template.jinja
gemma-4-26B-A4B-it-XeFP4/
  tokenizer.json  chat_template.jinja
  speed/    g4it_xe8.bin  g4it_xe8.json
  quality/  g4it_xe8_ne8.bin  g4it_xe8_ne8.json
  vision/   g4v_i8.bin  g4v_i8.json
  assistant/  model.safetensors  config.json   (original Gemma 4 MTP assistant, optional)
```
`tokenizer.json` and `chat_template.jinja` may sit in the model folder or one level up. For Gemma 4, pass `speed/` or `quality/` as the model folder.

The engine finds the main model, the Qwen MTP head, the Gemma vision encoder and the Gemma MTP assistant (`vision/` and
`assistant/` next to or inside the folder) by themselves. MTP is on when its files are present; `--no-mtp` turns it off.

## Commands

```bash
python -m xeq devices
python -m xeq run   <model folder> -p "Explain Rayleigh scattering in two sentences."
python -m xeq run   <gemma folder> --image photo.png -p "Describe the image."
python -m xeq chat  <model folder> --ctx 16384
python -m xeq serve <model folder> --port 8080 --ctx 32768
python -m xeq selftest <model folder>
python -m xeq bench <model folder> --out bench.json
```

| Option | Meaning |
|---|---|
| `--ctx N` | KV cache length (default 4096). Memory grows with it; see the model card for what fits in 16 GB. |
| `-n N` | maximum new tokens (default 1024) |
| `--think` | reasoning on (the chat template's `enable_thinking`) |
| `--system TEXT` | system message |
| `--image FILE` | Gemma 4 only, repeatable; loads the vision encoder |
| `--vision` | `serve`: load the vision encoder so requests may contain images |
| `--no-mtp` | plain decoding (no MTP) |
| `--kv fp16\|fp32` | Gemma 4 KV cache type (default fp16). fp16 halves the KV memory and is faster from about 1K tokens of context (Arc 140T: +8% at 1K, +23% at 32K; Arc A770: +3% at 1K-4K, +9% at 32K, -2% at a near-empty context; prefill +1-8%); perplexity is unchanged (6.9936 vs 6.9926, Speed). `fp32` keeps the previous behaviour |
| `--prefill-chunk auto\|small\|N` | prompt tokens per prefill pass. `auto` (default): 2048 where it helps and fits — Qwen3.5 always (about +1.1 GiB at 32K), Gemma 4 on GPUs that report 20 GiB or more (otherwise 512). `small` forces 512 / 1024 (less memory). Output tokens do not depend on the chunk size, except that the first 512 tokens of a Gemma 4 prompt take a different attention kernel at 512 |
| `--temperature T` | sampling temperature; 0 or unset = greedy |
| `--top-p P`, `--top-k K`, `--min-p P` | nucleus, top-k (at most 64) and min-p filters |
| `--presence-penalty X`, `--repetition-penalty X` | penalties (presence: generated tokens; repetition: prompt and generated tokens) |
| `--seed N` | makes a sampled run reproducible |

`serve` exposes `GET /v1/models` and `POST /v1/chat/completions` (OpenAI format, with streaming and tool calls).
It handles one request at a time. `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`, `repetition_penalty` and `seed`
are honoured; without `temperature` decoding is greedy. Images are accepted as `data:` URLs.
Both models reuse the work for the prefix shared with the previous request (chat cache), in `chat` and in `serve`:
Gemma 4 keeps its KV cache; Qwen3.5 also keeps snapshots of its linear-attention state (64 tokens before the end of
each prompt and at the end, about 50 MB each, 4 kept). Only the previous prompt is reused; the previous answer is
processed again, so a cached request gives the same tokens as the full prompt (checked over 4-turn conversations,
MTP on and off). On an Arc 140T a follow-up turn in a 5K-token conversation is prefilled in about 0.7 s instead of
7.7 s (Qwen3.5-9B) and in 1.0–1.6 s instead of 9–12 s (Gemma 4 Quality).

`selftest` generates 64 tokens for three fixed prompts and compares the token hashes with `xeq/reference.json`
(the same tokens come out on A770 and 140T). For Qwen3.5 it also checks that MTP returns the same tokens as plain decoding.

## Environment variables

| Variable | Meaning |
|---|---|
| `XEQ_DEVICE` | GPU to use: an index from `devices` or part of its name. Default: the first Arc GPU. |
| `XEQ_*` (others) | internal tuning switches. They are not a supported interface; the defaults reproduce the published benchmarks. |

## MTP

Both models decode with multi-token prediction when its files are present. The output is the same as without MTP
(greedy: identical tokens; sampling: the same distribution).
- Qwen3.5: the MTP head drafts one token per step.
- Gemma 4: the assistant model drafts k tokens (k adapts between 3 and 5) and the main model verifies k + 1 at once.
  One verification costs about three plain decoding steps with the Speed configuration and about two with Quality, so
  MTP pays off only when most drafts are accepted (code, reasoning). The engine compares both speeds while decoding and
  falls back to plain decoding on text where drafts are rarely accepted, retrying MTP from time to time.
- Memory: the Gemma assistant adds about 0.4 GB. The engine warns when the allocation comes close to the GPU memory.

### Recommended Gemma 4 setup on a 16 GB GPU

| Use | Setup |
|---|---|
| Default: chat, code, reasoning, long documents | Quality + MTP (FP16 KV cache, the default), `--ctx 32768` |
| The GPU also drives displays or runs other programs | the same with `--ctx 16384` (32K uses 15.5 of 15.6 GiB on an Arc A770) |
| Image input | Speed (`--image`), MTP optional |

Measured on an Arc A770 with a 30K-token prompt: Quality + MTP 60-64 tok/s, Speed + MTP 65 tok/s; on short answers
Quality + MTP is faster than Speed in every task tested (95-153 vs 92-142 tok/s) and closer to the original model.

### Integrated GPUs (Arc 140T and similar)

The GPU shares system memory; Windows lets it use a fixed share of it (by default about half). On a 32 GB machine,
raise the share to **75%** in the Intel graphics software (*Shared GPU memory* / *GPU memory allocation*; a reboot is
needed) so that Gemma 4 Quality + MTP fits a 32K context together with the faster 2048-token prefill. Check the result with
`python -m xeq devices` (it prints the memory the GPU reports). Measured on an Arc 140T with 32 GB (70% share, 21.8 GiB):

| Gemma 4, `--ctx 32768`, MTP | `--prefill-chunk auto` (2048) | `--prefill-chunk small` (512) |
|---|---|---|
| Quality: memory / 4K-token prefill | 16.3 GiB / about 640 tok/s | 15.3 GiB / about 460 tok/s |
| Speed: memory / 4K-token prefill | 15.5 GiB / about 620 tok/s | 14.5 GiB / about 420 tok/s |

With the default share (about 16 GiB reported) `auto` keeps the 512-token chunk; leave the remaining system memory for Windows
and other programs. Qwen3.5-9B needs about 10.3 GiB at 32K and prefills about 670–780 tok/s (4K tokens) on the 140T.

## Sampling

Greedy decoding (the default) suits code, factual answers, translation, summaries and tool calls: the answer is
repeatable and MTP gains the most. For conversation, creative writing and reasoning (`--think`), use the original
model's recommended sampling settings (see the model card); long reasoning under greedy decoding can fall into loops.

Sampling runs on the GPU. With MTP (Qwen3.5) the greedy draft is accepted with the target probability and a rejection
draws from the target distribution without the draft token, so the output distribution is the same as without MTP.
MTP gains less when sampling, because fewer drafts are accepted.

Filters: penalties, then temperature, top-k, top-p, min-p. Without top-k/top-p/min-p the token is drawn from the
whole vocabulary. With them the 64 most likely tokens are the candidates: `top_k` is capped at 64, and a nucleus that
would need more than 64 tokens is cut there.

## Limits

- Batch 1.
- Gemma 4 uses an FP16 KV cache by default (`--kv fp32` for FP32). Qwen3.5 keeps FP32.
- The chat cache follows the most recent request only; switching between conversations recomputes the prompt.
- The prefill kernels of 2026-11-20 were checked on the Arc A770 and 140T (same output tokens as the previous kernels; A770 prefill: Qwen3.5 +16–25%, Gemma 4 Speed +4–6%). The skinny expert kernel used at a 512-token chunk runs only on GPUs without split DPAS.

XeQ Runtime and XeFP4 are independent projects by KINOQ, not affiliated with or endorsed by Intel. "Intel", "Arc" and "Xe" are
trademarks of Intel Corporation and are used here only to name the supported hardware.
