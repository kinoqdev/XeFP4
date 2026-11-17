# XeFP4 and XeQ Runtime

**XeFP4** is an open 4-bit weight format for Intel Arc GPUs. **XeQ Runtime** (`xeq`) is the OpenCL inference engine that runs it.

XeFP4 stores LLM weights at about 4.1 bits per weight. Each weight combines:
- an E2M1 (FP4) value;
- a 3-bit integer factor per block of 32 weights;
- one FP16 scale per output row.

Every stored value is a small integer, so Intel XMX (DPAS) units can run the inner product in integer arithmetic.
XeQ Runtime is written from scratch in OpenCL. It does not depend on llama.cpp, SYCL or PyTorch at run time.

> XeFP4 is an independent project by KINOQ. It is not affiliated with or endorsed by Intel.
> "Intel", "Arc" and "Xe" are trademarks of Intel Corporation and are used here only to name the supported hardware.

## Models

| Model | Hugging Face | Size | Notes |
|---|---|---|---|
| Qwen3.5-9B | [KINOQ/Qwen3.5-9B-XeFP4](https://huggingface.co/KINOQ/Qwen3.5-9B-XeFP4) | 5.1 GB (+ 0.12 GB MTP) | XeFP4-8 everywhere, INT8 lm_head, MTP speculative decoding |
| Gemma 4 26B-A4B-it | [KINOQ/gemma-4-26B-A4B-it-XeFP4](https://huggingface.co/KINOQ/gemma-4-26B-A4B-it-XeFP4) | 14.2 GB (Quality) / 13.4 GB (Speed) | MoE; optional vision encoder; MTP with the original assistant model (+0.84 GB) |

## Performance (Intel Arc A770 16 GB, batch 1, greedy)

Scripted benchmark, 2026-11-17, driver 32.0.101.9150 (`python -m xeq bench`). Gemma 4 uses an FP16 KV cache (the default).

| Model | Configuration | Decode tok/s | Comparison |
|---|---|---|---|
| Qwen3.5-9B | MTP off | 65.7 | llama.cpp SYCL Q4_K_M: 37.9 |
| Qwen3.5-9B | MTP on | 94.2 | OpenVINO INT4 + MTP: 56.7 |
| Qwen3.5-9B | MTP on, 32K context | 62.7 | |
| Gemma 4 26B-A4B-it | Speed | 92.1 | llama.cpp SYCL IQ4_XS: 32.4 |
| Gemma 4 26B-A4B-it | Speed, 16K context | 76.6 | llama.cpp SYCL IQ4_XS at 16K: 19.3 |
| Gemma 4 26B-A4B-it | Speed, 32K context | 68.8 | |
| Gemma 4 26B-A4B-it | Quality | 72.1 | |
| Gemma 4 26B-A4B-it | Quality + MTP, thinking / code | about 150 / 120 | |

- Prefill, Qwen3.5-9B: 2,499 tok/s at 2K tokens, 1,738 tok/s at 32K (llama.cpp Q4_K_M pp512: 836).
- Prefill, Gemma 4: about 1,650 tok/s at 512 tokens, 1,330 at 32K (llama.cpp IQ4_XS pp512: 694).
- MTP gives the same output tokens as greedy decoding without MTP.
- Decoding is greedy by default; temperature, top-p/top-k/min-p, penalties and seed are available (see [ENGINE.md](ENGINE.md)).
- MTP speed depends on the text: large gains for thinking and code, little or none for short prose answers, where the engine switches to plain decoding by itself. The Qwen 32K figure was measured on Korean prose (`bench --text <file>`); Gemma MTP figures come from `bench --mtp`. See the model cards for per-task tables.

## Accuracy

Perplexity on 16 FineWeb documents × 512 tokens. Lower is better.

| Model | BF16 | XeFP4 | GGUF Q4_K_M | GGUF IQ4_XS |
|---|---|---|---|---|
| Qwen3.5-9B | 11.164 | 11.682 (+4.6%) | 11.267 (+0.9%) | 11.341 (+1.6%) |
| Gemma 4 26B-A4B (base) | 6.670 | 6.981 (+4.7%, Speed) / 6.731 (+0.9%, Quality) | — | 6.827 (+2.4%) |

- On Qwen3.5-9B, XeFP4 loses more accuracy than the GGUF K-quants. It trades that for speed: it uses no calibration and a fixed 4.1-bit layout. Improving accuracy without changing the format is ongoing work.
- On Gemma 4, the Quality configuration (INT8 outside the experts) is closer to BF16 than GGUF IQ4_XS.

## Supported hardware

| GPU | Architecture | Status |
|---|---|---|
| Intel Arc A770 / A750 (Alchemist) | Xe-HPG | tested (A770) |
| Intel Arc 140T (Arrow Lake-H iGPU) | Xe-LPG+ | tested: same output tokens as the A770; Qwen3.5-9B 23 tok/s with MTP; Gemma 4 Speed 23.6 tok/s, Quality + MTP 24–39 tok/s |
| Intel Arc B-series, Lunar Lake (Xe2) | Xe2 | not tested |

Requirements: Windows 10/11, Intel GPU OpenCL driver (tested: 32.0.101.9150 on the A770, 32.0.101.9033 on the 140T), Python 3.10 or later, `numpy`, `pyopencl`, `tokenizers`, `jinja2` (`pillow` for images). Linux is untested.

## Repository layout

| Path | Content |
|---|---|
| `spec/XEFP4_SPEC.md` | format specification v1.0 |
| `spec/vectors/` | conformance test vectors |
| `tools/xefp4_ref.py` | reference encoder and decoder (NumPy) |
| `tools/check_vectors.py` | conformance checker |
| `xeq/` | XeQ Runtime, the OpenCL inference engine (`python -m xeq`) |
| [`ENGINE.md`](ENGINE.md) | engine commands, model folder layout, server |

The converter (BF16 safetensors → XeFP4) is planned for the second release.

## Quick start

```bash
pip install -r requirements.txt
python -m xeq devices
python -m xeq selftest <path to Qwen3.5-9B-XeFP4>
python -m xeq chat <path to Qwen3.5-9B-XeFP4>
python -m xeq chat <path to gemma-4-26B-A4B-it-XeFP4>/quality --ctx 32768
```

Download the model folders from Hugging Face first. `selftest` checks that your GPU produces the reference tokens.
See [ENGINE.md](ENGINE.md) for the server (`serve`, OpenAI-compatible), image input, sampling, benchmarks and the recommended Gemma 4 setup on a 16 GB GPU.

## How XeFP4 differs from MXFP4 and NVFP4

| | XeFP4 | MXFP4 | NVFP4 |
|---|---|---|---|
| values | E2M1 | E2M1 | E2M1 |
| block | 32 | 32 | 16 |
| block scale | 3-bit integer m/8 | 8-bit power of two | 8-bit FP8 |
| outer scale | FP16 per row | — | FP32 per tensor |
| bits/weight | ≈ 4.10 | 4.25 | 4.5 |
| target | INT8 XMX/DPAS (no FP4 units needed) | FP4 tensor cores | FP4 tensor cores |

## License

The code (XeQ Runtime, tools) and the specification are licensed under Apache-2.0 (see `LICENSE`).
The model weights on Hugging Face follow the licenses of their source models (both Apache-2.0). See `NOTICE`.
