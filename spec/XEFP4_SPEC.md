# XeFP4 weight format — specification, version 1 (draft)

> Status: **draft for the first public release (v1.0).** It describes the bytes read by the XeFP4 reference engine
> (RC1, Qwen3.5-9B on Intel Arc A770), with the converters as source of truth.
>
> XeFP4 is an independent project by KINOQ. It is not affiliated with or endorsed by Intel.
> "Xe" refers only to the Intel GPU architectures the format was designed for.

## 1. Overview

XeFP4 stores the weights of linear layers at about 4.1 bits per weight. Each weight is built from three parts:
- a 4-bit grid code;
- a 3-bit factor per block of 32 weights;
- one FP16 scale per output row.

All grid values and factors are small integers. On integer matrix engines (Intel XMX/DPAS), a kernel can therefore
do the whole inner product in integers and apply one floating-point scale per row at the end. A kernel may also
decode to float and multiply with float activations; the engine does this for single-token decoding.

| | XeFP4 | MXFP4 (OCP MX) | NVFP4 |
|---|---|---|---|
| 4-bit values | E2M1 grid, stored ×2 as integers 0..12 | E2M1 | E2M1 |
| block | 32 | 32 | 16 |
| block scale | 3-bit integer factor m/8, m = 1..8 | 8-bit power of two (E8M0) | 8-bit FP8 (E4M3) |
| top-level scale | FP16 per output row | — | FP32 per tensor |
| bits per weight | ≈ 4.10 | 4.25 | 4.5 |

## 2. Weight encoding

Version 1.0 defines one variant, `fmt: "xe8"` (XeFP4-8). The value `"xe4"` (an integer −7..7 grid) is reserved and
outside v1.0.

A linear weight `W` has shape `[N, K]`: N output rows and K inputs. K is split into blocks of `G` consecutive values.
`G = 32` in version 1, and K must be a multiple of G.

```
W[n, k] ≈ rs[n] · (m[n, b] / 8) · g(code[n, k])          b = k // G
```

| Part | Range | Stored as |
|---|---|---|
| `g` grid value | {0, ±1, ±2, ±3, ±4, ±6, ±8, ±12} (E2M1 × 2) | 4-bit code |
| `m` block factor | 1..8 | 3 bits, `m − 1` |
| `rs` row scale | > 0 | FP16 |

### 2.1 4-bit code

- Bits 0–2: magnitude index `i`, giving the value `[0, 1, 2, 3, 4, 6, 8, 12][i]`.
- Bit 3: sign (1 = negative).
- Zero is always stored with sign 0, so code 8 never occurs.

### 2.2 Integer form

`g · m` is an integer with `|g · m| ≤ 96`, so it fits in INT8. Then:

```
Σ_k x[k] · W[n, k] = (rs[n] / 8) · Σ_k x[k] · (g · m)[n, k]
```

With INT8 activations `x = sx · xq`, everything up to the final multiply by `rs[n] · sx / 8` is integer arithmetic.

### 2.3 Encoder (deterministic, no calibration data)

The encoder is specified exactly, so two conforming encoders produce identical bytes.

1. **Row scale:** `rs[n] = fp16(max(max_k |W[n,k]| / 12, 1e-12))` (round to nearest). All later steps use the FP16-rounded value.
   - FP16 subnormal values are allowed, and decoders must decode them correctly.
   - A row whose scale rounds to 0 decodes to all zeros, whatever its codes and factors. This happens when max |W| is below about 7e-7. In such a row the codes and factors carry no meaning; encoders should write code 0 and m = 1 (stored 0), and conformance checks ignore them.
2. **Block factor:** for each block, set `Wb = W / rs`, then try `m = 1, 2, …, 8` in this order. For each `m`:
   1. Let `z = Wb / (m / 8)`.
   2. Map each `|z|` to a magnitude index: the number of boundaries in {0.5, 1.5, 2.5, 3.5, 5, 7, 10} that are *strictly smaller* than `|z|`. A value exactly on a boundary therefore goes to the smaller magnitude; anything above 10 maps to 12.
   3. Compute the block error `Σ (g · m/8 − Wb)²`.
   
   Keep the `m` with the smallest error. Comparison is strict, so **on an exact tie the smaller m wins**.
3. **Codes:** recompute `z` with the chosen `m` and store sign and magnitude index as in 2.1.

**Decoding is exact.** `rs` has an 11-bit significand, `m` at most 3 significant bits and `g` at most 2 (3 = 11₂, 6 = 110₂, 12 = 1100₂). The
product `rs · m · g / 8` therefore fits in a float32 significand, and every multiplication order gives the same
float32 value.

**Note on error sums.** Two encoders can disagree only when two candidate factors give block errors closer than the
rounding error of the sum. The reference encoder (`tools/xefp4_ref.py`) computes each term in float32 and sums in
float64. A conforming encoder must reproduce the test vectors (section 7) exactly. On real weights it may differ in
such near-tie blocks; decoders are unaffected. Measured on Qwen3.5-9B MLP weights against the RC1 converter (float32 sums): 26 of 1,835,008 blocks (0.0014%) chose a different factor.

## 3. Packing

Arrays are row-major and little-endian.

| Tensor suffix | dtype | shape | content |
|---|---|---|---|
| `.codes` | uint8 | `[N, K/2]` | byte `j` = `code[2j] | (code[2j+1] << 4)`: **even column in the low nibble** |
| `.fac` | uint32 | `[N, FW]` | per row, the 3-bit fields `m − 1` of blocks 0, 1, 2, … concatenated **LSB first**: bit `j` of block `b` is bit `3b + j` of the row's bit string, i.e. bit `(3b+j) mod 32` of word `⌊(3b+j)/32⌋`. `FW = ⌈3·(K/G) / 32⌉`; unused high bits of the last word are 0. |
| `.rs` | float16 | `[N]` | row scale |

## 4. Other tensor kinds

| kind | Tensors | Meaning |
|---|---|---|
| `codes` / `fac` / `rs` | `<name>.codes/.fac/.rs` | XeFP4 linear layer (sections 2–3) |
| `ecodes` / `efac` / `ers` | `embed.ecodes/.efac/.ers` | XeFP4 embedding table (same rules, rows = tokens) |
| `i8` + `f32` | `<name>.q` int8 `[N, K]`, `<name>.s` float32 `[N]` | INT8 row-scaled matrix, decoded as `W ≈ s · q`. The spec fixes only the stored format; how `s` was chosen is recorded in the header (`i8_scale`, section 5). The reference converters use `s = max|W[n,:]| / 127` and `q = round_half_even(W / s)` clamped to −127..127. Qwen keeps `s` exact in FP32; Gemma rounds `s` to FP16 first. |
| `f32` / `bf16` | `<name>` | stored unchanged (norms, router, conv, gates, …) |

## 5. Container

A model is a pair of files:

- **`<model>.bin`:** raw tensor data. Each tensor starts at an offset that is a multiple of 256 bytes, with zero padding in between.
- **`<model>.json`:** the header.

  | Key | Meaning |
  |---|---|
  | `fmt` | `"xe8"` |
  | `G` | `32` |
  | `tensors` | `name → {off, nbytes, shape, dtype, kind}` |

  - `dtype` is one of `uint8`, `uint32`, `float16`, `float32`, `int8`, `bfloat16`.
  - Linear layers made by stacking several source matrices along rows also carry `rows`: the row count of each part, in order (section 6).
  - Informational keys: `rel_rmse`, `norm_stats`, `lin_bits_per_weight`, `bytes_per_token`.

The main model header also holds:
- `config`: the source model's text config. An auxiliary container such as the Qwen MTP file has no `config` and uses the main container's.

Release headers add the following keys. Readers ignore keys they do not know, so adding them changes only the `.json` file, never the `.bin` or its checksum.

| Key | Example | Meaning |
|---|---|---|
| `xefp4_version` | `"1.0"` | spec version; a reader rejects a major version it does not know |
| `model_profile` | `"qwen3_5"`, `"gemma4"` | tensor naming and stacking rules (section 6) |
| `source` | `{"model": "<hub id>", "shards": {"<file>": "<sha256>"}}` | the exact weights the file was made from |
| `norm_form` | `{"rms": "1+w", "gated": "w", "qk": "1+w", "mtp": "1+w"}` | RMSNorm convention per norm group: multiply by `w` or by `1 + w`. If the key is absent, a reader may infer the form, but inference from weight statistics is unreliable: Qwen's final norm has mean 1.14 and is still `1+w`. |
| `i8_scale` | `"fp32"` or `"fp16-rounded"` | how the INT8 row scales were chosen (section 4) |

## 6. Model profiles

Tensor names are the engine's own (`layers.{i}.proj`, …), not the source model's. Layers that read the same input are
stacked along rows into one matrix.

| Name | Qwen3.5-9B | Gemma 4 26B-A4B |
|---|---|---|
| `layers.i.proj` | linear attention: in_proj_qkv + z + b + a; full attention: q + k + v | q + k (+ v on sliding layers; global layers share k = v) |
| `layers.i.out` | out_proj / o_proj | o_proj |
| `layers.i.gu` | gate + up | dense MLP gate + up |
| `layers.i.down` | down | dense MLP down |
| `layers.i.egu` | — | experts gate_up: `[128·1408, 2816]`. Expert e starts at row e·1408; the first 704 rows are gate, the next 704 are up. |
| `layers.i.edown` | — | experts down: `[128·2816, 704]` |

Profile notes:
- **Qwen3.5-9B:**
  - Norm forms: `gated` (linear-attention output norm) is `w`; all other norms, including MTP, are `1+w`.
  - The embedding is XeFP4 (`embed.e*`); lm_head is INT8.
  - The MTP head is in a separate `q35_mtp_xe8.bin/.json`. `mtp.fc` is split by input half into `mtp.fcA` and `mtp.fcB`, each with its own row scales.
  - `vocab_order.npy` orders token ids by corpus frequency. Only the draft step uses it; it does not affect the output.
- **Gemma 4 26B-A4B:**
  - Every norm is `w`. The v norm has no weight.
  - Embedding and lm_head are tied and stored once as INT8 (`lm_head.q/.s`).
  - The input embedding is multiplied by bf16(√2816) = 53.
  - Configuration *Quality* stores the non-expert linear layers (proj, out, gu, down) as INT8 instead of XeFP4.
  - The vision encoder is an optional separate file.

## 7. Conformance

`spec/vectors/` holds the test vectors:
- `vectors.json` lists each case's shape and the SHA-256 of every array.
- One `<case>.npz` per case holds `W` (float32 input), the expected `codes`, `fac` and `rs`, and the decoded `W_hat` (float32).

Cases cover: random Gaussian and heavy-tailed rows, exact factor ties, values on every rounding boundary, negative
zero, all-zero rows, rows whose scale is an FP16 subnormal or rounds to 0, the largest representable row, and a K
whose factor bits do not fill the last 32-bit word.

`tools/check_vectors.py` checks:
- an encoder (`--encoder module:function`), byte for byte, except the codes and factors of rows whose scale is 0;
- the reference decoder, value for value.

Input range: `max |W[n,:]| / 12` must not exceed 65504 (the FP16 maximum). Larger rows cannot be encoded.
