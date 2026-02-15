# Features and Platform Support

This document summarizes the current runtime capabilities of `gguf-runner`.

## Model Family Support

Model-family handling is selected from GGUF metadata (`general.architecture`) and family-specific keys.

Supported families:
- Llama-style architectures
- Gemma (`gemma`, `gemma2`, `gemma3`)
- Qwen / Qwen2
- Qwen3 MoE (`qwen3moe`)
- Qwen3 Next (`qwen3next`, including SSM-related tensors)

## Quantization / Tensor Type Support

Supported tensor data paths include:
- `F32`, `F16`, `BF16`
- `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`
- `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`
- `IQ4_NL`

## Runtime Features

- GGUF parsing from local files
- lazy model bootstrap/download flow with `--url`
- tokenizer initialization from GGUF vocab/metadata
- model-family-specific chat prompt rendering
- autoregressive generation loop
- sampling modes:
  - greedy (`--temperature 0`)
  - stochastic temperature sampling
  - top-k / top-p (note: `top-p` is applied when `top-k > 0`)
- runtime diagnostics:
  - `--debug`
  - `--show-tokens`
  - `--show-timings`
  - `--profiling`

## CLI + Environment Configuration

User-facing CLI options are defined in `src/cli.rs`.

Exposed env var:
- `GGUF_RAYON_THREADS` (same as `--threads`)

Hidden runtime tuning env vars (advanced use):
- `GGUF_PAR_MATMUL_MIN_ROWS`
- `GGUF_PAR_MATMUL_CHUNK_ROWS`
- `GGUF_PAR_ATTN_MIN_HEADS`
- `GGUF_PAR_QWEN3NEXT_MIN_HEADS`
- `GGUF_LAYER_DEBUG`
- `GGUF_LAYER_DEBUG_POS`
- `GGUF_AARCH64_DOTPROD_Q8` (aarch64 only)
- `GGUF_AARCH64_QK_MR4` (aarch64 only)
- `GGUF_X86_AVX2` (x86_64 only)
- `GGUF_X86_F16C` (x86_64 only)
- `GGUF_X86_QK_MR4` (x86_64 only)

## Supported Platforms

Current target platforms:
- macOS (aarch64 and x86_64)
- Linux (aarch64 and x86_64)

Notes:
- runtime uses Unix memory-mapping paths for GGUF loading
- platform-specific SIMD paths are implemented for `aarch64` and `x86_64`
- non-Unix platforms (for example Windows) are not currently the primary target

## Current Boundaries

- CPU-only runtime (no GPU backend)
- GGUF-only model format
- model compatibility depends on expected tensor layout and metadata presence
