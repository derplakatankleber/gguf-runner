# Features and Platform Support

This document summarizes the current runtime capabilities of `gguf-runner`.

## Model Family Support

Model-family handling is selected from GGUF metadata (`general.architecture`) and family-specific keys.

Supported families:
- Llama-style architectures
- Gemma (`gemma`, `gemma2`, `gemma3`)
- Qwen / Qwen2
- Qwen3.5 (`qwen35`, loaded through the Qwen3Next-style recurrent/full-attention path with dense FFN)
- Qwen3-VL (`qwen3vl`)
- Qwen3 MoE (`qwen3moe`)
- Qwen3 Next (`qwen3next`, including SSM-related tensors)

Currently unsupported:
- DeepSeek architectures (`deepseek*` GGUF metadata)

## Quantization / Tensor Type Support

Supported tensor data paths include:
- `F32`, `F16`, `BF16`
- `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`
- `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`
- `IQ4_NL`

## Runtime Features

- GGUF parsing from local files
- Linux mmap memory-advice hints for mapped model pages (best-effort)
- lazy model bootstrap/download flow with `--url`
- tokenizer initialization from GGUF vocab/metadata
- model-family-specific chat prompt rendering
- multimodal request/model capability scaffolding for Qwen3-VL and Qwen3.5:
  - startup capability probe for native image/video/audio support (token + tensor checks)
  - llama-style local `mmproj*.gguf` sidecar auto-discovery/probe (no extra CLI flag)
  - strict native-only multimodal execution (no metadata fallback path)
  - multimodal tensor-group probe during runtime load:
    - vision encoder tensor groups
    - multimodal projector tensor groups
    - audio tensor groups
    - explicit missing-group errors when backend is marked native-capable
- multimodal request scaffolding:
  - repeatable `--video <path>` input parsing/validation (`mp4`)
  - repeatable `--audio <path>` input parsing/validation (extension-agnostic)
  - structured prompt encoding for multimodal requests with placeholder span mapping (image/video/audio)
  - runtime prompt/media alignment validation before preprocessing
  - generation loop prefill hook for external embeddings (`transformer_with_embedding`) is wired for future native media injection
  - clearer media capability diagnostics when GGUF is missing native multimodal tensor groups
    - includes sidecar search results and effective support status
  - native preprocessing foundation:
    - image:
      - PNG/JPEG/WebP decode
      - deterministic resize + center crop
      - RGB -> CHW tensor conversion
      - normalization profiles (`UnitRange` and `MeanStd`)
    - video:
      - currently unavailable in no-external-dependency mode
    - audio:
      - currently unavailable in no-external-dependency mode
  - current runtime returns explicit "not implemented yet" errors for native image/video/audio embedding execution paths
- autoregressive generation loop
- quantized KV cache for attention state:
  - default `Q8` KV cache
  - automatic `Q4` KV cache fallback if `Q8` allocation fails
- optional tool-agent loop (`--agent`) with host-side file tools:
  - `read_file`
  - `list_dir`
  - `write_file`
  - `mkdir` (recursive directory creation)
  - `rmdir` (recursive directory removal)
  - `shell_list_allowed` (reports currently enabled tools + allowed shell commands)
  - `shell_exec` (restricted to operator-defined allowed commands)
  - `shell_request_allowed` (asks operator to allow a specific shell command)
- sampling modes:
  - greedy (`--temperature 0`)
  - stochastic temperature sampling
  - top-k / top-p (note: `top-p` is applied when `top-k > 0`)
  - repetition control (`--repeat-penalty`, `--repeat-last-n`)
- runtime diagnostics:
  - `--debug`
  - `--show-tokens`
  - `--show-timings`
  - `--profiling`

## CLI + Environment Configuration

User-facing CLI options are defined in `src/cli.rs`.

Agent config file (optional):
- `~/.gguf-runner/config.toml`
- `./.gguf-runner/config.toml` (loaded after home config and overrides it)

Shell allowed-commands config schema:
```toml
[tools]
read_file = true
list_dir = true
write_file = true
mkdir = true
rmdir = true
shell_list_allowed = true
shell_exec = true
shell_request_allowed = true

[shell.cmd]
rg = "Fast recursive text search."
ls = "List directory entries."
cat = "Read file content."
cwd = "Show current working directory (shell_exec built-in helper)."
```

Exposed env var:
- `GGUF_RAYON_THREADS` (same as `--threads`)
- `GGUF_ALLOW_SHELL_COMMANDS` (comma-separated allowed commands for `shell_exec`)

Hidden runtime tuning env vars (advanced use):
- `GGUF_PAR_MATMUL_MIN_ROWS`
- `GGUF_PAR_MATMUL_CHUNK_ROWS`
- `GGUF_PAR_ATTN_MIN_HEADS`
- `GGUF_PAR_QWEN3NEXT_MIN_HEADS`
- `GGUF_KV_CACHE_MODE` (`auto`, `q8`, `q4`)
- `GGUF_LAYER_DEBUG`
- `GGUF_LAYER_DEBUG_POS`
- `GGUF_AARCH64_DOTPROD_Q8` (aarch64 only)
- `GGUF_AARCH64_QK_MR4` (aarch64 only)
- `GGUF_X86_AVX2` (x86_64 only)
- `GGUF_X86_F16C` (x86_64 only)
- `GGUF_X86_QK_MR4` (x86_64 only)
- `GGUF_X86_AVXVNNI` (x86_64 only)
- `GGUF_X86_AVX512VNNI_Q8` (x86_64 only; optional Q8_0 VNNI path)

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
- native video/audio decode paths are currently unavailable in no-external-dependency mode
