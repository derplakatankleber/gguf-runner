# gguf-runner

`gguf-runner` is a pure Rust, CPU-first inference runtime for GGUF language models.

The project focuses on:
- straightforward local inference
- readable code structure
- support for multiple model families in one binary

## Quick Start

1. Build:

```bash
cargo build --release
```

2. Run with a local GGUF file:

```bash
cargo run --release -- \
  --model ./Meta-Llama-3-8B-Instruct-Q4_K_M.gguf \
  --prompt "Explain what this project does."
```

3. Show all options:

```bash
cargo run -- --help
```

## Basic Usage

Required flags:
- `--model <path>`
- `--prompt <text>`

Common optional flags:
- `--system-prompt <text>`
- `--agent` (enable tool-agent loop)
- `--tool-root <path>` (filesystem root for tools; defaults to current directory)
- `--max-tool-calls <int>` (agent loop budget)
- `--temperature <float>`
- `--top-k <int>`
- `--top-p <float>`
- `--repeat-penalty <float>`
- `--repeat-last-n <int>`
- `--max-tokens <int>`
- `--context-size <int>`
- `--threads <int>`
- `--show-tokens`
- `--show-timings`
- `--profiling`
- `--debug`
- `--url <model-url>` (lazy bootstrap/download path for missing or invalid local file)

## What Is Supported

| Model file | Family | Quantization | Benchmarked in `docs/performance.md` |
|---|---|---|---|
| `Meta-Llama-3-8B-Instruct-Q4_K_M.gguf` | Llama 3 (8B Instruct) | `Q4_K_M` | Yes |
| `gemma-3-4b-it-Q4_K_M.gguf` | Gemma 3 (4B IT) | `Q4_K_M` | Yes |
| `Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf` | Qwen2.5 Coder (14B Instruct) | `Q4_K_M` | Yes |
| `Qwen3-0.6B-Q4_K_M.gguf` | Qwen3 (0.6B) | `Q4_K_M` | Yes |
| `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` | Qwen3 (4B Instruct) | `Q4_K_M` | Yes |
| `Qwen3-30B-A3B-Instruct-2507-Q4_K_S.gguf` | Qwen3 (30B A3B Instruct) | `Q4_K_S` | Yes |
| `Qwen3-Coder-Next-Q4_K_M.gguf` | Qwen3 Coder Next | `Q4_K_M` | Yes |
| `Qwen3-235B-A22B-Instruct-2507-Q4_K_M.gguf` | Qwen3 (235B A22B Instruct) | `Q4_K_M` | No |

The table above reflects model files present in this repository root and historical benchmark entries.

This runtime currently supports multiple model families (Llama, Gemma, Qwen variants), common GGUF quantization types, and platform-specific CPU optimizations.

For detailed feature coverage and platform notes, see:
- `docs/features.md`

For historical benchmark snapshots and performance notes, see:
- `docs/performance.md`

For current module/layout reference, see:
- `docs/module-structure.md`

## GGUF Dump Example

Use the example binary to inspect GGUF metadata without running inference:

```bash
cargo run --example gguf_dump -- --model ./model.gguf --dump-kv --dump-tensors
```

If you omit `--dump-kv` and `--dump-tensors`, both are dumped by default.

## Project Scope

- CPU inference only
- GGUF model files only
- focus on transparent implementation over broad framework abstraction

## Contributing

Issues and pull requests are welcome.

Before opening a PR, run:

```bash
cargo fmt --all --check
cargo clippy --all-targets --all-features
cargo check
```
