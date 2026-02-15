# Performance Notes (Historical)

This document summarizes benchmark snippets that previously lived in `README.md` and `test.md`.

These numbers are useful as historical reference points, not as strict apples-to-apples benchmarks across machines.

## Benchmark Table Template

Use this section as a starting point for structured performance collection.

### Host Profiles

| host_id | os | cpu | cores | memory gb | notes |
|---|---|---|---|---:|---|
| mac-m4-32g | macOS 15.3 | Apple M4 | 10 | 32 | laptop |
| lnx-n150-12g | Gentoo Linux | Intel N150 | 4 | 12 | Beelink ME mini |
| lnx-13600k-8g | Ubuntu 24.04 | Intel i5-13600K | 20 | 8 | |
| lnx-125h-32g | Gentoo Linux | Intel Ultra 125h | 18 | 32 | Minisforum M1 Pro-125H |
| lnx-9700-64g | Ubuntu 24.04 | AMD Ryzen 7 PRO 8700GE | 16 | 64 | Hetzner AX42 |

### Prompts

#### png_to_jpeg_v1
  "Can you write me a programm in Rust that can convert PNG images to JPEG"

```bash
gguf-runner --model Qwen3-4B-Instruct-2507-Q4_K_M.gguf --prompt "Can you write me a programm in Rust that can convert PNG images to JPEG" --temperature=0 --show-tokens --show-timings
```

### Benchmark Runs

| date | host_id | model | prompts | tokens/sec | runtime sec | notes |
|---|---|---|---|---:|---:|---|
| 2026-02-15 | mac-m4-32g | gemma-3-4b-it-Q4_K_M.gguf | png_to_jpeg_v1 | 4.753 | 206.488 | |
| 2026-02-15 | mac-m4-32g | Meta-Llama-3-8B-Instruct-Q4_K_M.gguf | png_to_jpeg_v1 | 2.770 | 135.304 | |
| 2026-02-15 | mac-m4-32g | Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf | png_to_jpeg_v1 | 1.251 | 421.389 | |
| 2026-02-15 | lnx-n150-12g | Qwen3-0.6B-Q4_K_M.gguf | png_to_jpeg_v1 | 6.236 | 179.751 | |
| 2026-02-15 | mac-m4-32g | Qwen3-0.6B-Q4_K_M.gguf | png_to_jpeg_v1 | 24.575 | 46.232 | |
| 2026-02-15 | lnx-9700-64g | Qwen3-0.6B-Q4_K_M.gguf | png_to_jpeg_v1 | 27.721 | 41.037 | |
| 2026-02-15 | lnx-n150-12g | Qwen3-4B-Instruct-2507-Q4_K_M.gguf | png_to_jpeg_v1 | 1.607 | 528.286 | |
| 2026-02-15 | lnx-13600k-8g | Qwen3-4B-Instruct-2507-Q4_K_M.gguf | png_to_jpeg_v1 | 3.813 | 222.872 | |
| 2026-02-15 | mac-m4-32g | Qwen3-4B-Instruct-2507-Q4_K_M.gguf | png_to_jpeg_v1 | 4.881 | 175.791 | |
| 2026-02-15 | lnx-125h-32g | Qwen3-4B-Instruct-2507-Q4_K_M.gguf | png_to_jpeg_v1 | 4.982 | 170.833 | |
| 2026-02-15 | lnx-9700-64g | Qwen3-4B-Instruct-2507-Q4_K_M.gguf | png_to_jpeg_v1 |  |  | |
| 2026-02-15 | mac-m4-32g | Qwen3-Coder-Next-Q4_K_M.gguf | png_to_jpeg_v1 | 1.613 | 530.794 | |

## Benchmark Caveats

- Results come from different dates, machines, and code revisions.
- Some runs include profiling or debug behavior that affects runtime.
- Some commands use the old binary name (`llama3pure`), others use `gguf-runner`.

## Legacy README Snapshots

### Llama 3 8B prompt run progression

Prompt: `Tell me in 1 line what is Microsoft.`

| Variant | Reported wall time |
|---|---:|
| C version (`llama3pure`) | 2:41.17 |
| Rust (early baseline) | 4:48.39 |
| Rust + SIMD | 2:02.36 |
| Rust + Rayon | 15.553s |
| Rust + Rayon + `RUSTFLAGS="-C target-cpu=native"` | 14.758s |

### Qwen3-Coder-Next full run (legacy README)

Command pattern:

```bash
./target/release/llama3pure -model Qwen3-Coder-Next-Q4_K_M.gguf \
  -prompt "Can you write me a programm in Rust that can convert PNG images to JPEG" \
  -max_tokens 50000 -context_size 250000
```

Reported:
- `real`: `4:54.93`
- `user`: `1055.66s`
- `sys`: `452.01s`

### Legacy comparison: `llama-cli` vs `llama3pure`

Same Qwen3-Coder-Next prompt workload (`/usr/bin/time -l`):

| Tool | real | user | sys | max RSS |
|---|---:|---:|---:|---:|
| `llama-cli` | 840.84s | 723.82s | 271.35s | 23,993,057,280 |
| `llama3pure` | 402.09s | 1471.08s | 615.37s | 24,622,071,808 |

## `test.md` Optimization Timeline (2026-02-10)

Workload used repeatedly:

```bash
/usr/bin/time -l ./target/release/llama3pure -model Qwen3-Coder-Next-Q4_K_M.gguf \
  -prompt "Can you write me a programm in Rust that can convert PNG images to JPEG" \
  -max_tokens 50000 -context_size 250000
```

| Label in notes | real | user | sys | max RSS |
|---|---:|---:|---:|---:|
| Baseline reference | 402.09s | 1471.08s | 615.37s | 24,622,071,808 |
| `updates (2026-02-10)` | 329.40s | 863.52s | 503.91s | 17,901,813,760 |
| `deep optimization pass` | 327.78s | 884.33s | 499.50s | 15,154,610,176 |
| `arm kernels + profiling` | 505.45s | 1881.64s | 565.64s | 14,985,953,280 |
| `full run after matmul 1/2/3` | 427.90s | 1384.12s | 639.31s | 14,742,552,576 |

Notes:
- The profiling-enabled run is expected to be slower.
- Memory footprint trends downward across most optimization passes.

## `test.md` Short Profiling Snapshot (120 tokens)

Default short profiling run:
- `real`: `62.28s`
- `[PROFILE] transformer_total`: `60900.578 ms` (`507.505 ms/pass`)
- `[PROFILE] matmul`: `59851.859 ms` (`98.3%`)

Tuned short profiling run (`-threads 8` + threshold env vars):
- `real`: `60.13s`
- `[PROFILE] transformer_total`: `59018.606 ms` (`491.822 ms/pass`)
- `[PROFILE] matmul`: `58015.625 ms` (`98.3%`)

## Device Throughput Snippets (from `test.md`)

Reported rough throughput values for a Qwen3-4B workload:

| Device | Throughput |
|---|---:|
| Beelink Intel N150 | ~1.5 tok/s |
| Framework Intel i5-1340P | ~3.4 tok/s |
| MacBook Air M4 | ~5 tok/s |

## Reproducibility Guidance

From the original notes:
- keep model, prompt, `max_tokens`, and `context_size` fixed
- use deterministic decoding for comparisons:
  - `-temperature 0 -top_k 1 -top_p 1`
- compare both wall time and token throughput

