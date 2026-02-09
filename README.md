# llama3pure

original repo:
https://github.com/lrusso/llama3pure?tab=readme-ov-file


A pure Rust, CPU-first GGUF inference runtime focused on readable implementation and local experimentation.

## Project Goal

This project is aiming to be a transparent, hackable inference engine for modern open-weight models in GGUF format, without depending on external runtimes like `llama.cpp`.

The practical goal is:
- load GGUF models directly
- run autoregressive text generation locally
- support multiple model families in one codebase
- keep the implementation understandable enough to modify and extend

## Current Repository Content

Current top-level content in this workspace:
- `src/main.rs` - main runtime (GGUF parser, tokenizer, quantized matmul, transformer forward pass, sampling, CLI)
- `src/model/config.rs` - model-family detection and config extraction from GGUF metadata
- `src/model/chat.rs` - chat prompt templates per model family
- `Cargo.toml` / `Cargo.lock` - Rust project config (`rayon` dependency)
- `target/` - local build artifacts
- multiple local `.gguf` model files (very large, from ~2.3 GB to ~45 GB each)

Local model files currently present:
- `gemma-3-4b-it-Q4_K_M.gguf` (~2.3 GB)
- `Meta-Llama-3-8B-Instruct-Q4_K_M.gguf` (~4.6 GB)
- `phi-4-Q4_K_M.gguf` (~8.3 GB)
- `Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf` (~8.4 GB)
- `qwen-image-2512-Q4_K_M.gguf` (~12 GB)
- `Qwen3-30B-A3B-Instruct-2507-Q4_K_S.gguf` (~16 GB)
- `Qwen3-235B-A22B-Thinking-2507-Q4_K_M-00003-of-00003.gguf` (~39 GB)
- `Qwen3-Coder-Next-Q4_K_M.gguf` (~45 GB)

## What the Runtime Supports Today

### Model families
Detected from GGUF metadata (`general.architecture` + family keys):
- Llama-style architectures (default/fallback)
- Gemma family (`gemma`, `gemma2`, `gemma3` key layouts)
- Qwen/Qwen2
- Qwen3 MoE (`qwen3moe`)
- Qwen3 Next (`qwen3next`, including SSM-specific tensors)

### Quantization / tensor formats used in matmul
- `F32`, `F16`, `BF16`
- `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`
- `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`
- `IQ4_NL`

### Runtime behavior
- memory-maps GGUF files on Unix-like systems
- builds tokenizer from GGUF vocab/merges/scores
- applies family-specific chat templates (Llama 3 / Gemma / Qwen ChatML variants)
- supports greedy (`-temperature 0`) and stochastic sampling
- supports `top_k` and optional `top_p` (note: `top_p` is only used when `top_k > 0`)
- supports debug logs with throughput output (`-debug`)

## Build

```bash
cargo build --release
```

## Run

Minimal usage:

```bash
cargo run --release -- \
  -model ./Meta-Llama-3-8B-Instruct-Q4_K_M.gguf \
  -prompt "Explain in one paragraph what this project does."
```

With common options:

```bash
cargo run --release -- \
  -model ./Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf \
  -system_prompt "You are a concise coding assistant." \
  -prompt "Write a Rust function that validates IPv4 addresses." \
  -temperature 0.7 \
  -top_k 40 \
  -top_p 0.9 \
  -max_tokens 256 \
  -context_size 4096 \
  -debug
```

## CLI Arguments

Required:
- `-model <path>`: GGUF model file path
- `-prompt <text>`: user prompt

Optional:
- `-system_prompt <text>` (default: `You are a helpful assistant.`)
- `-temperature <float>` (default: `0.9`, use `0.0` for greedy)
- `-top_k <int>` (default: `0`, disabled when zero)
- `-top_p <float>` (default: `1.0`, valid range `(0, 1]`, only used with `-top_k > 0`)
- `-max_tokens <int>` (default: `256`)
- `-context_size <int>` (default: model max; qwen3 models may clamp to 8192 unless explicitly overridden)
- `-debug` (enables verbose load + performance logs)

## Notes and Limitations

- This is currently a Rust-only runtime in this workspace.
- No test suite is currently checked in under `src/`.
- Model compatibility depends on GGUF metadata and expected tensor layout for the detected family.
- Memory usage is substantial for large checkpoints.
- For Qwen3 Next, specific SSM metadata/tensors are required and validated at load time.

## Development Direction

Near-term focus appears to be:
- keep extending model-family coverage in a single pure Rust engine
- keep improving correctness for architecture-specific tensor mappings
- maintain performant CPU inference while preserving code readability

# sample runs


## c-version
time ./llama3pure -model Meta-Llama-3-8B-Instruct-Q4_K_M.gguf -prompt "Tell me in 1 line what is Microsoft."

Microsoft is a multinational technology corporation that develops, manufactures, licenses, and supports a wide range of software products, services, and devices, including the Windows operating system, Office software suite, and Xbox gaming console.


160.03s user 0.76s system 99% cpu 2:41.17 total

## rust version

time ./target/release/llama3pure -model Meta-Llama-3-8B-Instruct-Q4_K_M.gguf -prompt "Tell me in 1 line what is Microsoft."
Microsoft is a multinational technology corporation that develops, manufactures, licenses, and supports a wide range of software products, services, and devices, including the Windows operating system, Microsoft Office, and Xbox gaming consoles.
./target/release/llama3pure -model Meta-Llama-3-8B-Instruct-Q4_K_M.gguf    285.89s user 1.66s system 99% cpu 4:48.39 total

## SIMD
time ./target/release/llama3pure -model Meta-Llama-3-8B-Instruct-Q4_K_M.gguf -prompt "Tell me in 1 line what is Microsoft."
Microsoft is a multinational technology company that develops, manufactures, licenses, and supports a wide range of software products, services, and devices, including the Windows operating system, Office software suite, and Xbox gaming console.
./target/release/llama3pure -model Meta-Llama-3-8B-Instruct-Q4_K_M.gguf    120.79s user 0.93s system 99% cpu 2:02.36 total

## rayon

jens@Mac llama3pure % time ./target/release/llama3pure -model Meta-Llama-3-8B-Instruct-Q4_K_M.gguf -prompt "Tell me in 1 line what is Microsoft."
Microsoft is a multinational technology corporation that develops, manufactures, licenses, and supports a wide range of software products, services, and devices, including the Windows operating system, Microsoft Office productivity suite, and Xbox gaming console.
./target/release/llama3pure -model Meta-Llama-3-8B-Instruct-Q4_K_M.gguf    120.97s user 8.55s system 832% cpu 15.553 total


 RUSTFLAGS="-C target-cpu=native" cargo build --release
jens@Mac llama3pure % time ./target/release/llama3pure -model Meta-Llama-3-8B-Instruct-Q4_K_M.gguf -prompt "Tell me in 1 line what is Microsoft."
Microsoft is a multinational technology corporation that develops, manufactures, licenses, and supports a wide range of software products, services, and devices, including the Windows operating system, Microsoft Office application software, and Xbox gaming console.
./target/release/llama3pure -model Meta-Llama-3-8B-Instruct-Q4_K_M.gguf    121.48s user 6.85s system 869% cpu 14.758 total

## full Qwen-Coder 80B

time ./target/release/llama3pure -model Qwen3-Coder-Next-Q4_K_M.gguf -prompt "Can you write me a programm in Rust that can convert PNG images to JPEG" -max_tokens 50000 -context_size 250000

./target/release/llama3pure -model Qwen3-Coder-Next-Q4_K_M.gguf -prompt       1055.66s user 452.01s system 511% cpu 4:54.93 total

# working models

gemma-3-4b-it-Q4_K_M.gguf
Qwen3-Coder-Next-Q4_K_M.gguf
Qwen3-30B-A3B-Instruct-2507-Q4_K_S.gguf
Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf
Meta-Llama-3-8B-Instruct-Q4_K_M.gguf


# llama3pure vs llama-cli

/usr/bin/time -l llama-cli -m Qwen3-Coder-Next-Q4_K_M.gguf --n-gpu-layers 0 -p "Can you write me a programm in Rust that can convert PNG images to JPEG"
      840.84 real       723.82 user       271.35 sys
         23993057280  maximum resident set size
                   0  average shared memory size
                   0  average unshared data size
                   0  average unshared stack size
            24205272  page reclaims
            10304639  page faults
                   0  swaps
                   0  block input operations
                   0  block output operations
                   0  messages sent
                   0  messages received
                   2  signals received
             3739287  voluntary context switches
            13029133  involuntary context switches
      13004308126129  instructions retired
       3311218222110  cycles elapsed
         43459789696  peak memory footprint

/usr/bin/time -l ./target/release/llama3pure -model Qwen3-Coder-Next-Q4_K_M.gguf -prompt "Can you write me a programm in Rust that can convert PNG images to JPEG" -max_tokens 50000 -context_size 250000
      402.09 real      1471.08 user       615.37 sys
         24622071808  maximum resident set size
                   0  average shared memory size
                   0  average unshared data size
                   0  average unshared stack size
             3249301  page reclaims
             5537088  page faults
                   0  swaps
                   0  block input operations
                   0  block output operations
                   0  messages sent
                   0  messages received
                   0  signals received
                 111  voluntary context switches
            41269987  involuntary context switches
      17121227687878  instructions retired
       5829526672922  cycles elapsed
          1485049760  peak memory footprint
