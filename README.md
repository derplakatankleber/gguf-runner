# gguf-runner

`gguf-runner` is a small command-line tool that lets you run AI models on your own machine.

The idea behind this project is simple: local AI should feel like a normal Unix-style tool.
You point it to a `.gguf` model file, ask a question, and stream the answer in your terminal.
No cloud API, no GPU setup maze, and no heavy platform around it.

It is built for people who want to:
- run models fully offline
- keep data local
- script prompts in shell workflows
- experiment with different model sizes on regular hardware

Under the hood, `gguf-runner` uses memory mapping (`mmap`) and CPU-only inference.
This means execution is not constrained by GPU availability or fixed GPU memory (VRAM) limits.
In theory, the upper bound shifts toward storage capacity, with the tradeoff that larger working sets become slower.
In practice, performance is often as good as your filesystem caching behavior allows, so warm-cache runs can feel much faster than cold starts.

If you are new to the project, start with the quick steps below and you should get your first response in a few minutes.

## Fastest Start (No Rust Needed)

If you just want to try `gguf-runner`, download a prebuilt binary from
[GitHub Releases](https://github.com/apimeister/gguf-runner/releases).

Release assets are packaged per platform (for example `linux-amd64`, `linux-arm64`, `macos-arm64`, `windows-amd64`).

Example (Linux/macOS):

```bash
tar -xzf gguf-runner-<tag>-linux-amd64.tar.gz
./gguf-runner --help
```

Example (Windows PowerShell):

```powershell
Expand-Archive .\gguf-runner-<tag>-windows-amd64.zip
.\gguf-runner.exe --help
```

## Getting Started: Qwen3.5 Text

Download `Qwen3.5-0.8B` from Hugging Face:

```bash
wget https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_K_M.gguf
```

Run a first "hello" prompt:

```bash
./gguf-runner \
  --model ./Qwen3.5-0.8B-Q4_K_M.gguf \
  --prompt "hello"
```

## Getting Started: Qwen3.5 Vision (JPEG)

For vision, download both the model and matching `mmproj` sidecar (same directory):

```bash
wget https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_K_M.gguf
wget https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/mmproj-Qwen3.5-2B-F16.gguf
```

Run a JPEG image prompt:

```bash
./gguf-runner \
  --model ./Qwen3.5-2B-Q4_K_M.gguf \
  --image sample-image.jpg \
  --prompt "Describe that image."
```

More model download examples:
- `docs/downloading-models.md`

## 5-Minute Quick Start

1. Build once:

```bash
cargo build --release
```

2. Run a model:

```bash
cargo run --release -- \
  --model ./Meta-Llama-3-8B-Instruct-Q4_K_M.gguf \
  --prompt "Explain what this project does."
```

3. Show all CLI options:

```bash
cargo run -- --help
```

## Working Models

Known-good status from `docs/performance.md` (text benchmarks) and local model/mmproj availability.

| Model | Text | Vision |
|---|---|---|
| `Meta-Llama-3-8B-Instruct-Q4_K_M.gguf` | ✅ | ❌ |
| `gemma-3-4b-it-Q4_K_M.gguf` | ✅ | ❌ |
| `Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf` | ✅ | ❌ |
| `Qwen3-0.6B-Q4_K_M.gguf` | ✅ | ❌ |
| `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` | ✅ | ❌ |
| `Qwen3-30B-A3B-Instruct-2507-Q4_K_S.gguf` | ✅ | ❌ |
| `Qwen3-Coder-Next-Q4_K_M.gguf` | ✅ | ❌ |
| `Qwen3-VL-2B-Instruct-Q4_K_M.gguf` | ⚪ | ✅ |
| `Qwen3-VL-30B-A3B-Instruct-Q4_K_M.gguf` | ⚪ | ✅ |
| `Qwen3.5-0.8B-Q4_K_M.gguf` | ✅ | ✅ |
| `Qwen3.5-2B-Q4_K_M.gguf` | ✅ | ✅ |
| `Qwen3.5-35B-A3B-UD-Q4_K_M.gguf` | ✅ | ✅ |

## What You Need

- A local `.gguf` model file.
- Enough RAM for the model you choose.
- Rust toolchain (only if you build from source).

## Basic Command Pattern

```bash
cargo run --release -- \
  --model ./your-model.gguf \
  --prompt "Your question"
```

Most common options (and what they do):
- `--max-tokens 256`: Maximum number of generated output tokens. Use lower values for short answers and faster test runs.
- `--context-size 4096`: Sets how much conversation/history the model can keep in context.
- `--temperature 0.7`: Controls randomness. Lower is more deterministic, higher is more creative/variable.
- `--threads 8`: Number of CPU threads to use. Usually set this near your available CPU cores.
- `--show-features`: Prints detected CPU features (compiled vs runtime) and exits.
- `--show-tokens`: Streams token-level output/diagnostics while generating.
- `--show-timings`: Prints timing breakdowns so you can inspect performance bottlenecks.
- `--profiling`: Enables deeper profiling output for performance analysis.
- `--debug`: Enables additional debug logging/details during execution.

## Vision Example (Image Input)

For vision-capable models (for example Qwen3-VL / Qwen3.5 multimodal variants):

```bash
cargo run --release -- \
  --model ./Qwen3-VL-2B-Instruct-Q4_K_M.gguf \
  --image ./regression/IMG_0138.jpg \
  --prompt "Describe this image."
```

If required multimodal tensors/components are missing, the runner fails fast with a clear error.

## Agent Mode (Optional)

```bash
cargo run --release -- \
  --model ./Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf \
  --prompt "Review this Rust function for bugs." \
  --agent
```

Useful agent options:
- `--tool-root <path>`
- `--max-tool-calls <int>`

## Regression Testing

There is a local regression harness in `regression/`:

```bash
./regression/run.sh smoke
./regression/run.sh full
```

## Project Scope

- CPU inference only
- GGUF model files only
- Focus on clear, readable implementation

## Useful Docs

- Feature coverage: `docs/features.md`
- Performance history: `docs/performance.md`
- Performance ideas/tuning notes: `docs/performance-improvement-suggestions.md`
- Module layout: `docs/module-structure.md`

## GGUF Metadata Dump (No Inference)

```bash
cargo run --example gguf_dump -- --model ./model.gguf --dump-kv --dump-tensors
```

## Contributing

Before opening a PR, run:

```bash
cargo fmt --all --check
cargo clippy --all-targets --all-features
cargo check
```
