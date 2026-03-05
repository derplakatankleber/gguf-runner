# Module Structure Reference

This project is currently a binary crate (`src/main.rs`) with internal modules (`mod ...`), not a separate `lib.rs`.

## Maintenance Rule

- When architecture-related code changes, update this file in the same change.
- Keep this file as the current-state snapshot.

## Top-Level Layout

```text
src/
  main.rs
  app/
    mod.rs
    generation.rs
    agent.rs
    tools.rs
  cli.rs
  engine/
    mod.rs
    types.rs
    io/
      mod.rs
      gguf.rs
    multimodal/
      mod.rs
      injection.rs
      qwen3vl.rs
    vision/
      mod.rs
      preprocess.rs
    tokenizer/
      mod.rs
    weights.rs
    kernels/
      mod.rs
      math.rs
      quant.rs
      sampling.rs
    runtime/
      mod.rs
      inference.rs
      parallel.rs
    switches.rs
    profiling.rs
  vendors/
    mod.rs
    llama.rs
    gemma.rs
    qwen.rs
```

## Module Responsibilities

### `src/main.rs`

- Binary entrypoint (`fn main()`) and crate root wiring.
- Delegates runtime orchestration to `app::run()`.

### `src/app/mod.rs`

- Application orchestration entrypoint (`run()`).
- Executes end-to-end run pipeline:
  - parse CLI options
  - map CLI tuning flags into `engine::switches::RuntimeSwitchConfig`
  - initialize runtime switches via `engine::switches::init_runtime_config(...)`
  - initialize profiling
  - load runtime/model context via `app::generation`
  - validate non-empty media file paths for standard generation mode:
    - `--image` (`png/jpg/jpeg/webp`)
    - `--video` (`mp4`)
    - `--audio` (extension-agnostic validation in current scaffold)
  - route to standard generation mode or tool-agent mode
  - print profiling/timing summaries

### `src/app/generation.rs`

- Model/runtime bootstrap for inference:
  - GGUF parse/load
  - llama-style local `mmproj*.gguf` sidecar discovery/probe for multimodal models (no extra CLI switch)
  - sidecar probe enforces checkpoint variant token matching (for example `2b`, `35b`, `a3b`) to prevent silent cross-size pairing
  - vendor config + tokenizer + weights initialization
  - multimodal weight-group probe/initialization for multimodal backends
  - context/thread overrides
- Token generation loop implementation.
- Exposes reusable generation APIs:
  - `generate_text(...)` for text-only prompts
  - `generate_text_with_images(...)` (image path routes through structured request execution)
    - qwen35 image route appends a non-hallucination guard instruction for unreadable text regions
  - `generate_request(...)` for structured multimodal requests (`GenerationRequest`) with:
    - structured prompt encoding via `vendors::encode_generation_request(...)`
    - placeholder-span/media alignment checks for image/video/audio inputs
    - fail-fast native capability checks when required multimodal tensors/components are missing
      - capability-probe details (`image/video/audio`) are included in diagnostics, including sidecar search/probe status
    - native image/video/audio preprocessing execution prior to embedding path
      - images: decode + resize/crop + normalize + CHW tensor conversion
      - for external mmproj vision backends, image resize target is taken from encoder metadata (for example `clip.vision.image_size`) instead of a fixed 224x224 fallback
      - qwen35 backend uses fit-within preprocessing (aspect-ratio preserving, patch-aligned) at a balanced intermediate scale to retain full-frame text regions while limiting visual-token overload
      - videos: currently unavailable in no-external-dependency mode
      - audio: currently unavailable in no-external-dependency mode
    - native image embedding execution via in-engine multimodal backend (`qwen3vl`/`qwen35` mmproj sidecar path)
    - explicit current-state errors for unimplemented native video/audio embedding execution
  - shared decode core `generate_from_prefill(...)` for text + multimodal routes (supports per-position embedding overrides during prefill)

### `src/app/agent.rs`

- Tool-agent orchestration loop for multi-step runs.
- Builds turn prompt transcript, requests one model response per turn, and parses JSON outputs.
- Builds system prompt with tool catalog metadata (description / when-to-use) and optional allowed-shell-command descriptions supplied by `cli`.
- Executes tool calls through `app::tools::ToolExecutor` and appends tool results back into transcript.
- Terminates on `final` response or configured tool-call limit.

### `src/app/tools.rs`

- Host-side tool execution for agent mode.
- Provides safe file tools:
  - `read_file`
  - `write_file`
  - `list_dir`
  - `mkdir` (recursive create)
  - `rmdir` (recursive delete; never removes `tool_root`)
  - `shell_list_allowed`
- Provides restricted external command tools:
  - `shell_exec` (only for allowed command names)
  - `shell_request_allowed` (structured request to operator)
- Enforces tool root path constraints and per-call payload limits.

### `src/cli.rs`

- All clap parsing lives here.
- Public parser result type: `CliOptions`.
- Parses user-facing flags plus hidden tuning/debug options.
- Env var integration is here (via clap `env = ...`), currently `GGUF_*` variables.
- Loads optional layered TOML config for agent shell allowed commands:
  - `~/.gguf-runner/config.toml`
  - `./.gguf-runner/config.toml` (overrides home config)
- Supports single-source command metadata in `[shell.cmd]` (key=command, value=description); `[shell.md]` and older formats remain accepted for compatibility.
- Supports optional tool config in `[tools]`:
  - internal tool toggles (all default enabled)
- Includes agent-related switches:
  - `--agent`
  - `--tool-root`
  - `--allow-shell-command`
  - `--max-tool-calls`
- Includes multimodal switch:
  - repeatable `--image <path>` for image inputs (standard generation mode)
  - repeatable `--video <path>` for video inputs (standard generation mode)
  - repeatable `--audio <path>` for audio inputs (standard generation mode)

### `src/engine/mod.rs`

- Aggregates engine submodules:
  - `io`, `kernels`, `profiling`, `runtime`, `switches`, `tokenizer`, `types`, `vision`, `weights`.

### `src/engine/types.rs`

- Core data model and shared constants.
- Defines:
  - GGUF constants and ggml quantization constants
  - Core structs like `Config`, `GGUFFile`, `TransformerWeights`, `RunState`, `Tokenizer`, `QuantizedTensor`
  - Qwen3-VL input embedding shaping metadata on `Config`:
    - `input_embedding_dim` (language dim plus optional deepstack lanes)
    - `n_deepstack_layers`
  - Multimodal request domain types used by app/runtime boundary:
    - `GenerationRequest`
    - `ContentPart`
    - `MediaRef`
    - `EncodedPrompt`
    - `PlaceholderSpan`
  - Model multimodal capability metadata:
    - `MultimodalBackend`
    - `ModelCapabilities`
  - Extended model identity flags:
    - `Config::is_qwen35`
    - `Config::rope_sections` for Qwen3.5 M-RoPE section metadata
  - Lazy loader implementation (`LazyModelLoader`)
  - Unix mmap wrapper (`MappedFile`) including Linux memory advice hints for model mappings
  - Global lazy loader state:
    - `LAZY_MODEL_LOADER: OnceLock<Arc<LazyModelLoader>>`
  - `ensure_model_range(...)` helper used by quantized matmul paths.
  - GGUF metadata value variants include integer arrays (`I64Array`) for keys such as `*.rope.dimension_sections`.

### `src/engine/io/*`

- GGUF parsing and low-level read helpers.
- `io/gguf.rs`:
  - Parses GGUF metadata/tensors.
  - Maps model file.
  - Provides metadata access helpers:
    - `get_gguf_int_from_map`, `get_gguf_float_from_map`, `get_gguf_i64_array_from_map`, `get_gguf_string_from_map`, `find_gguf_tensor`.

### `src/engine/vision/*`

- Shared multimodal preprocessing utilities.
- `vision/preprocess.rs` currently provides deterministic preprocessing:
  - images:
    - decode (`png`/`jpeg`/`webp` via `image` crate)
    - resize + center crop to profile target size
    - CHW float tensor conversion
    - configurable normalization profile (`UnitRange` / `MeanStd`)
  - videos:
    - currently unavailable in no-external-dependency mode (native decode path removed)
  - audio:
    - currently unavailable in no-external-dependency mode (native decode path removed)

### `src/engine/multimodal/*`

- Native multimodal embedding and prompt-injection subsystem.
- `multimodal/injection.rs`:
  - expands image placeholder spans into token-aligned embedding injection maps
  - builds expanded prefill token stream for variable-length image embedding sequences
- `multimodal/qwen3vl.rs`:
  - Qwen3-VL CLIP/mmproj image encoder path (`qwen3vl_merger`)
  - loads mmproj tensors, runs patch embedding + vision transformer + projector in Rust
  - reads `clip.vision.image_mean/std` normalization metadata from mmproj GGUF when available
  - loads and applies optional `v.deepstack.*` layer branches, fused into projected media embeddings
  - uses SIMD dot products + rayon head-parallel attention for the vision self-attention hot path
  - emits language-space image token embeddings for prompt injection
- `multimodal/mod.rs`:
  - backend construction (`build_vision_encoder_from_mmproj`)
  - validates sidecar compatibility (`validate_mmproj_for_backend`) and enables external `mmproj` construction for `qwen3vl` and `qwen35` backends
  - enforces strict family checks to prevent cross-pairing (`qwen3vl` sidecar on `qwen35` text model and vice versa)
  - encoder abstraction (`VisionEncoder`)

### `src/engine/tokenizer/mod.rs`

- Tokenizer initialization and encode/decode logic.
- Handles sentencepiece/tiktoken-ish paths and special token resolution.
- Exposes `init_tokenizer_from_gguf(...)`.

### `src/engine/weights.rs`

- Loads and validates model tensors from GGUF into `TransformerWeights`.
- Handles per-family tensor layout differences and optional tensors.
- Handles Qwen3.5 split-SMM gate tensor compatibility (`ssm_alpha.weight` / `ssm_beta.weight`) alongside fused `ssm_ba.weight`.
- Provides multimodal tensor-group initialization/probe (`init_multimodal_weights_from_gguf`) with explicit missing-group diagnostics for native multimodal backends.
- Exposes `init_weights_from_gguf(...)`.

### `src/engine/kernels/*`

- Numerical and sampling kernels used by inference.
- `math.rs`: normalization, softmax, vector math, Qwen3Next SSM linear attention helpers.
- `quant.rs`: quantized dequant/dot/matmul paths, architecture-specific fast paths (including x86 AVX-VNNI/AVX512-VNNI Q8 paths), MR4 validation.
- `sampling.rs`: token selection helpers (`argmax`, multinomial sample, top-k/top-p sampler).

### `src/engine/runtime/*`

- Runtime-specific execution and threading config.
- `runtime/inference.rs`:
  - `malloc_run_state(...)`
  - `transformer(...)`
  - `transformer_with_embedding(...)` (prefill hook for external embedding vectors)
  - accepts multimodal prefill vectors at either `dim` or `input_embedding_dim`
  - applies per-layer deepstack residual injection for Qwen3-VL-style expanded embeddings
  - applies llama-style Qwen3.5 M-RoPE cache reconstruction for text decode (`[t,h,w,e]=[pos,pos,pos,0]`)
  - quantized KV cache storage for attention state:
    - default Q8 cache
    - automatic Q4 fallback when Q8 allocation fails
- `runtime/parallel.rs`:
  - `configure_rayon_threads(...)`
- `runtime/mod.rs`:
  - Re-exports runtime helpers.
  - `apply_context_size_overrides(...)`.

### `src/engine/switches.rs`

- Runtime tuning and feature switches.
- Keeps `OnceLock` / atomic-backed switch state as system-of-record.
- Includes:
  - `RuntimeSwitchConfig` (engine-owned overrides struct)
  - Parallel thresholds (`par_matmul_min_rows`, `par_matmul_chunk_rows`, `par_attn_min_heads`, `par_qwen3next_min_heads`)
  - KV cache selection switch (`kv_cache_mode`: `auto` / `q8` / `q4`)
  - Arch feature toggles (`use_x86_*`, `use_aarch64_*`, including x86 AVX2/F16C/QK-MR4/AVX-VNNI/AVX512VNNI-Q8 switches)
  - Layer debug toggles
  - MR4 status atomics
  - `init_runtime_config(&RuntimeSwitchConfig)`.

### `src/engine/profiling.rs`

- Profiling counters and helper functions.
- Contains all profiling atomics and report formatting:
  - `set_profiling_enabled`, `prof_start`, `prof_end`, `profiling_reset`, `record_forward_pass`, `print_profile_report`.

### `src/vendors/*`

- Vendor/model-family specific config parsing and prompt templating.
- `vendors/mod.rs`:
  - Detects model family from GGUF metadata.
  - Rejects unsupported DeepSeek architectures (`deepseek*`) with a clear config error.
  - Builds `Config` from family-specific key conventions.
  - Detects `qwen35` explicitly and maps it onto the Qwen3Next-style runtime path.
  - Probes multimodal capability from tokenizer special tokens + GGUF tensor prefixes for `qwen3vl` and `qwen35`.
  - Routes both simple chat prompt encoding and structured `GenerationRequest` encoding to family-specific implementation.
- `vendors/llama.rs`, `vendors/gemma.rs`, `vendors/qwen.rs`:
  - Family-specific defaults, validations, and prompt rendering (including Qwen MoE routing defaults/scaling and Qwen image/video/audio placeholder-span mapping).

## Runtime Data Flow

1. `main.rs` invokes `app::run()`.
2. `app::run()` parses CLI (`CliOptions`).
3. `app::run()` builds `RuntimeSwitchConfig` and calls `engine::switches::init_runtime_config(...)`.
4. GGUF parsed via `engine::io::parse_gguf_file(...)`.
5. Vendor config built with `vendors::build_config_from_gguf(...)`.
6. Tokenizer initialized (`engine::tokenizer::init_tokenizer_from_gguf(...)`).
7. Runtime overrides applied (`engine::runtime::apply_context_size_overrides(...)`).
8. Weights loaded (`engine::weights::init_weights_from_gguf(...)`).
9. Standard mode:
  - CLI media inputs are normalized into `engine::types::GenerationRequest` with `ContentPart` items.
  - prompt encoded via `vendors::encode_generation_request(...)`, including placeholder spans for image/video/audio on Qwen multimodal paths.
  - runtime validates prompt/media alignment before starting preprocessing.
  - if native multimodal tensors are unavailable, runtime fails with a qualified native-capability error that includes architecture/token/tensor probe details.
  - token loop executes forward passes (`engine::runtime::transformer(...)`) + sampling (`engine::kernels`); native media embedding injection remains in progress.
10. Agent mode:
  - tool transcript prompt encoded per turn
  - model emits JSON `tool_call` / `final`
  - host executes tool call (`app::tools`) and loops until final response or limit.
  - `shell_exec` calls are restricted to CLI/env-provided allowed commands; model can request missing commands with `shell_request_allowed`.
11. Profiling/timings printed from `engine::profiling` + `app::run()`.

## Placement Rules For Future Changes

- New CLI flags or env vars: `src/cli.rs`.
- End-to-end run orchestration: `src/app/mod.rs`.
- New runtime tuning switches or arch toggles: `src/engine/switches.rs`.
- New profiling counters/reporting: `src/engine/profiling.rs`.
- New math/quant/sampling primitive: `src/engine/kernels/*`.
- New model-family metadata or prompt format: `src/vendors/<family>.rs` + dispatch in `src/vendors/mod.rs`.
- New GGUF parsing logic: `src/engine/io/gguf.rs`.
- New tensor loading logic: `src/engine/weights.rs`.
- Keep `src/main.rs` focused on entrypoint + crate wiring.

## Validation Workflow

- Run these checks for refactor validation:
  - `cargo check`
  - `rg -n "^use crate::\\*;" src/engine -g'*.rs'`
  - `rg -n "^use crate::engine::[A-Za-z0-9_:]+::\\*;" src/engine -g'*.rs'`
  - `rg -n "crate::cli::" src/engine -g'*.rs'`
- Expected result for all `rg` checks above: no matches.

## Known Coupling (Current State)

- `engine` no longer depends on `cli` types directly; runtime switch wiring happens in `app`.
- `main.rs` no longer re-exports `engine::types::*`; `vendors` and `engine` import from `engine::*` modules directly.
- `src/engine/*` no longer uses wildcard imports for internal crate modules.
- Remaining wildcard imports in `engine` are architecture-intrinsics (`std::arch::*`) for SIMD code paths.
