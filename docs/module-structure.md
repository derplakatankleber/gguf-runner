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
  cli.rs
  engine/
    mod.rs
    types.rs
    io/
      mod.rs
      gguf.rs
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
  - load GGUF + vendor config + tokenizer + weights
  - execute generation loop
  - print profiling/timing summaries

### `src/cli.rs`

- All clap parsing lives here.
- Public parser result type: `CliOptions`.
- Parses user-facing flags plus hidden tuning/debug options.
- Env var integration is here (via clap `env = ...`), currently `GGUF_*` variables.

### `src/engine/mod.rs`

- Aggregates engine submodules:
  - `io`, `kernels`, `profiling`, `runtime`, `switches`, `tokenizer`, `types`, `weights`.

### `src/engine/types.rs`

- Core data model and shared constants.
- Defines:
  - GGUF constants and ggml quantization constants
  - Core structs like `Config`, `GGUFFile`, `TransformerWeights`, `RunState`, `Tokenizer`, `QuantizedTensor`
  - Lazy loader implementation (`LazyModelLoader`)
  - Global lazy loader state:
    - `LAZY_MODEL_LOADER: OnceLock<Arc<LazyModelLoader>>`
  - `ensure_model_range(...)` helper used by quantized matmul paths.

### `src/engine/io/*`

- GGUF parsing and low-level read helpers.
- `io/gguf.rs`:
  - Parses GGUF metadata/tensors.
  - Maps model file.
  - Provides metadata access helpers:
    - `get_gguf_int_from_map`, `get_gguf_float_from_map`, `get_gguf_string_from_map`, `find_gguf_tensor`.

### `src/engine/tokenizer/mod.rs`

- Tokenizer initialization and encode/decode logic.
- Handles sentencepiece/tiktoken-ish paths and special token resolution.
- Exposes `init_tokenizer_from_gguf(...)`.

### `src/engine/weights.rs`

- Loads and validates model tensors from GGUF into `TransformerWeights`.
- Handles per-family tensor layout differences and optional tensors.
- Exposes `init_weights_from_gguf(...)`.

### `src/engine/kernels/*`

- Numerical and sampling kernels used by inference.
- `math.rs`: normalization, softmax, vector math, Qwen3Next SSM linear attention helpers.
- `quant.rs`: quantized dequant/dot/matmul paths, architecture-specific fast paths, MR4 validation.
- `sampling.rs`: token selection helpers (`argmax`, multinomial sample, top-k/top-p sampler).

### `src/engine/runtime/*`

- Runtime-specific execution and threading config.
- `runtime/inference.rs`:
  - `malloc_run_state(...)`
  - `transformer(...)`
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
  - Arch feature toggles (`use_x86_*`, `use_aarch64_*`)
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
  - Builds `Config` from family-specific key conventions.
  - Routes chat prompt encoding to family-specific implementation.
- `vendors/llama.rs`, `vendors/gemma.rs`, `vendors/qwen.rs`:
  - Family-specific defaults, validations, and prompt rendering.

## Runtime Data Flow

1. `main.rs` invokes `app::run()`.
2. `app::run()` parses CLI (`CliOptions`).
3. `app::run()` builds `RuntimeSwitchConfig` and calls `engine::switches::init_runtime_config(...)`.
4. GGUF parsed via `engine::io::parse_gguf_file(...)`.
5. Vendor config built with `vendors::build_config_from_gguf(...)`.
6. Tokenizer initialized (`engine::tokenizer::init_tokenizer_from_gguf(...)`).
7. Runtime overrides applied (`engine::runtime::apply_context_size_overrides(...)`).
8. Weights loaded (`engine::weights::init_weights_from_gguf(...)`).
9. Run state allocated (`engine::runtime::malloc_run_state(...)`).
10. Token loop executes forward passes (`engine::runtime::transformer(...)`) and sampling (`engine::kernels`).
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
