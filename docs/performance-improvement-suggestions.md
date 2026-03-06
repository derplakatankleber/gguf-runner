# Performance Improvement Suggestions (M4 + Rust 1.94 AVX)

This document captures concrete, codebase-specific performance improvements identified from source inspection and local profiling.

## Baseline (Local, Apple Silicon)

Model: `Qwen3.5-35B-A3B-UD-Q4_K_M.gguf`  
Prompt: `Can you write me a programm in Rust that can convert PNG images to JPEG`  
Command profile:

```bash
target/release/gguf-runner \
  --model ./Qwen3.5-35B-A3B-UD-Q4_K_M.gguf \
  --prompt "Can you write me a programm in Rust that can convert PNG images to JPEG" \
  --max-tokens 64 \
  --context-size 4096 \
  --temperature 0 \
  --top-k 1 \
  --profiling \
  --show-timings \
  --show-tokens
```

Observed warm runs:
- ~`8.7 tok/s`
- ~`12.5s` overall runtime
- `matmul` is the dominant bucket (~94% of transformer time; overlapping counters)

## Priority Improvements

## 1) Enable fast SIMD paths by default when runtime feature detection passes ✅ Completed (2026-03-06)

### Why
- Some fast paths are compiled and runtime-available but not enabled by default.
- This can silently leave performance on the table.

### Current behavior
- `aarch64 dotprod Q8` defaults to disabled unless explicitly configured.
- `x86 AVX512 VNNI Q8` defaults to disabled unless explicitly configured.

### Suggestion
- Default both to runtime autodetect-enabled behavior.
- Keep manual override flags/env vars for forced disable in debugging/regression triage.

### Expected impact
- Medium-to-high, model-dependent (especially Q8-heavy paths).

## 2) Vectorize KV-cache attention inner loops for Q8/Q4 block-scale paths (Apple M4 focus) ✅ Completed (2026-03-06)

### Why
- Hot attention loops still use scalar element-by-element math in Q8/Q4 block-scale helpers.
- On M4, these are strong candidates for NEON acceleration.

### Suggestion
- Add NEON vectorized versions for:
  - Q8 block-scale dot/axpy
  - Q4 block-scale dot/axpy (dequant + accumulate in vector-friendly chunks)
- Dispatch similarly to other architecture-specialized kernels.

### Expected impact
- High for long-context decode and attention-heavy workloads.

## 3) Pre-quantize activation vector once per matmul call in Q8 kernels

### Why
- Several Q8 paths quantize the same `x` repeatedly per output row or row-pair.
- That duplicates work and burns cycles in a matmul-dominant runtime.

### Suggestion
- Quantize `x` once per call (or once per chunk) into scratch buffers.
- Reuse packed/quantized `x` across all row computations in that call.

### Expected impact
- High in Q8 kernels where per-row quantization currently dominates overhead.

## 4) Add AArch64 prefetch strategy for MR kernels ⚙️ Implemented (2026-03-06, tuning pending)

### Why
- x86 path has explicit row prefetch support.
- AArch64 MR chunks do not currently mirror this optimization.

### Suggestion
- Add architecture-specific prefetch helper for AArch64 (guarded and benchmarked).
- Tune prefetch distance empirically on M4.

### Expected impact
- Medium; improves memory behavior, especially with larger models/layers.

## 5) Tune parallel thresholds for non-x86 targets dynamically

### Why
- Non-x86 defaults currently use fixed constants for some parallel thresholds.
- M4 core topology and workload size can benefit from dynamic thresholds.

### Suggestion
- Use available parallelism + workload shape heuristics on aarch64 too.
- Keep CLI/env overrides for reproducibility and tuning.

### Expected impact
- Medium; reduces under/over-parallelization overhead.

## 6) Rust 1.94 AVX follow-up: expand x86 kernels beyond current AVX2 ceiling

### Why
- Q8 has AVX-VNNI/AVX512-VNNI support paths.
- Q4/Q5/Q6 x86 multi-row kernels are currently AVX2-oriented.

### Suggestion
- Prototype AVX-VNNI/AVX512 variants for Q4_K/Q5_K/Q6_K multi-row kernels.
- Add clear runtime gating and validation fallbacks (existing pattern already used).

### Expected impact
- Medium-to-high on capable x86 hosts (not M4-specific, but relevant to Rust 1.94 ask).

## 7) Reduce decoding-side sampler overhead

### Why
- Current top-k sampler repeatedly scans for minimum candidate position.
- This is O(V * k) and adds avoidable overhead at large vocab sizes.

### Suggestion
- Replace with fixed-size heap or selection approach with lower constant factors.

### Expected impact
- Low-to-medium; secondary to matmul/attention but still worthwhile.

## Recommended Execution Order

1. [x] Enable safe default dispatch for dormant fast paths (low risk, quick win).  
2. [ ] Implement and benchmark Q8 activation pre-quantization reuse.  
3. [x] Vectorize KV block-scale attention loops on aarch64 (M4-focused).  
4. [ ] Add AArch64 prefetch + threshold retuning.  
5. [ ] Expand x86 AVX/VNNI coverage under Rust 1.94.  

## Measurement Guidance

- Keep prompts/settings fixed for A/B comparisons.
- Use deterministic sampling (`temperature=0`, `top_k=1`, `top_p=1`).
- Record:
  - end-to-end tok/s
  - overall runtime
  - profile split (`matmul`, `attention`, `moe`, `ssm`)
- For each kernel change:
  - run correctness validation first
  - then benchmark at least 3 runs (cold + warm + warm)
