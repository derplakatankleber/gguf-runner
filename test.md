# performance

To test for performance and utilization, we use the following command as reference point.
```bash
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
```

## updates (2026-02-10)

### code changes

- Hoisted quantized matmul bounds checks out of per-row hot loops (`matmul_quantized` and `matmul_quantized_rows`).
- Removed per-token/per-layer heap allocations in MoE routing (`select_topk_softmax`) by reusing `RunState` scratch buffers.

### current runtime and memory consumption

Command used:
```bash
/usr/bin/time -l ./target/release/llama3pure -model Qwen3-Coder-Next-Q4_K_M.gguf -prompt "Can you write me a programm in Rust that can convert PNG images to JPEG" -max_tokens 50000 -context_size 250000 >/tmp/qwen3next_current_bench.out
```

Results:
```text
      329.40 real       863.52 user       503.91 sys
         17901813760  maximum resident set size
                   0  average shared memory size
                   0  average unshared data size
                   0  average unshared stack size
             3359508  page reclaims
             5177113  page faults
                   0  swaps
                   0  block input operations
                   0  block output operations
                   0  messages sent
                   0  messages received
                   0  signals received
                 170  voluntary context switches
            42442706  involuntary context switches
       9948690440809  instructions retired
       3778221944402  cycles elapsed
          1458540400  peak memory footprint
```

## updates (2026-02-10, deep optimization pass)

### code changes

- Qwen3Next recurrent state update (`qwen3next_linear_attention_autoregressive`) now uses:
  - dedicated per-head scratch buffers (`ssm_kv_mem`, `ssm_delta`)
  - optional parallel execution across heads (`LLAMA3PURE_PAR_QWEN3NEXT_MIN_HEADS`, default `8`)
  - reduced hot-loop overhead by moving repeated work into `qwen3next_state_head_step`
- Full-attn packed Q/gate path now avoids storing gate values into a separate `q_gate` buffer:
  - removed `q_gate` from `RunState`
  - reads gate values directly from packed `s.hb` during gating
  - optional parallel Q deinterleave copy by head
- Parallel tuning improvements:
  - raised defaults to `matmul_min_rows=256`, `attn_min_heads=8`
  - added runtime tuning via env vars:
    - `LLAMA3PURE_PAR_MATMUL_MIN_ROWS`
    - `LLAMA3PURE_PAR_ATTN_MIN_HEADS`
    - `LLAMA3PURE_PAR_QWEN3NEXT_MIN_HEADS`
  - added `-threads` CLI option and `LLAMA3PURE_RAYON_THREADS` env var
- ARM (Apple Silicon) SIMD optimization:
  - unrolled aarch64 NEON kernels for `dot_f32_simd_ptr`, `axpy_simd_ptr`, and `scale_simd_inplace` (16-float stride)

### current runtime and memory consumption

Command used:
```bash
/usr/bin/time -l ./target/release/llama3pure -model Qwen3-Coder-Next-Q4_K_M.gguf -prompt "Can you write me a programm in Rust that can convert PNG images to JPEG" -max_tokens 50000 -context_size 250000 >/tmp/qwen3_newopt_full.out
```

Results:
```text
      327.78 real       884.33 user       499.50 sys
         15154610176  maximum resident set size
                   0  average shared memory size
                   0  average unshared data size
                   0  average unshared stack size
             4299598  page reclaims
             6955697  page faults
                   0  swaps
                   0  block input operations
                   0  block output operations
                   0  messages sent
                   0  messages received
                   0  signals received
                  65  voluntary context switches
            43254543  involuntary context switches
      11023421691139  instructions retired
       4057781656515  cycles elapsed
          1463750536  peak memory footprint
```

## updates (2026-02-10, arm kernels + profiling)

### code changes

- Added ARM-specific fused QK kernels for `Q4_K`, `Q5_K`, and `Q6_K` on aarch64 (Apple Silicon path), reducing temporary buffer traffic in quantized dot products.
- Added additional aarch64 NEON unrolling for core vector primitives (`dot`, `axpy`, `scale`).
- Added `-profiling` flag (and disabled-by-default profiling counters) for:
  - `transformer_total`
  - `matmul`
  - `ssm`
  - `attention`
  - `moe`
  - `ffn`
- Added runtime parallel tuning controls:
  - `LLAMA3PURE_PAR_MATMUL_MIN_ROWS`
  - `LLAMA3PURE_PAR_ATTN_MIN_HEADS`
  - `LLAMA3PURE_PAR_QWEN3NEXT_MIN_HEADS`
  - `-threads` / `LLAMA3PURE_RAYON_THREADS`

### current runtime and memory consumption

Command used:
```bash
/usr/bin/time -l ./target/release/llama3pure -model Qwen3-Coder-Next-Q4_K_M.gguf -prompt "Can you write me a programm in Rust that can convert PNG images to JPEG" -max_tokens 50000 -context_size 250000 >/tmp/qwen3_armprof_opt.out
```

Results:
```text
      505.45 real      1881.64 user       565.64 sys
         14985953280  maximum resident set size
                   0  average shared memory size
                   0  average unshared data size
                   0  average unshared stack size
             4542311  page reclaims
             9868880  page faults
                   0  swaps
                   0  block input operations
                   0  block output operations
                   0  messages sent
                   0  messages received
                   0  signals received
                 199  voluntary context switches
            54602037  involuntary context switches
      22747896767141  instructions retired
       7993103889940  cycles elapsed
          1471205256  peak memory footprint
```

## latest changes (2026-02-10)

- Added fused aarch64 quantized dot paths for `Q4_K`, `Q5_K`, and `Q6_K` to reduce temporary-buffer traffic on Apple Silicon.
- Added/extended aarch64 NEON unrolling for `dot`, `axpy`, and `scale` primitives.
- Added `-profiling` mode with timing counters for:
  - `transformer_total`
  - `matmul`
  - `ssm`
  - `attention`
  - `moe`
  - `ffn`
- Kept profiling disabled by default; it only prints when `-profiling` is passed.
- Added/kept runtime tuning knobs for parallelism:
  - `-threads` and `LLAMA3PURE_RAYON_THREADS`
  - `LLAMA3PURE_PAR_MATMUL_MIN_ROWS`
  - `LLAMA3PURE_PAR_ATTN_MIN_HEADS`
  - `LLAMA3PURE_PAR_QWEN3NEXT_MIN_HEADS`

Example profiling command:
```bash
./target/release/llama3pure -model Qwen3-Coder-Next-Q4_K_M.gguf -prompt "hi" -max_tokens 1 -context_size 64 -profiling
```

## updates (2026-02-10, matmul optimization points 1/2/3)

### code changes

- Implemented aarch64 multi-row (`MR=4`) quantized matmul kernels for:
  - `Q4_K`
  - `Q5_K`
  - `Q6_K`
- Integrated these kernels into matmul row execution (`matmul_quantized` and `matmul_quantized_rows`) so they run directly for the above types.
- Removed function-pointer dispatch from matmul hot loops:
  - switched to per-type direct match dispatch, enabling better inlining and specialization.
- Added an ARM-specific, opt-in Q8 path (`LLAMA3PURE_AARCH64_DOTPROD_Q8=1`) that uses aarch64 int8 accumulation intrinsics for faster int8 dot-style accumulation.
  - This path is disabled by default.

### current runtime and memory/profile snapshot

Default short profiling run:
```bash
/usr/bin/time -p ./target/release/llama3pure -model Qwen3-Coder-Next-Q4_K_M.gguf -prompt "Can you write me a programm in Rust that can convert PNG images to JPEG" -max_tokens 120 -context_size 250000 -profiling
```

Results:
```text
[PROFILE] forward_passes=120
[PROFILE] transformer_total=60900.578 ms (507.505 ms/pass)
[PROFILE] matmul=59851.859 ms (98.3%)
[PROFILE] ssm=9896.553 ms (16.3%)
[PROFILE] attention=2161.048 ms (3.5%)
[PROFILE] moe=47421.085 ms (77.9%)
real 62.28
user 173.34
sys 81.31
```

Tuned short profiling run (`-threads 8` and coarser thresholds):
```bash
LLAMA3PURE_PAR_MATMUL_MIN_ROWS=512 LLAMA3PURE_PAR_ATTN_MIN_HEADS=16 LLAMA3PURE_PAR_QWEN3NEXT_MIN_HEADS=16 /usr/bin/time -p ./target/release/llama3pure -model Qwen3-Coder-Next-Q4_K_M.gguf -prompt "Can you write me a programm in Rust that can convert PNG images to JPEG" -max_tokens 120 -context_size 250000 -profiling -threads 8
```

Results:
```text
[PROFILE] forward_passes=120
[PROFILE] transformer_total=59018.606 ms (491.822 ms/pass)
[PROFILE] matmul=58015.625 ms (98.3%)
[PROFILE] ssm=10182.781 ms (17.3%)
[PROFILE] attention=2322.239 ms (3.9%)
[PROFILE] moe=45007.687 ms (76.3%)
real 60.13
user 164.24
sys 60.68
```

## updates (2026-02-10, full run after matmul 1/2/3)

### command used

```bash
/usr/bin/time -l ./target/release/llama3pure -model Qwen3-Coder-Next-Q4_K_M.gguf -prompt "Can you write me a programm in Rust that can convert PNG images to JPEG" -max_tokens 50000 -context_size 250000 >/tmp/qwen3_full_after_matmul123.out
```

### results

```text
      427.90 real      1384.12 user       639.31 sys
         14742552576  maximum resident set size
                   0  average shared memory size
                   0  average unshared data size
                   0  average unshared stack size
             1351306  page reclaims
             7824908  page faults
                   0  swaps
                   0  block input operations
                   0  block output operations
                   0  messages sent
                   0  messages received
                   0  signals received
                  85  voluntary context switches
            59082946  involuntary context switches
      17782191475445  instructions retired
       5635860167134  cycles elapsed
          1464127368  peak memory footprint
```

# intel ultra 125h

real	7m30.149s
user	47m30.305s
sys	63m43.536s

# apple m4
427.90 real      1384.12 user       639.31 sys

## updates (2026-02-10, x86 optimization guide)

### x86 code paths now available

- Added runtime-dispatched x86 SIMD paths.
- `LLAMA3PURE_X86_AVX2=1` enables AVX2+FMA kernels for core `dot/axpy/scale` and BF16 prefix dot.
- `LLAMA3PURE_X86_F16C=1` enables F16C-based F16 prefix dot.
- `LLAMA3PURE_X86_QK_MR4=1` enables x86 `MR=4` matmul kernels for `Q4_K`, `Q5_K`, `Q6_K`.
- Added chunked parallel row scheduling for matmul.
- `LLAMA3PURE_PAR_MATMUL_CHUNK_ROWS` (new)
- `LLAMA3PURE_PAR_MATMUL_MIN_ROWS` (kept)

### recommended baseline command (x86)

```bash
/usr/bin/time -l ./target/release/llama3pure -model Qwen3-Coder-Next-Q4_K_M.gguf -prompt "Can you write me a programm in Rust that can convert PNG images to JPEG" -max_tokens 50000 -context_size 250000 -temperature 0 -top_k 1 -top_p 1 -show-tokens
```

### intel ultra preset (starting point)

```bash
LLAMA3PURE_X86_AVX2=1 \
LLAMA3PURE_X86_F16C=1 \
LLAMA3PURE_X86_QK_MR4=1 \
LLAMA3PURE_PAR_MATMUL_MIN_ROWS=512 \
LLAMA3PURE_PAR_MATMUL_CHUNK_ROWS=96 \
LLAMA3PURE_PAR_ATTN_MIN_HEADS=16 \
LLAMA3PURE_PAR_QWEN3NEXT_MIN_HEADS=16 \
/usr/bin/time -l ./target/release/llama3pure -model Qwen3-Coder-Next-Q4_K_M.gguf -prompt "Can you write me a programm in Rust that can convert PNG images to JPEG" -max_tokens 50000 -context_size 250000 -temperature 0 -top_k 1 -top_p 1 -threads 8 -show-tokens
```

### amd zen 5 preset (starting point)

```bash
LLAMA3PURE_X86_AVX2=1 \
LLAMA3PURE_X86_F16C=1 \
LLAMA3PURE_X86_QK_MR4=1 \
LLAMA3PURE_PAR_MATMUL_MIN_ROWS=640 \
LLAMA3PURE_PAR_MATMUL_CHUNK_ROWS=128 \
LLAMA3PURE_PAR_ATTN_MIN_HEADS=16 \
LLAMA3PURE_PAR_QWEN3NEXT_MIN_HEADS=16 \
/usr/bin/time -l ./target/release/llama3pure -model Qwen3-Coder-Next-Q4_K_M.gguf -prompt "Can you write me a programm in Rust that can convert PNG images to JPEG" -max_tokens 50000 -context_size 250000 -temperature 0 -top_k 1 -top_p 1 -threads 12 -show-tokens
```
./target/release/llama3pure -model Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf -prompt "Can you write me a programm in Rust that can convert PNG images to JPEG" -max_tokens 50000 -context_size 250000 -temperature 0 -top_k 1 -top_p 1 -threads 12 -show-tokens

### quick ablation checks

```bash
# disable MR4 only
LLAMA3PURE_X86_QK_MR4=0 ./target/release/llama3pure -model Qwen3-Coder-Next-Q4_K_M.gguf -prompt "test" -max_tokens 512 -context_size 8192 -temperature 0 -top_k 1 -top_p 1 -show-tokens

# disable AVX2/FMA path only
LLAMA3PURE_X86_AVX2=0 ./target/release/llama3pure -model Qwen3-Coder-Next-Q4_K_M.gguf -prompt "test" -max_tokens 512 -context_size 8192 -temperature 0 -top_k 1 -top_p 1 -show-tokens

# disable F16C path only
LLAMA3PURE_X86_F16C=0 ./target/release/llama3pure -model Qwen3-Coder-Next-Q4_K_M.gguf -prompt "test" -max_tokens 512 -context_size 8192 -temperature 0 -top_k 1 -top_p 1 -show-tokens
```

### notes

- Keep benchmark inputs fixed: same model file, prompt, sampling params, `max_tokens`, and `context_size`.
- Use `-temperature 0 -top_k 1 -top_p 1` for deterministic decoding and higher comparability.
- Compare both `real` and `achieved tok/s`; watch `involuntary context switches` for oversubscription.

# beelink - intel n150

time ./target/release/gguf-runner -model Qwen3-4B-Instruct-2507-Q4_K_M.gguf -url https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf -prompt "Can you write me a programm in Rust that can convert PNG images to JPEG" -show-tokens

1.5token/s

# framwork - intel i5-1340P

3.4token/s

# macbook air - m4

5 token/s

# macbook air

./target/release/gguf-runner -model Qwen3-4B-Instruct-2507-Q4_K_M.gguf -prompt "Can you write me a programm in Rust that can convert PNG images to JPEG" -show-tokens