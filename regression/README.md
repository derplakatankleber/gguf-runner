# Local Regression Harness

This directory contains a local regression runner for correctness and performance checks across GGUF models present in the repository root.

It is designed for long-running breadth tests as requested:
- `smoke` mode: a smaller, quick representative set.
- `full` mode: all discovered `*.gguf` text models (excluding `mmproj-*.gguf`).
- vision mode per model: when a matching `mmproj-*.gguf` is discovered, an additional vision scenario is run with a fixed image/prompt.

The runner logs:
- pass/fail status and reason per run
- profile timing buckets (`matmul`, `attention`, `moe`, `ffn`)
- `achieved tok/s` from CLI output
- median token/s per `(model, test_kind)` across measured runs
- optional regression comparison against a baseline file

## Files

- `run.sh`: main entrypoint
- `manifests/smoke-models.txt`: model list for smoke mode
- `manifests/skip-models.txt`: optional global skip list
- `manifests/prompts.tsv`: regex-based prompt selection
- `results/`: timestamped run artifacts

## Quick Start

Build release first:

```bash
cargo build --release
```

Run smoke:

```bash
./regression/run.sh smoke
```

Run full breadth:

```bash
./regression/run.sh full
```

Override the default fixed vision image:

```bash
./regression/run.sh full --vision-image /path/to/other-image.jpg
```

Run with custom settings:

```bash
./regression/run.sh full \
  --warmups 1 \
  --runs 2 \
  --max-tokens 64 \
  --context-size 4096 \
  --threads 8 \
  --kv-cache-mode q4
```

## Model Filtering (`--include-regex` / `--exclude-regex`)

Filtering is applied to the model filename (for example `Qwen3.5-2B-Q4_K_M.gguf`) in this order:

1. `--include-regex`: keep only matching models (if set)
2. `--exclude-regex`: remove matching models (if set)
3. `manifests/skip-models.txt`: remove explicit skip-list entries

Notes:
- Regex is evaluated with `rg` against the model name.
- Matching is case-sensitive by default.
- Use `^...$` anchors for an exact single-model match.

Run a single exact model:

```bash
./regression/run.sh full \
  --include-regex '^Qwen3\.5-35B-A3B-UD-Q4_K_M\.gguf$'
```

Run a single model quickly (no warmup, one measured run):

```bash
./regression/run.sh full \
  --include-regex '^Qwen3\.5-35B-A3B-UD-Q4_K_M\.gguf$' \
  --warmups 0 \
  --runs 1
```

Include a family but exclude one variant:

```bash
./regression/run.sh full \
  --include-regex '^Qwen3\.5-.*\.gguf$' \
  --exclude-regex '35B'
```

## Baseline Workflow

Create/update a baseline from a run:

```bash
./regression/run.sh smoke --accept-baseline regression/baselines/smoke.tsv
```

Compare future runs against baseline:

```bash
./regression/run.sh smoke --baseline regression/baselines/smoke.tsv
```

Default thresholds:
- warn on slowdown worse than `5%`
- fail on slowdown worse than `10%`

Override with:

```bash
--perf-warn-pct 3 --perf-fail-pct 7
```

## Output Artifacts

Each run creates:

- `results/<timestamp>-<mode>/meta.txt`
- `results/<timestamp>-<mode>/runs.tsv`
- `results/<timestamp>-<mode>/aggregate.tsv`
- `results/<timestamp>-<mode>/regressions.tsv` (when baseline supplied)
- `results/<timestamp>-<mode>/raw/*.log`
- `results/<timestamp>-<mode>/summary.txt`

## Notes

- The harness uses deterministic sampling flags:
  - `--temperature 0 --top-k 1 --top-p 1`
- Token/s extraction depends on `--show-tokens --show-timings`.
- For vision scenarios, the fixed default image is:
  - `regression/IMG_0138.jpg` (Image #1)
- For local CPU tuning comparisons, combine with env toggles such as:
  - `GGUF_Q8_BLOCK_SCALES=1`
  - `GGUF_AARCH64_MATMUL_PREFETCH_ROWS=0`
