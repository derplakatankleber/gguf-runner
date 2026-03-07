#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

MODE="smoke"
BINARY="${ROOT_DIR}/target/release/gguf-runner"
RESULTS_ROOT="${SCRIPT_DIR}/results"
PROMPTS_FILE="${SCRIPT_DIR}/manifests/prompts.tsv"
SMOKE_MODELS_FILE="${SCRIPT_DIR}/manifests/smoke-models.txt"
SKIP_MODELS_FILE="${SCRIPT_DIR}/manifests/skip-models.txt"
BASELINE_FILE=""
ACCEPT_BASELINE=""
WARMUPS=1
RUNS=2
MAX_TOKENS=64
CONTEXT_SIZE=4096
THREADS=""
KV_CACHE_MODE=""
PERF_WARN_PCT=5
PERF_FAIL_PCT=10
INCLUDE_REGEX=""
EXCLUDE_REGEX=""
VISION_IMAGE="${SCRIPT_DIR}/IMG_0138.jpg"
VISION_PROMPT="Describe the content of this image."

usage() {
    cat <<'USAGE'
Usage:
  ./regression/run.sh [smoke|full] [options]

Options:
  --binary PATH                 Runner binary (default: target/release/gguf-runner)
  --results-root DIR            Output root (default: regression/results)
  --prompts-file FILE           Regex prompt map (default: regression/manifests/prompts.tsv)
  --smoke-models FILE           Smoke model list (default: regression/manifests/smoke-models.txt)
  --skip-models FILE            Skip list (default: regression/manifests/skip-models.txt)
  --warmups N                   Warmup runs per model (default: 1)
  --runs N                      Measured runs per model (default: 2)
  --max-tokens N                Max tokens (default: 64)
  --context-size N              Context size (default: 4096)
  --threads N                   Pass --threads N
  --kv-cache-mode MODE          Pass --kv-cache-mode (auto|q8|q4)
  --vision-image PATH           Image path used for vision regression runs
  --vision-prompt TEXT          Prompt used for vision regression runs
  --include-regex REGEX         Include only model names matching regex
  --exclude-regex REGEX         Exclude model names matching regex
  --baseline FILE               Compare against prior aggregate.tsv
  --accept-baseline FILE        Copy current aggregate.tsv to FILE
  --perf-warn-pct N             Warn threshold for slowdown (default: 5)
  --perf-fail-pct N             Fail threshold for slowdown (default: 10)
  -h, --help                    Show help

Examples:
  ./regression/run.sh smoke
  ./regression/run.sh full --baseline regression/baselines/full.tsv
  ./regression/run.sh smoke --runs 1 --warmups 0 --include-regex 'Qwen3-0.6B'
  ./regression/run.sh smoke --vision-image docs/IMG_0192.jpg
USAGE
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

is_nonnegative_int() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        smoke|full)
            MODE="$1"
            shift
            ;;
        --binary)
            BINARY="$2"
            shift 2
            ;;
        --results-root)
            RESULTS_ROOT="$2"
            shift 2
            ;;
        --prompts-file)
            PROMPTS_FILE="$2"
            shift 2
            ;;
        --smoke-models)
            SMOKE_MODELS_FILE="$2"
            shift 2
            ;;
        --skip-models)
            SKIP_MODELS_FILE="$2"
            shift 2
            ;;
        --warmups)
            WARMUPS="$2"
            shift 2
            ;;
        --runs)
            RUNS="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --context-size)
            CONTEXT_SIZE="$2"
            shift 2
            ;;
        --threads)
            THREADS="$2"
            shift 2
            ;;
        --kv-cache-mode)
            KV_CACHE_MODE="$2"
            shift 2
            ;;
        --vision-image)
            VISION_IMAGE="$2"
            shift 2
            ;;
        --vision-prompt)
            VISION_PROMPT="$2"
            shift 2
            ;;
        --include-regex)
            INCLUDE_REGEX="$2"
            shift 2
            ;;
        --exclude-regex)
            EXCLUDE_REGEX="$2"
            shift 2
            ;;
        --baseline)
            BASELINE_FILE="$2"
            shift 2
            ;;
        --accept-baseline)
            ACCEPT_BASELINE="$2"
            shift 2
            ;;
        --perf-warn-pct)
            PERF_WARN_PCT="$2"
            shift 2
            ;;
        --perf-fail-pct)
            PERF_FAIL_PCT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
done

[[ -x "$BINARY" ]] || die "binary not executable: $BINARY"
[[ -f "$PROMPTS_FILE" ]] || die "missing prompts file: $PROMPTS_FILE"
[[ -f "$SMOKE_MODELS_FILE" ]] || die "missing smoke model list: $SMOKE_MODELS_FILE"
[[ -f "$SKIP_MODELS_FILE" ]] || die "missing skip list: $SKIP_MODELS_FILE"
is_nonnegative_int "$WARMUPS" || die "--warmups must be nonnegative int"
is_nonnegative_int "$RUNS" || die "--runs must be nonnegative int"
is_nonnegative_int "$MAX_TOKENS" || die "--max-tokens must be nonnegative int"
is_nonnegative_int "$CONTEXT_SIZE" || die "--context-size must be nonnegative int"
is_nonnegative_int "$PERF_WARN_PCT" || die "--perf-warn-pct must be nonnegative int"
is_nonnegative_int "$PERF_FAIL_PCT" || die "--perf-fail-pct must be nonnegative int"

if [[ -n "$THREADS" ]]; then
    is_nonnegative_int "$THREADS" || die "--threads must be nonnegative int"
    [[ "$THREADS" -gt 0 ]] || die "--threads must be >= 1"
fi

if [[ -n "$KV_CACHE_MODE" ]]; then
    case "$KV_CACHE_MODE" in
        auto|q8|q4) ;;
        *) die "--kv-cache-mode must be one of auto|q8|q4" ;;
    esac
fi

if [[ -n "$BASELINE_FILE" && ! -f "$BASELINE_FILE" ]]; then
    die "baseline file not found: $BASELINE_FILE"
fi

if [[ ! -f "$VISION_IMAGE" ]]; then
    die "vision image not found: $VISION_IMAGE"
fi

timestamp="$(date +%Y%m%d-%H%M%S)"
run_dir="${RESULTS_ROOT}/${timestamp}-${MODE}"
raw_dir="${run_dir}/raw"
mkdir -p "$raw_dir"

runs_tsv="${run_dir}/runs.tsv"
aggregate_tsv="${run_dir}/aggregate.tsv"
regressions_tsv="${run_dir}/regressions.tsv"
summary_txt="${run_dir}/summary.txt"
meta_txt="${run_dir}/meta.txt"

cat >"$meta_txt" <<META
timestamp=${timestamp}
mode=${MODE}
root_dir=${ROOT_DIR}
binary=${BINARY}
warmups=${WARMUPS}
runs=${RUNS}
max_tokens=${MAX_TOKENS}
context_size=${CONTEXT_SIZE}
threads=${THREADS}
kv_cache_mode=${KV_CACHE_MODE}
perf_warn_pct=${PERF_WARN_PCT}
perf_fail_pct=${PERF_FAIL_PCT}
vision_image=${VISION_IMAGE}
vision_prompt=${VISION_PROMPT}
baseline_file=${BASELINE_FILE}
git_sha=$(git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null || echo unknown)
rustc=$(rustc --version 2>/dev/null || echo unknown)
host=$(hostname)
uname=$(uname -a)
META

printf "timestamp\tmode\tmodel\tparam_size\ttest_kind\trun_kind\trun_idx\texit_code\tstatus\treason\tachieved_toks\toverall_runtime_s\tforward_passes\ttransformer_total_ms\tmatmul_ms\tattention_ms\tmoe_ms\tffn_ms\tlog_path\n" >"$runs_tsv"
printf "timestamp\tmode\tmodel\tparam_size\ttest_kind\tmeasured_runs\tpass_runs\tfail_runs\tmedian_toks\tmedian_runtime_s\tcorrectness_status\n" >"$aggregate_tsv"

collect_models() {
    if [[ "$MODE" == "smoke" ]]; then
        while IFS= read -r model; do
            [[ -z "$model" || "$model" =~ ^# ]] && continue
            [[ -f "${ROOT_DIR}/${model}" ]] || continue
            [[ "$model" == mmproj-* ]] && continue
            echo "$model"
        done <"$SMOKE_MODELS_FILE"
    else
        find "$ROOT_DIR" -maxdepth 1 -type f -name "*.gguf" -print \
            | sed "s|^${ROOT_DIR}/||" \
            | sort \
            | awk '$0 !~ /^mmproj-/ { print }'
    fi
}

collect_mmproj_models() {
    find "$ROOT_DIR" -maxdepth 1 -type f -name "mmproj-*.gguf" -print \
        | sed "s|^${ROOT_DIR}/||" \
        | sort
}

model_match_key() {
    local name="$1"
    local base="${name##*/}"
    base="${base%.gguf}"
    base="${base#mmproj-}"
    base="$(printf '%s' "$base" | sed -E 's/-(Q[0-9A-Za-z_]+|F16|BF16|FP16)$//')"
    printf '%s' "$base" | tr '[:upper:]' '[:lower:]' | tr -cd '[:alnum:]'
}

find_mmproj_for_model() {
    local model="$1"
    local model_key
    local best=""
    local best_score=0
    model_key="$(model_match_key "$model")"
    for mmproj in "${mmproj_models[@]}"; do
        local mmproj_key
        mmproj_key="$(model_match_key "$mmproj")"
        if [[ "$mmproj_key" == "$model_key"* || "$model_key" == "$mmproj_key"* ]]; then
            local score=${#mmproj_key}
            if (( ${#model_key} < score )); then
                score=${#model_key}
            fi
            if (( score > best_score )); then
                best="$mmproj"
                best_score=$score
            fi
        fi
    done
    printf '%s' "$best"
}

is_skipped_model() {
    local model="$1"
    awk -v m="$model" '
        /^[[:space:]]*#/ { next }
        /^[[:space:]]*$/ { next }
        { gsub(/^[[:space:]]+|[[:space:]]+$/, "", $0); if ($0 == m) { found=1; exit } }
        END { exit(found ? 0 : 1) }
    ' "$SKIP_MODELS_FILE"
}

pick_prompt() {
    local model="$1"
    while IFS=$'\t' read -r pattern prompt; do
        [[ -z "$pattern" || "$pattern" =~ ^# ]] && continue
        if printf '%s\n' "$model" | rg -q -e "$pattern"; then
            printf '%s' "$prompt"
            return 0
        fi
    done <"$PROMPTS_FILE"
    printf 'Can you write me a programm in Rust that can convert PNG images to JPEG'
}

sanitize_name() {
    printf '%s' "$1" | tr '/ :' '___'
}

extract_param_size() {
    local model="$1"
    local lower
    lower="$(printf '%s' "$model" | tr '[:upper:]' '[:lower:]')"
    if [[ "$lower" =~ ([0-9]+([.][0-9]+)?b(-a[0-9]+([.][0-9]+)?b)?) ]]; then
        printf '%s' "${BASH_REMATCH[1]}" | tr '[:lower:]' '[:upper:]'
        return 0
    fi
    if [[ "$lower" =~ ([0-9]+([.][0-9]+)?m) ]]; then
        printf '%s' "${BASH_REMATCH[1]}" | tr '[:lower:]' '[:upper:]'
        return 0
    fi
    printf '%s' "unknown"
}

extract_metric() {
    local pattern="$1"
    local file="$2"
    (rg -o -r '$1' "$pattern" "$file" || true) | tail -n 1
}

median_from_list() {
    local values_file="$1"
    if [[ ! -s "$values_file" ]]; then
        echo ""
        return 0
    fi
    local sorted_file="${values_file}.sorted"
    sort -n "$values_file" >"$sorted_file"
    local n
    n="$(wc -l <"$sorted_file" | tr -d '[:space:]')"
    if (( n % 2 == 1 )); then
        local idx=$((n / 2 + 1))
        sed -n "${idx}p" "$sorted_file"
    else
        local idx1=$((n / 2))
        local idx2=$((idx1 + 1))
        awk -v a="$idx1" -v b="$idx2" 'NR==a{v1=$1} NR==b{v2=$1} END{printf "%.6f\n", (v1+v2)/2.0}' "$sorted_file"
    fi
}

run_one() {
    local model="$1"
    local param_size="$2"
    local test_kind="$3"
    local prompt="$4"
    local run_kind="$5"
    local run_idx="$6"
    local vision_image_path="${7:-}"
    local model_abs="${ROOT_DIR}/${model}"
    local log_path="${raw_dir}/$(sanitize_name "$model").${test_kind}.${run_kind}.${run_idx}.log"

    local -a cmd
    cmd=(
        "$BINARY"
        --model "$model_abs"
        --prompt "$prompt"
        --max-tokens "$MAX_TOKENS"
        --context-size "$CONTEXT_SIZE"
        --temperature 0
        --top-k 1
        --top-p 1
        --repeat-penalty 1.0
        --repeat-last-n 0
        --show-tokens
        --show-timings
        --profiling
        --think hidden
    )

    if [[ -n "$THREADS" ]]; then
        cmd+=(--threads "$THREADS")
    fi
    if [[ -n "$KV_CACHE_MODE" ]]; then
        cmd+=(--kv-cache-mode "$KV_CACHE_MODE")
    fi
    if [[ "$test_kind" == "vision" ]]; then
        cmd+=(--image "$vision_image_path")
    fi

    set +e
    "${cmd[@]}" >"$log_path" 2>&1
    local exit_code=$?
    set -e

    local status="pass"
    local reason="ok"
    local toks
    local runtime
    local forward_passes
    local transformer_total_ms
    local matmul_ms
    local attention_ms
    local moe_ms
    local ffn_ms

    toks="$(extract_metric '^achieved tok/s: ([0-9.][0-9.]*)$' "$log_path")"
    runtime="$(extract_metric '^overall runtime: ([0-9.][0-9.]*)s$' "$log_path")"
    forward_passes="$(extract_metric '^\[PROFILE\] forward_passes=([0-9][0-9]*)$' "$log_path")"
    transformer_total_ms="$(extract_metric '^\[PROFILE\] transformer_total=([0-9.][0-9.]*) ms.*$' "$log_path")"
    matmul_ms="$(extract_metric '^\[PROFILE\] matmul=([0-9.][0-9.]*) ms.*$' "$log_path")"
    attention_ms="$(extract_metric '^\[PROFILE\] attention=([0-9.][0-9.]*) ms.*$' "$log_path")"
    moe_ms="$(extract_metric '^\[PROFILE\] moe=([0-9.][0-9.]*) ms.*$' "$log_path")"
    ffn_ms="$(extract_metric '^\[PROFILE\] ffn=([0-9.][0-9.]*) ms.*$' "$log_path")"

    if [[ "$exit_code" -ne 0 ]]; then
        status="fail"
        reason="exit_${exit_code}"
    elif rg -qi '(thread .+ panicked|panic|segmentation fault|fatal error)' "$log_path"; then
        status="fail"
        reason="panic_or_crash"
    elif rg -qi '\b(nan|inf)\b' "$log_path"; then
        status="fail"
        reason="nan_or_inf"
    elif [[ -z "$toks" ]]; then
        status="fail"
        reason="missing_toks"
    elif [[ -z "$runtime" ]]; then
        status="fail"
        reason="missing_runtime"
    fi

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$timestamp" "$MODE" "$model" "$param_size" "$test_kind" "$run_kind" "$run_idx" "$exit_code" \
        "$status" "$reason" "${toks:-}" "${runtime:-}" "${forward_passes:-}" \
        "${transformer_total_ms:-}" "${matmul_ms:-}" "${attention_ms:-}" "${moe_ms:-}" \
        "${ffn_ms:-}" "$log_path" >>"$runs_tsv"
}

all_models=()
while IFS= read -r model; do
    all_models+=("$model")
done < <(collect_models)
if [[ "${#all_models[@]}" -eq 0 ]]; then
    die "no models discovered for mode=${MODE}"
fi

mmproj_models=()
while IFS= read -r mmproj; do
    mmproj_models+=("$mmproj")
done < <(collect_mmproj_models)

models=()
for model in "${all_models[@]}"; do
    if [[ -n "$INCLUDE_REGEX" ]] && ! printf '%s\n' "$model" | rg -q -e "$INCLUDE_REGEX"; then
        continue
    fi
    if [[ -n "$EXCLUDE_REGEX" ]] && printf '%s\n' "$model" | rg -q -e "$EXCLUDE_REGEX"; then
        continue
    fi
    if is_skipped_model "$model"; then
        continue
    fi
    models+=("$model")
done

if [[ "${#models[@]}" -eq 0 ]]; then
    die "all models filtered out"
fi

echo "Run directory: $run_dir"
echo "Models selected (${#models[@]}):"
printf '  - %s\n' "${models[@]}"
echo "mmproj files discovered: ${#mmproj_models[@]}"

for model in "${models[@]}"; do
    param_size="$(extract_param_size "$model")"
    text_prompt="$(pick_prompt "$model")"
    mmproj="$(find_mmproj_for_model "$model")"
    echo "==> ${model}"
    echo "    text scenario"
    for ((i = 1; i <= WARMUPS; i++)); do
        echo "      warmup ${i}/${WARMUPS}"
        run_one "$model" "$param_size" "text" "$text_prompt" "warmup" "$i"
    done
    for ((i = 1; i <= RUNS; i++)); do
        echo "      run ${i}/${RUNS}"
        run_one "$model" "$param_size" "text" "$text_prompt" "measured" "$i"
    done

    if [[ -n "$VISION_IMAGE" && -n "$mmproj" ]]; then
        echo "    vision scenario (mmproj=${mmproj})"
        for ((i = 1; i <= WARMUPS; i++)); do
            echo "      warmup ${i}/${WARMUPS}"
            run_one "$model" "$param_size" "vision" "$VISION_PROMPT" "warmup" "$i" "$VISION_IMAGE"
        done
        for ((i = 1; i <= RUNS; i++)); do
            echo "      run ${i}/${RUNS}"
            run_one "$model" "$param_size" "vision" "$VISION_PROMPT" "measured" "$i" "$VISION_IMAGE"
        done
    fi
done

correctness_failures=0
measured_keys_file="${run_dir}/tmp.measured.keys.tsv"
awk -F '\t' 'NR>1 && $6=="measured" { print $3 "\t" $4 "\t" $5 }' "$runs_tsv" | sort -u >"$measured_keys_file"
tests_total="$(awk 'END{print NR+0}' "$measured_keys_file")"

while IFS=$'\t' read -r model param_size test_kind; do
    [[ -z "$model" ]] && continue
    measured_file="${run_dir}/tmp.$(sanitize_name "$model").${test_kind}.measured.tsv"
    awk -F '\t' -v m="$model" -v t="$test_kind" '$3==m && $5==t && $6=="measured"' "$runs_tsv" >"$measured_file"

    measured_runs="$(awk 'END { print NR+0 }' "$measured_file")"
    pass_runs="$(awk -F '\t' '$9=="pass" { c++ } END { print c+0 }' "$measured_file")"
    fail_runs="$((measured_runs - pass_runs))"
    correctness_status="pass"
    if [[ "$fail_runs" -gt 0 ]]; then
        correctness_status="fail"
        correctness_failures=$((correctness_failures + 1))
    fi

    toks_vals="${run_dir}/tmp.$(sanitize_name "$model").${test_kind}.toks.txt"
    runtime_vals="${run_dir}/tmp.$(sanitize_name "$model").${test_kind}.runtime.txt"
    awk -F '\t' '$9=="pass" && $11 != "" { print $11 }' "$measured_file" >"$toks_vals"
    awk -F '\t' '$9=="pass" && $12 != "" { print $12 }' "$measured_file" >"$runtime_vals"

    median_toks="$(median_from_list "$toks_vals")"
    median_runtime="$(median_from_list "$runtime_vals")"

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$timestamp" "$MODE" "$model" "$param_size" "$test_kind" "$measured_runs" \
        "$pass_runs" "$fail_runs" "${median_toks:-}" "${median_runtime:-}" \
        "$correctness_status" >>"$aggregate_tsv"
done <"$measured_keys_file"

perf_warn_count=0
perf_fail_count=0
if [[ -n "$BASELINE_FILE" ]]; then
    printf "model\ttest_kind\tbaseline_toks\tcurrent_toks\tdelta_pct\tseverity\n" >"$regressions_tsv"
    while IFS=$'\t' read -r _ts _mode model _param_size test_kind measured_runs pass_runs fail_runs median_toks _median_runtime correctness_status; do
        [[ "$model" == "model" ]] && continue
        [[ "$correctness_status" == "pass" ]] || continue
        [[ -n "$median_toks" ]] || continue
        baseline_toks="$(awk -F '\t' -v m="$model" -v t="$test_kind" '$3==m && $5==t { print $9; exit }' "$BASELINE_FILE")"
        [[ -n "$baseline_toks" ]] || continue
        delta_pct="$(awk -v c="$median_toks" -v b="$baseline_toks" 'BEGIN { if (b == 0) print ""; else printf "%.3f", ((c-b)/b)*100.0 }')"
        [[ -n "$delta_pct" ]] || continue

        severity="ok"
        if awk -v d="$delta_pct" -v t="-$PERF_FAIL_PCT" 'BEGIN { exit !(d <= t) }'; then
            severity="fail"
            perf_fail_count=$((perf_fail_count + 1))
        elif awk -v d="$delta_pct" -v t="-$PERF_WARN_PCT" 'BEGIN { exit !(d <= t) }'; then
            severity="warn"
            perf_warn_count=$((perf_warn_count + 1))
        fi
        printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$model" "$test_kind" "$baseline_toks" "$median_toks" "$delta_pct" "$severity" >>"$regressions_tsv"
    done <"$aggregate_tsv"
fi

{
    echo "run_dir=${run_dir}"
    echo "mode=${MODE}"
    echo "models=${#models[@]}"
    echo "test_cases=${tests_total}"
    echo "correctness_failures=${correctness_failures}"
    echo "perf_warn_count=${perf_warn_count}"
    echo "perf_fail_count=${perf_fail_count}"
    echo "aggregate_tsv=${aggregate_tsv}"
    if [[ -n "$BASELINE_FILE" ]]; then
        echo "baseline_file=${BASELINE_FILE}"
        echo "regressions_tsv=${regressions_tsv}"
    fi
} >"$summary_txt"

echo "Summary:"
cat "$summary_txt"

echo
echo "Model Results:"
printf "%-45s | %-7s | %-10s | %-7s | %-10s\n" "model" "test" "params" "success" "token/s"
printf "%-45s-+-%-7s-+-%-10s-+-%-7s-+-%-10s\n" "---------------------------------------------" "-------" "----------" "-------" "----------"
awk -F '\t' '
    NR == 1 { next }
    {
        model = $3
        test = $5
        params = $4
        success = ($11 == "pass") ? "yes" : "no"
        toks = ($9 == "" ? "n/a" : $9)
        printf "%-45s | %-7s | %-10s | %-7s | %-10s\n", model, test, params, success, toks
    }
' "$aggregate_tsv"

if [[ -n "$ACCEPT_BASELINE" ]]; then
    mkdir -p "$(dirname -- "$ACCEPT_BASELINE")"
    cp "$aggregate_tsv" "$ACCEPT_BASELINE"
    echo "Baseline updated: $ACCEPT_BASELINE"
fi

exit_code=0
if [[ "$correctness_failures" -gt 0 ]]; then
    exit_code=1
fi
if [[ "$perf_fail_count" -gt 0 ]]; then
    exit_code=1
fi
exit "$exit_code"
