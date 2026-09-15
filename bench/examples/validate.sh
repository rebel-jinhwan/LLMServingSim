#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PYTHON:-python3}"

OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-../validation}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
PREFIX="${PREFIX:-}"
TITLE_PREFIX="${TITLE_PREFIX:-vLLM vs LLMServingSim}"

# An example is any directory holding a config.json, named on the command
# line by its path under EXAMPLES_DIR -- <hardware>/<model> in tree, one
# flat <hardware>--<model>--<variant> folder in some out-of-tree platforms.
# Each carries its own config.json, so nothing has to be kept in sync with a
# parallel configs/ tree. They live under this folder by default; an
# out-of-tree platform ships its own, so point EXAMPLES_DIR at its examples/
# folder and paths outside this repo are passed through absolute.
EXAMPLES_DIR="${EXAMPLES_DIR:-$SCRIPT_DIR}"

DEFAULT_EXAMPLES=(
    "RTXPRO6000/Llama-3.1-8B"
    "RTXPRO6000/Qwen3-32B"
    "RTXPRO6000/Qwen3-30B-A3B-Instruct-2507"
    "RTX4090/Llama-3.1-8B"
)

# Relative to the repo root when the path is inside it (the simulator runs
# from astra-sim/ and prefixes ../), absolute when it is not.
repo_relative_path() {
    local path="$1"
    case "$path" in
        "$REPO_ROOT"/*) printf '%s\n' "${path#"$REPO_ROOT"/}" ;;
        *) printf '%s\n' "$path" ;;
    esac
}

validate_example() {
    local model_dir="$1"   # <hardware>/<model>
    local vllm_dir="$EXAMPLES_DIR/$model_dir/vllm"
    local sim_csv="$EXAMPLES_DIR/$model_dir/outputs/sim.csv"
    local sim_log="$EXAMPLES_DIR/$model_dir/outputs/sim.log"
    local vllm_dir_rel
    local sim_csv_rel
    local sim_log_rel

    [[ -d "$vllm_dir" ]] || { echo "Missing vLLM bench dir: $vllm_dir" >&2; exit 1; }
    [[ -f "$sim_csv" ]] || { echo "Missing sim CSV: $sim_csv" >&2; exit 1; }
    [[ -f "$sim_log" ]] || { echo "Missing sim log: $sim_log" >&2; exit 1; }

    vllm_dir_rel="$(repo_relative_path "$vllm_dir")"
    sim_csv_rel="$(repo_relative_path "$sim_csv")"
    sim_log_rel="$(repo_relative_path "$sim_log")"

    local cmd=(
        "$PYTHON" -m llmservingsim.bench validate
        --bench-dir "$vllm_dir_rel"
        --sim-csv "$sim_csv_rel"
        --sim-log "$sim_log_rel"
        --output-subdir "$OUTPUT_SUBDIR"
        --title "$TITLE_PREFIX - $model_dir"
        --log-level "$LOG_LEVEL"
    )

    [[ -n "$PREFIX" ]] && cmd+=(--prefix "$PREFIX")

    echo "============================================================"
    echo "Example:    $model_dir"
    echo "Bench dir:  $vllm_dir_rel"
    echo "Sim CSV:    $sim_csv_rel"
    echo "Sim log:    $sim_log_rel"
    echo "Output dir: $model_dir/validation"
    echo "Running: ${cmd[*]}"

    (
        cd "$REPO_ROOT"
        "${cmd[@]}"
    )
}

if [[ $# -eq 0 ]]; then
    if [[ "$EXAMPLES_DIR" == "$SCRIPT_DIR" ]]; then
        set -- "${DEFAULT_EXAMPLES[@]}"
    else
        # An out-of-tree examples folder has no curated list: run every
        # directory holding a config.json, at either depth.
        mapfile -t found < <(cd "$EXAMPLES_DIR" && find . -mindepth 2 -maxdepth 3 \
            -name config.json -printf '%h\n' 2>/dev/null | sed 's|^\./||' | sort)
        [[ ${#found[@]} -gt 0 ]] || { echo "No examples under $EXAMPLES_DIR" >&2; exit 2; }
        set -- "${found[@]}"
    fi
fi

for example in "$@"; do
    if [[ -d "$EXAMPLES_DIR/$example" ]]; then
        validate_example "$example"
    else
        echo "Unknown example: $example" >&2
        echo "Known examples: ${DEFAULT_EXAMPLES[*]}" >&2
        exit 2
    fi
done
