#!/usr/bin/env bash
set -euo pipefail

CONCURRENCY=$1
PORT=$2
NUM_INPUTS=$3
REPEATS_PER_INPUT=$4
MAX_TOKENS=$5
OUTPUT_DIR=$6
DATASET=${7:-evaluation/datasets/aime.json}
MODEL_NAMES=${8:-}

RESULT_FILE="$OUTPUT_DIR/result.csv"

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"
: > "$RESULT_FILE"

if [ "$CONCURRENCY" -le 0 ]; then
    echo "CONCURRENCY must be positive; got CONCURRENCY=$CONCURRENCY"
    exit 1
fi

TOTAL_TASKS=$((NUM_INPUTS * REPEATS_PER_INPUT))
if [ "$TOTAL_TASKS" -le 0 ]; then
    echo "TOTAL_TASKS must be positive; got NUM_INPUTS=$NUM_INPUTS REPEATS_PER_INPUT=$REPEATS_PER_INPUT"
    exit 1
fi

DATASET_SIZE=$(python -c 'import json, sys; print(len(json.load(open(sys.argv[1]))))' "$DATASET")
if [ "$NUM_INPUTS" -gt "$DATASET_SIZE" ]; then
    echo "NUM_INPUTS must be <= dataset size; got NUM_INPUTS=$NUM_INPUTS dataset_size=$DATASET_SIZE dataset=$DATASET"
    exit 1
fi

task_index=0
finished=0

declare -A running

cleanup() {
    local pid
    for pid in "${!running[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}

trap cleanup EXIT INT TERM

launch() {
    local worker_id=$1
    local input_id=$2
    local task_id=$3
    local record=$4

    python evaluation/test_throughput.py \
        --worker_id "$worker_id" \
        --input_id "$input_id" \
        --port "$PORT" \
        --input_file "$DATASET" \
        --output_file "$RESULT_FILE" \
        --record "$record" \
        --max_tokens "$MAX_TOKENS" \
        --model_names "$MODEL_NAMES" &

    pid=$!
    running[$pid]="$task_id:$worker_id"
}

# ---------------- initial concurrent slots ----------------
initial_slots=$CONCURRENCY
if [ "$TOTAL_TASKS" -lt "$initial_slots" ]; then
    initial_slots=$TOTAL_TASKS
fi

for ((i=0; i<initial_slots; i++)); do
    input_id=$((task_index % NUM_INPUTS))
    launch $i $input_id $task_index 1
    task_index=$((task_index + 1))
done

# ---------------- scheduler ----------------
while true; do

    for pid in "${!running[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then

            meta=${running[$pid]}
            IFS=":" read task_id worker_id <<< "$meta"
            unset running[$pid]

            if ! wait "$pid"; then
                echo "Evaluation task failed: task_id=$task_id worker_id=$worker_id" >&2
                exit 1
            fi

            if [ "$task_id" -lt "$TOTAL_TASKS" ]; then
                finished=$((finished + 1))
            fi

            input_id=$((task_index % NUM_INPUTS))

            if [ "$task_index" -lt "$TOTAL_TASKS" ]; then
                launch "$worker_id" "$input_id" "$task_index" 1
            else
                launch "$worker_id" "$input_id" "$task_index" 0
            fi

            task_index=$((task_index + 1))
        fi
    done

    if [ "$finished" -ge "$TOTAL_TASKS" ]; then
        break
    fi

    sleep 0.05
done

for pid in "${!running[@]}"; do
    meta=${running[$pid]}
    IFS=":" read task_id worker_id <<< "$meta"
    if ! wait "$pid"; then
        echo "Evaluation task failed: task_id=$task_id worker_id=$worker_id" >&2
        exit 1
    fi
    unset running[$pid]
done

trap - EXIT INT TERM

ACTUAL_ROWS=$(python -c 'import csv, sys; print(sum(1 for _ in csv.DictReader(open(sys.argv[1]))))' "$RESULT_FILE")
if [ "$ACTUAL_ROWS" -ne "$TOTAL_TASKS" ]; then
    echo "Expected $TOTAL_TASKS result rows, found $ACTUAL_ROWS in $RESULT_FILE" >&2
    exit 1
fi

python evaluation/report.py --result "$RESULT_FILE"

echo "Done: collected $TOTAL_TASKS valid samples"
