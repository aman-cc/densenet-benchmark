#!/bin/bash
set -euo pipefail

OUTPUT_DIR="./results"
GPU_ENABLED="true"

while [[ $# -gt 0 ]]; do
	key="$1"
	case $key in
		--output-dir)
			OUTPUT_DIR="$2"; shift; shift;;
		--gpu-enabled)
			GPU_ENABLED="$2"; shift; shift;;
		*)
			shift;;
	esac
done

mkdir -p "$OUTPUT_DIR" ./logs

echo "[1/4] Building image..."
docker build -t densenet-bench:latest .

RUN_OPTS=(
	--rm
	--name densenet-bench-app
	-v "$(pwd)/$OUTPUT_DIR:/srv/results"
	-v "$(pwd)/logs:/srv/logs"
)
if [[ "$GPU_ENABLED" == "true" ]]; then
	RUN_OPTS+=(--gpus all)
fi

echo "[2/4] Starting TensorBoard..."
docker compose up -d tensorboard-logger || docker-compose up -d tensorboard-logger
docker compose up -d tensorboard-profiler || docker-compose up -d tensorboard-profiler

echo "[3/4] Running benchmark container..."
docker run "${RUN_OPTS[@]}" densenet-bench

echo "[4/4] Summary:"
if [[ -f "$OUTPUT_DIR/benchmark_results.csv" ]]; then
	printf "\nLatest results (tail):\n"
	tail -n +1 "$OUTPUT_DIR/benchmark_results.csv" | column -t -s,
else
	echo "No results found at $OUTPUT_DIR/benchmark_results.csv"
fi

