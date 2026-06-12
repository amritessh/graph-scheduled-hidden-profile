#!/bin/bash
GPU0="vllm:8002/openai/gpt-oss-120b"
GPU1="vllm:8000/openai/gpt-oss-120b"
WORKERS=10

echo "=== Starting all 3 classifiers in parallel (GPU0=port8002, GPU1=port8000) ==="

python -m gshp.cli classify-facts results/vllm_8000_qwen3-30b_20260405_141321 \
  --model "$GPU1" --workers $WORKERS 2>&1 | sed 's/^/[qwen3|gpu1] /' &
PID1=$!

python -m gshp.cli classify-facts results/vllm_8000_openai_gpt-oss-120b_20260610_002603 \
  --model "$GPU0" --workers $WORKERS 2>&1 | sed 's/^/[gpt120b|gpu0] /' &
PID2=$!

python -m gshp.cli classify-facts results/vllm_8000_meta-llama_Llama-3.1-8B-Instruct_20260610_095842 \
  --model "$GPU0" --workers $WORKERS 2>&1 | sed 's/^/[llama8b|gpu0] /' &
PID3=$!

wait $PID1 $PID2 $PID3
echo "=== All done ==="
