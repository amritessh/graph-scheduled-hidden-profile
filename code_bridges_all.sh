#!/bin/bash
MODEL="vllm:8000/openai/gpt-oss-120b"

echo "=== Coding bridge conversations (DV4) for all 3 models ==="

python -m gshp.cli code-bridges results/vllm_8000_qwen3-30b_20260405_141321 \
  --model "$MODEL" 2>&1 | sed 's/^/[qwen3] /' &
PID1=$!

python -m gshp.cli code-bridges results/vllm_8000_openai_gpt-oss-120b_20260610_002603 \
  --model "$MODEL" 2>&1 | sed 's/^/[gpt120b] /' &
PID2=$!

python -m gshp.cli code-bridges results/vllm_8000_meta-llama_Llama-3.1-8B-Instruct_20260610_095842 \
  --model "$MODEL" 2>&1 | sed 's/^/[llama8b] /' &
PID3=$!

wait $PID1 $PID2 $PID3
echo "=== All bridge coding done ==="
