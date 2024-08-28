#!/bin/bash

echo "model path: ${MODEL_PATH}"
echo "max model len: $MAX_MODEL_LEN"
echo "tensor parallel size: $NUM_GPU"
echo "gpu memory utilization: $GPU_MEMORY_UTILIZATION"

if [ "$MODE" = "model_hub" ]; then
  # Start the strategy server
  uvicorn app.main2:app \
          --host 0.0.0.0 \
          --port 80
else
  # Start the vLLM OpenAI-Compatiable server
  python3 -m vllm.entrypoints.openai.api_server \
          --model "${MODEL_PATH}" \
          --port 8000 \
          --max-model-len $MAX_MODEL_LEN \
          --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
          --tensor-parallel-size $NUM_GPU \
          --disable-log-stats \
          --disable-log-requests \
          --enforce-eager &

  # Start the strategy server
  uvicorn app.main2:app \
          --host 0.0.0.0 \
          --port 80 &

  # Wait for any process to exit
  wait -n

  # Exit with status of process that exited first
  exit $?
fi
