#!/usr/bin/env bash
#SBATCH -J vllm_tp2                # Job name
#SBATCH -o vllm_tp2.%j.out         # Output file (%j = job ID)
#SBATCH -e vllm_tp2.%j.err         # Error file
#SBATCH -p amd-rtx                 # Queue (partition)
#SBATCH -N 1                       # 1 node
#SBATCH -n 1                       # 1 task
#SBATCH -t 01:00:00                # Wall time
#SBATCH -A IRI26004                # Allocation

set -e

# --------------------------------------------------------------------------
# Serve Qwen2.5-32B-Instruct with 2-way tensor parallelism. The model is
# sharded across 2 GPUs for lower latency per request.
# --------------------------------------------------------------------------

MODEL="Qwen/Qwen2.5-32B-Instruct"
PORT=8000
TP=2

echo "=== vLLM Tensor-Parallel Serving ==="
echo "Date:  $(date)"
echo "Node:  $(hostname)"
echo "Job:   $SLURM_JOB_ID"
echo "Model: $MODEL"
echo "Config: tp=$TP ($TP GPUs)"

# Use $WORK for HuggingFace model cache
export HF_HOME=$WORK/.cache/huggingface

# Initialize micromamba
export MAMBA_EXE="$HOME/.local/bin/micromamba"
export MAMBA_ROOT_PREFIX="$WORK/micromamba"
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2>/dev/null)"

# --------------------------------------------------------------------------
# Create / activate environment
# --------------------------------------------------------------------------
ENV_NAME="vllm"
if micromamba env list | grep -q "$ENV_NAME"; then
    echo "Environment '$ENV_NAME' exists, activating..."
else
    echo "Creating environment '$ENV_NAME'..."
    micromamba create -n "$ENV_NAME" python=3.11 -c conda-forge -y
fi
micromamba activate "$ENV_NAME"

if python -c "import vllm" 2>/dev/null; then
    echo "vLLM $(python -c 'import vllm; print(vllm.__version__)') already installed"
else
    echo "Installing vLLM..."
    pip install --upgrade pip
    pip install vllm
fi

# --------------------------------------------------------------------------
# Start the server with tensor parallelism
# --------------------------------------------------------------------------
echo ""
echo "=== Starting vLLM server (tp=$TP) ==="
vllm serve "$MODEL" \
    --port $PORT \
    --tensor-parallel-size $TP \
    --gpu-memory-utilization 0.9 &
SERVER_PID=$!

echo "Waiting for server to start (this may take several minutes on first launch)..."
for i in $(seq 1 120); do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "Server is ready (took ~${i}0 seconds)"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "ERROR: Server process exited unexpectedly"
        exit 1
    fi
    sleep 10
done

# --------------------------------------------------------------------------
# Send a test request
# --------------------------------------------------------------------------
echo ""
echo "=== Test Request ==="
curl -s http://localhost:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [
      {\"role\": \"system\", \"content\": \"You are a helpful assistant.\"},
      {\"role\": \"user\", \"content\": \"Compare and contrast tensor parallelism and data parallelism for serving large language models. Be concise.\"}
    ],
    \"max_tokens\": 256,
    \"temperature\": 0.7
  }" | python -m json.tool

# --------------------------------------------------------------------------
# Cleanup
# --------------------------------------------------------------------------
echo ""
echo "=== Shutting down ==="
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null || true
echo "Date: $(date)"
echo "Done."
