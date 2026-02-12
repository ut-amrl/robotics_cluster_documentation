#!/usr/bin/env bash
#SBATCH -J vllm_single             # Job name
#SBATCH -o vllm_single.%j.out      # Output file (%j = job ID)
#SBATCH -e vllm_single.%j.err      # Error file
#SBATCH -p amd-rtx                 # Queue (partition)
#SBATCH -N 1                       # 1 node
#SBATCH -n 1                       # 1 task
#SBATCH -t 01:00:00                # Wall time
#SBATCH -A IRI26004                # Allocation

set -e

# --------------------------------------------------------------------------
# Serve Qwen2.5-14B-Instruct on a single GPU with vLLM, send a test query,
# and shut down. Demonstrates basic single-GPU model serving.
# --------------------------------------------------------------------------

MODEL="Qwen/Qwen2.5-14B-Instruct"
PORT=8000

echo "=== vLLM Single-GPU Serving ==="
echo "Date:  $(date)"
echo "Node:  $(hostname)"
echo "Job:   $SLURM_JOB_ID"
echo "Model: $MODEL"

# Use $WORK for HuggingFace model cache (avoid $HOME quota)
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

# Install vLLM if not present
if python -c "import vllm" 2>/dev/null; then
    echo "vLLM $(python -c 'import vllm; print(vllm.__version__)') already installed"
else
    echo "Installing vLLM..."
    pip install --upgrade pip
    pip install vllm
fi

# --------------------------------------------------------------------------
# Start the server in the background
# --------------------------------------------------------------------------
echo ""
echo "=== Starting vLLM server (tp=1) ==="
vllm serve "$MODEL" \
    --port $PORT \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 &
SERVER_PID=$!

# Wait for the server to be ready
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
      {\"role\": \"user\", \"content\": \"Explain reinforcement learning in two sentences.\"}
    ],
    \"max_tokens\": 128,
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
