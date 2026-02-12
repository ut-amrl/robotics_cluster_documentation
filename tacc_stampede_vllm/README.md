# vLLM Serving Guide (Stampede3 / Micromamba)

How to install [vLLM](https://docs.vllm.ai/) on Stampede3 `amd-rtx` nodes and serve large language models with single-GPU, data-parallel, and tensor-parallel configurations.

## Quick Start

```bash
sbatch serve_single_gpu.slurm      # 14B model on 1 GPU
sbatch serve_data_parallel.slurm   # 14B model on 8 GPUs (data parallel)
sbatch serve_tensor_parallel.slurm # 32B model on 2 GPUs (tensor parallel)
```

Each script creates the environment (if needed), installs vLLM, starts the server, sends a test request, and shuts down. Check the output file for results:

```bash
tail -f vllm_*.<job_id>.out
```

## Node Specs

Each `amd-rtx` node has **8x NVIDIA RTX PRO 6000 Blackwell** GPUs with **96 GB VRAM each** (768 GB total GPU memory per node). This is enough to serve:

| Model Size | BF16 Memory | Fits on 1 GPU? | Recommended Config |
|-----------|-------------|-----------------|-------------------|
| 7-8B | ~16 GB | Yes | `-tp 1` |
| 14B | ~28 GB | Yes | `-tp 1` |
| 32B | ~64 GB | Yes | `-tp 1` or `-tp 2` for lower latency |
| 70B | ~140 GB | No | `-tp 2` (minimum) or `-tp 4` |
| 70B+ MoE | varies | No | `-tp 4` or `-tp 8` |

## Step 1: Create the Micromamba Environment

```bash
micromamba create -n vllm python=3.11 -c conda-forge -y
micromamba activate vllm
```

## Step 2: Install vLLM

```bash
pip install --upgrade pip
pip install vllm
```

This installs vLLM with PyTorch and all dependencies (~5 GB). The install includes PyTorch 2.9+ with CUDA 12.8 support, which works on Blackwell GPUs out of the box.

Verify the installation:

```bash
python -c "
import vllm, torch
print(f'vLLM:    {vllm.__version__}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA:    {torch.cuda.is_available()} ({torch.version.cuda})')
print(f'GPUs:    {torch.cuda.device_count()}x {torch.cuda.get_device_name(0)}')
"
```

Expected output:

```
vLLM:    0.15.1
PyTorch: 2.9.1+cu128
CUDA:    True (12.8)
GPUs:    8x NVIDIA RTX PRO 6000 Blackwell Server Edition
```

## Step 3: Set HuggingFace Cache Directory

Models are downloaded from HuggingFace Hub. The default cache is `~/.cache/huggingface`, which is on `$HOME` (15 GB quota). Redirect it to `$WORK`:

```bash
export HF_HOME=$WORK/.cache/huggingface
```

Add this to your `~/.bashrc` to make it permanent:

```bash
echo 'export HF_HOME=$WORK/.cache/huggingface' >> ~/.bashrc
```

### Gated Models

Some models (e.g., Llama, Mistral) require accepting a license on HuggingFace. After accepting, create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and log in:

```bash
pip install huggingface_hub
huggingface-cli login
```

The Qwen models used in these examples are **not** gated and require no token.

## Example 1: Serve a 14B Model on 1 GPU

Serve [Qwen/Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct), a high-quality 14B instruction-tuned model, on a single GPU:

```bash
vllm serve Qwen/Qwen2.5-14B-Instruct \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9
```

The first launch downloads the model (~28 GB) and compiles CUDA graphs (2-5 minutes). Subsequent starts are faster.

### Test with curl

Once the server prints `Application startup complete`, send a request:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-14B-Instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain reinforcement learning in two sentences."}
    ],
    "max_tokens": 128,
    "temperature": 0.7
  }' | python -m json.tool
```

### Test with Python (OpenAI SDK)

vLLM exposes an OpenAI-compatible API. Use the `openai` Python package (installed with vLLM):

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-14B-Instruct",
    messages=[{"role": "user", "content": "What is Q-learning?"}],
    max_tokens=256,
)
print(response.choices[0].message.content)
```

## Example 2: Serve a 14B Model on 8 GPUs (Data Parallel)

Data parallelism runs **multiple replicas** of the same model, each on its own GPU. vLLM automatically load-balances incoming requests across replicas. This is ideal for **maximizing throughput** when serving many concurrent users.

```bash
vllm serve Qwen/Qwen2.5-14B-Instruct \
    --port 8000 \
    --tensor-parallel-size 1 \
    --data-parallel-size 8 \
    --gpu-memory-utilization 0.9
```

This starts 8 independent model replicas (one per GPU), fronted by a single API endpoint on port 8000. The 14B model uses ~28 GB per GPU, leaving ~68 GB per GPU for KV cache — enough for thousands of concurrent sequences.

The API is identical to Example 1; clients see a single endpoint.

## Example 3: Serve a 32B Model on 2 GPUs (Tensor Parallel)

Tensor parallelism **shards a single model** across multiple GPUs. This is required when a model doesn't fit in one GPU's memory, and also reduces per-token latency by splitting computation.

Serve [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) across 2 GPUs:

```bash
vllm serve Qwen/Qwen2.5-32B-Instruct \
    --port 8000 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9
```

The 32B model in BF16 needs ~64 GB. With `-tp 2`, each GPU holds ~32 GB of weights, leaving ~64 GB per GPU for KV cache.

> **Note:** The 32B model technically fits on a single 96 GB GPU, but tensor parallelism halves per-token latency by splitting the computation. Use `-tp 1` if you prefer to maximize throughput with more KV cache headroom.

### Scaling Up

For larger models, increase the tensor parallel size:

```bash
# 70B model across 4 GPUs (~140 GB in BF16)
vllm serve Qwen/Qwen2.5-72B-Instruct -tp 4

# Combine TP + DP: shard across 2 GPUs, run 4 replicas = 8 GPUs total
vllm serve Qwen/Qwen2.5-32B-Instruct -tp 2 -dp 4
```

## Running as a Batch Job

For long-running inference servers, submit via `sbatch`. The included scripts handle environment setup, model serving, test queries, and cleanup. Use them as templates:

```bash
sbatch serve_single_gpu.slurm       # Example 1
sbatch serve_data_parallel.slurm    # Example 2
sbatch serve_tensor_parallel.slurm  # Example 3
```

### Keeping the Server Running

The example scripts run a test query and exit. To keep the server running for the duration of your job allocation (e.g., for interactive use from another terminal):

```bash
# In your slurm script, replace the curl test with:
vllm serve Qwen/Qwen2.5-14B-Instruct --port 8000 -tp 1
# (This blocks until the job's wall time expires or the job is cancelled.)
```

Then from a login node, find your compute node and SSH to it:

```bash
squeue -u $USER        # find the node name (NODELIST column)
ssh <node-name>        # e.g., ssh c571-002
curl http://localhost:8000/v1/models
```

## Performance Tips

1. **First launch is slow.** vLLM compiles CUDA graphs on first start (~2-10 minutes depending on model size and TP degree). The compiled cache is saved to `~/.cache/vllm/` and reused on subsequent launches.

2. **Use `--gpu-memory-utilization 0.9`** (default) to maximize KV cache. Lower it only if you need GPU memory for other processes.

3. **Data parallel vs. tensor parallel:**
   - Use **data parallel** (`-dp N`) to maximize throughput (requests/second) for models that fit on 1 GPU.
   - Use **tensor parallel** (`-tp N`) to serve models larger than 1 GPU, or to reduce latency per request.
   - Combine both (`-tp 2 -dp 4`) for large models with high throughput needs.

4. **Model downloads:** Large models take time to download. Pre-download before your batch job:
   ```bash
   # On a login node (no GPU needed):
   pip install huggingface_hub
   huggingface-cli download Qwen/Qwen2.5-14B-Instruct
   ```

## Files in This Directory

| File | Description |
|------|-------------|
| [`serve_single_gpu.slurm`](serve_single_gpu.slurm) | Serve Qwen2.5-14B-Instruct on 1 GPU with a test query |
| [`serve_data_parallel.slurm`](serve_data_parallel.slurm) | Serve Qwen2.5-14B-Instruct on 8 GPUs (data parallel) with a test query |
| [`serve_tensor_parallel.slurm`](serve_tensor_parallel.slurm) | Serve Qwen2.5-32B-Instruct on 2 GPUs (tensor parallel) with a test query |
