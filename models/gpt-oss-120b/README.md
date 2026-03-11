# gpt-oss-120b

GPT-OSS 120B inference on GB10 Blackwell (SM121) using MXFP4 quantization.

**Model:** `openai/gpt-oss-120b` (~240 GB, MXFP4 quantized by OpenAI for GB10)
**Docker image:** `vllm-mxfp4-spark:latest`

## Quick start

```bash
# Download weights (~240 GB)
just download-gpt-oss

# Build the spark image from your fork (only needed once)
just build-spark

# Switch gateway to focus profile (loads GPT-OSS)
just focus
```

## Prerequisites

The `vllm-mxfp4-spark:latest` image must be built from [github.com/alejandroacho/gb10-vllm-mxfp4-docker](https://github.com/alejandroacho/gb10-vllm-mxfp4-docker) before running this model. That repo contains GB10-specific patches that are not in the official vLLM image. Set the `SPARK_REPO` environment variable to point to your local clone, or place it at `~/gb10-vllm-mxfp4-docker` (the default).

## Files

### `Dockerfile`

A single-line Dockerfile that inherits from `vllm-mxfp4-spark:latest`:

```dockerfile
FROM vllm-mxfp4-spark:latest
```

No additional patches are needed because the spark base image already includes everything required for GPT-OSS 120B on GB10:

- **CUTLASS MXFP4 MoE kernels** compiled for SM121 — these are the custom CUDA kernels that make MXFP4 inference possible on Blackwell. Not present in upstream vLLM.
- **PyTorch + Triton** compiled natively for GB10 — ensures full SM121 support without fallback paths.
- **fastsafetensors** — fast parallel weight loader optimized for NVMe, significantly reduces the ~240 GB model load time.
- **FP8 KV cache** with GPT-OSS attention sink support — keeps the key/value cache in FP8 to save VRAM during long-context inference.

The Dockerfile exists as a placeholder so this model has the same structure as other models in this repo (its own folder, its own image tag). It also makes it easy to add GPT-OSS-specific layers in the future if needed.

### `download.sh`

Downloads the model weights from HuggingFace to a local directory using the `hf` CLI with 8 parallel workers.

```
openai/gpt-oss-120b  →  $MODELS_DIR/gpt-oss-120b-q8
```

Unlike Qwen3, GPT-OSS **must** be downloaded to a local directory (not the HF cache) because the spark image uses `fastsafetensors` to load it directly from disk using the `--load-format fastsafetensors` flag, which requires a local path.

The default destination is `~/Models/gpt-oss-120b-q8`. Override with `MODELS_DIR=/your/path ./download.sh`.

Re-running the script resumes an interrupted download automatically.
