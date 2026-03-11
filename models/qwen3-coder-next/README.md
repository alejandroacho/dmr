# qwen3-coder-next

Qwen3-Coder-Next FP8 inference on GB10 Blackwell (SM121).

**Model:** `Qwen/Qwen3-Coder-Next-FP8` (~95 GB, FP8 native quantization)
**Docker image:** `blackwell-vllm:latest`

## Quick start

```bash
# Download weights
just download-qwen3

# Build image
just build-qwen3

# Start gateway in focus_code profile
just focus-code
```

## Files

### `Dockerfile`

Builds `blackwell-vllm:latest` on top of `vllm/vllm-openai:cu130-nightly` (the official vLLM image with CUDA 13.0 support). It bakes two GB10 compatibility patches into the image at build time so no runtime mounts or environment variables are needed:

1. Applies `fix_slowness.diff` in reverse (`patch -R`) to undo vLLM PR #34279.
2. Copies `_triton_alloc_setup.py` and `_triton_alloc_setup.pth` into Python's `dist-packages`, where they are auto-loaded on every Python startup.

The `|| echo "...skipping"` guard on the patch step makes the build idempotent: if a future nightly already has that PR reverted upstream, the build continues instead of failing.

### `fix_slowness.diff`

A git-format diff that records the changes introduced by vLLM PR #34279. That PR added `tl.int64` type annotations to the `stride_*` parameters of the Triton MoE kernels (`fused_moe_kernel` and `fused_moe_kernel_gptq_awq`). On GB10 (SM121) those annotations cause Triton to compile the kernels much more slowly — observed as severely reduced token throughput.

The Dockerfile applies this diff with `patch -R` (reverse), which **undoes** those annotations and restores the original untyped parameters. The diff is stored here rather than being deleted so you can inspect exactly what was changed.

### `_triton_alloc_setup.py`

Monkey-patches Triton's `NullAllocator` to use PyTorch's CUDA caching allocator. Without this patch, running Qwen3-Next produces the error:

```
Kernel requires runtime memory allocation but no allocator was provided
```

Triton's `NullAllocator` is a no-op placeholder that crashes when a kernel actually needs to allocate scratch memory at runtime. This file replaces its `__call__` method with a one-liner that delegates to `torch.cuda.caching_allocator_alloc`, which performs the real allocation using the same memory pool PyTorch already manages.

The `try/except` block ensures the patch fails silently if Triton is not installed or its internal API changes in a future version.

### `_triton_alloc_setup.pth`

A Python `.pth` file. Python reads all `.pth` files found in `site-packages` / `dist-packages` at interpreter startup and executes any `import` statements they contain. This file contains a single line:

```
import _triton_alloc_setup
```

This causes the allocator patch to be applied **automatically every time Python starts** inside the container, without modifying any vLLM startup scripts or adding environment variables.

### `download.sh`

Downloads the model weights from HuggingFace using the `hf` CLI with 8 parallel workers. Two modes:

| Mode | Command | Destination |
|------|---------|-------------|
| HF cache (default) | `./download.sh` | `~/.cache/huggingface/hub` |
| Local directory | `MODELS_DIR=/path ./download.sh` | `/path/qwen3-coder-next-fp8` |

The HF cache mode is what the gateway uses by default — vLLM can load directly from the cache by model ID. The local directory mode is useful if you want explicit control over where the 95 GB of weights are stored (e.g., on a separate NVMe).

Re-running the script resumes an interrupted download automatically (huggingface_hub handles this).
