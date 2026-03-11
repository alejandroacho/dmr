# GPT-OSS 120B — MXFP4 CUTLASS sm_121 (GB10 Blackwell)
#
# This image depends on vllm-mxfp4-spark:latest, which must be built first
# from the spark-vllm-mxfp4-docker fork:
#
#   just build-spark   (builds from $SPARK_REPO, defaults to ~/spark-vllm-mxfp4-docker)
#   just build-gpt-oss (tags the result as gpt-oss-120b-vllm:latest)
#
# The spark image already contains everything needed:
#   - CUTLASS MXFP4 MoE kernels for SM121
#   - PyTorch + Triton compiled natively for GB10
#   - fastsafetensors for fast NVMe weight loading
#   - FP8 KV cache with GPT-OSS attention sink support

FROM vllm-mxfp4-spark:latest
