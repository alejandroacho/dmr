# Custom vLLM image for GB10 Blackwell (sm_121) with Qwen3-Coder-Next patches.
#
# Patches applied:
#   - fix_slowness.diff reverted: removes tl.int64 Triton stride annotations (PR #34279)
#     that cause slowness on GB10. Already reverted in upstream but not in this nightly.
#   - _triton_alloc_setup.py: patches NullAllocator to use CUDA caching allocator,
#     fixing "Kernel requires runtime memory allocation" errors with Qwen3-Next.
#
# Build:
#   docker build -f Dockerfile.vllm -t blackwell-vllm:latest .

FROM vllm/vllm-openai:cu130-nightly

COPY fix_slowness.diff /tmp/fix_slowness.diff
COPY _triton_alloc_setup.py /usr/local/lib/python3.12/dist-packages/_triton_alloc_setup.py
COPY _triton_alloc_setup.pth /usr/local/lib/python3.12/dist-packages/_triton_alloc_setup.pth

RUN patch -p1 -R -d /usr/local/lib/python3.12/dist-packages < /tmp/fix_slowness.diff \
    && rm /tmp/fix_slowness.diff \
    || echo "fix_slowness already reverted in this build, skipping"
