# Reuse the node's ARM64 CUDA 13 / PyTorch build with GB10 support.
FROM media-node:latest
ARG FISH_REF=214da3cd841bda85da2496b96cd3c4d7edb1337e
RUN git clone https://github.com/fishaudio/fish-speech.git /opt/fish-speech \
    && cd /opt/fish-speech && git checkout ${FISH_REF}
WORKDIR /opt/fish-speech
# Inference dependencies only; retain the working CUDA torch/torchaudio wheels.
RUN pip install --no-cache-dir 'transformers==4.57.3' lightning hydra-core \
    natsort librosa rich kui loguru loralib pyrootutils resampy \
    'einx[torch]==0.2.2' ormsgpack tiktoken cachetools descript-audio-codec \
    && pip install --no-cache-dir 'protobuf>=3.20,<6'
ENV PYTHONPATH=/opt/fish-speech
EXPOSE 8011
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD curl -fsS http://localhost:8011/v1/health || exit 1
ENTRYPOINT ["python3", "tools/api_server.py"]
CMD ["--llama-checkpoint-path", "/models/fish-s2-pro", "--decoder-checkpoint-path", "/models/fish-s2-pro/codec.pth", "--listen", "0.0.0.0:8011", "--workers", "1"]
