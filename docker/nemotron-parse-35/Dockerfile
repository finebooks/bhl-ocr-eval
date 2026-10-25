# Nemotron 3.5 Parse serving image for the BHL OCR benchmark.
#
#   docker buildx build --platform linux/amd64 -t <tag> docker/nemotron-parse-35
#
# = the board-wide engine pin (vllm/vllm-openai:v0.26.0, amd64, by digest)
#   + NVIDIA's vLLM plugin (`nemotron-parse` 3.5.0.post7) from the model repo, installed by
#     install-plugin.sh: the same script a job runs at startup when it uses the bare base
#     image instead, so the two paths cannot drift.
#
# Differs from the vendor Dockerfile on purpose: that one makes a fresh venv and resolves
# `vllm>=0.25,<0.33` again, which can land on a different vLLM than the base. Installing into
# the base env, with vLLM not requested, keeps vLLM 0.26.0 as shipped. The base image already
# ships uv (/usr/local/bin/uv). Weights are NOT baked in: the driver serves the Hub repo
# pinned with --model-revision.
FROM vllm/vllm-openai:v0.26.0@sha256:770fe65b2c73ee74a5c42165cf3433de4048cc2cd9c57a937ca4e35aba5aa87b

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY install-plugin.sh /tmp/install-plugin.sh
RUN --mount=type=cache,target=/root/.cache/uv bash /tmp/install-plugin.sh

LABEL org.opencontainers.image.title="nemotron-parse-35-vllm" \
      org.opencontainers.image.source="https://github.com/finebooks/bhl-ocr-eval" \
      org.opencontainers.image.description="vllm/vllm-openai:v0.26.0 + nemotron-parse 3.5.0.post7 plugin (nvidia/Nemotron-3.5-Parse@62cb1e4)" \
      org.opencontainers.image.licenses="OpenMDW-1.1"
