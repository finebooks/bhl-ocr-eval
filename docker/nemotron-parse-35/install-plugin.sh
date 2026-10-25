#!/bin/bash
# Install NVIDIA's vLLM plugin for Nemotron 3.5 Parse into vllm/vllm-openai:v0.26.0's own Python.
#
# Used two ways: by the Dockerfile next to this file, and at job startup on the bare base image
# (drivers/nemotron-parse-35-port.py docstring). Package source and lock both come from the
# model repo at one revision: the lock is NVIDIA's reference/parse_3_5/inference.constraints.txt
# minus its `nemotron-parse==3.5.0rc3` line, because the package installed here is the post7
# source at that same revision. vLLM itself is not requested, so the base image's 0.26.0 stays;
# the final check fails the install if anything moved it.
set -euo pipefail
REV=62cb1e4ba153d113ff64aaf2be6e068c377cf53c
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('nvidia/Nemotron-3.5-Parse', revision='$REV', local_dir='/tmp/np', allow_patterns=['pyproject.toml','LICENSE','THIRD_PARTY_NOTICES.md','TOKENIZER_ATTRIBUTION.md','postprocessing.py','latex2html.py','nemotron_parse/*','nemotron_parse/**','vllm_tied_patch/*','reference/parse_3_5/inference.constraints.txt'])"
grep -v '^nemotron-parse==' /tmp/np/reference/parse_3_5/inference.constraints.txt > /tmp/np-constraints.txt
uv pip install --system --constraint /tmp/np-constraints.txt /tmp/np
python3 -c "import vllm, torch, transformers; print('versions', vllm.__version__, torch.__version__, transformers.__version__); assert vllm.__version__ == '0.26.0'"
