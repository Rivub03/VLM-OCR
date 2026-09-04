#!/usr/bin/env bash
set -euo pipefail

: "${OCR_MODEL_ID:?OCR_MODEL_ID must be set}"
: "${OCR_VRAM_CAP_GIB:=18}"
: "${OCR_MAX_NUM_SEQS:=4}"
# Preserve the NVIDIA error text so host GPU-runtime failures are actionable.
gpu_query="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>&1)" || {
  echo "GPU is not available inside the inference container." >&2
  echo "Ensure Docker has NVIDIA Container Toolkit support, then run: docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi" >&2
  echo "nvidia-smi output: ${gpu_query}" >&2
  exit 1
}
total_mib="$(printf '%s\n' "${gpu_query}" | awk '{ for (i = 1; i <= NF; i++) if ($i ~ /^[0-9]+$/) { print $i; exit } }')"
if [[ -z "${total_mib}" ]]; then
  echo "GPU is not available inside the inference container: nvidia-smi did not return numeric memory.total." >&2
  echo "nvidia-smi output: ${gpu_query}" >&2
  exit 1
fi
memory_fraction="$(TOTAL_MIB="${total_mib}" CAP_GIB="${OCR_VRAM_CAP_GIB}" python3 -c 'import os
total = float(os.environ["TOTAL_MIB"])
cap = float(os.environ["CAP_GIB"]) * 1024
print(f"{max(0.05, min((cap - 512) / total, 0.95)):.4f}")')"
echo "Serving ${OCR_MODEL_ID} with SGLang; static memory fraction=${memory_fraction}"
exec python3 -m sglang.launch_server \
  --model-path "${OCR_MODEL_ID}" \
  --host 0.0.0.0 \
  --port 8000 \
  --mem-fraction-static "${memory_fraction}" \
  --max-running-requests "${OCR_MAX_NUM_SEQS}"
