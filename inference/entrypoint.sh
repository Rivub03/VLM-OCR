#!/usr/bin/env bash
set -euo pipefail

: "${OCR_MODEL_ID:?OCR_MODEL_ID must be set}"
: "${OCR_VRAM_CAP_GIB:=18}"
: "${OCR_MAX_MODEL_LEN:=8192}"
: "${OCR_MAX_NUM_SEQS:=4}"

# Preserve the NVIDIA error text: an empty/N/A value normally means that the
# container was started without a GPU device request or without the NVIDIA
# Container Toolkit, not that the GPU has insufficient VRAM.
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

# vLLM reserves this fraction for the model executor (weights + KV cache).  Keep
# 512 MiB for CUDA/driver overhead and never ask for more than 95% of the card.
gpu_fraction="$(TOTAL_MIB="${total_mib}" CAP_GIB="${OCR_VRAM_CAP_GIB}" python3 -c 'import os
total = float(os.environ["TOTAL_MIB"])
cap = float(os.environ["CAP_GIB"]) * 1024
print(f"{max(0.05, min((cap - 512) / total, 0.95)):.4f}")')"
if (( total_mib < 4096 )); then
  echo "At least 4 GiB VRAM is required; GPU reports ${total_mib} MiB." >&2
  exit 1
fi

echo "Serving ${OCR_MODEL_ID} with vLLM; executor cap=${OCR_VRAM_CAP_GIB} GiB, GPU fraction=${gpu_fraction}"
exec vllm serve "${OCR_MODEL_ID}" \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name "${OCR_MODEL_ID}" \
  --trust-remote-code \
  --dtype auto \
  --gpu-memory-utilization "${gpu_fraction}" \
  --max-model-len "${OCR_MAX_MODEL_LEN}" \
  --max-num-seqs "${OCR_MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${OCR_MAX_MODEL_LEN}" \
  --limit-mm-per-prompt '{"image": 1}'
