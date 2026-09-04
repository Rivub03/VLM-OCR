#!/usr/bin/env bash
set -euo pipefail

: "${OCR_MODEL_ID:?OCR_MODEL_ID must be set}"
: "${OCR_VRAM_CAP_GIB:=18}"
: "${OCR_MAX_MODEL_LEN:=8192}"
: "${OCR_MAX_NUM_SEQS:=4}"

total_mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
if [[ -z "${total_mib}" ]]; then
  echo "No NVIDIA GPU was detected by nvidia-smi." >&2
  exit 1
fi

# vLLM reserves this fraction for the model executor (weights + KV cache).  Keep
# 512 MiB for CUDA/driver overhead and never ask for more than 95% of the card.
gpu_fraction="$(python3 -c "total=${total_mib}; cap=${OCR_VRAM_CAP_GIB}*1024; print(f'{max(0.05, min((cap - 512) / total, 0.95)):.4f}')")"
if ! python3 -c "import sys; sys.exit(0 if ${total_mib} >= 4096 else 1)"; then
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

