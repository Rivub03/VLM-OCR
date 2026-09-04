#!/usr/bin/env bash
set -euo pipefail

: "${OCR_MODEL_ID:?OCR_MODEL_ID must be set}"
: "${OCR_VRAM_CAP_GIB:=18}"
: "${OCR_UMA_TOTAL_MEMORY_GIB:=}"
: "${OCR_MAX_NUM_SEQS:=4}"
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1)" || {
  echo "GPU is not available inside the inference container." >&2
  echo "Ensure Docker has NVIDIA Container Toolkit support, then run: docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi" >&2
  echo "nvidia-smi output: ${gpu_name}" >&2
  exit 1
}

gpu_query="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>&1 || true)"
total_mib="$(printf '%s\n' "${gpu_query}" | awk '{ for (i = 1; i <= NF; i++) if ($i ~ /^[0-9]+$/) { print $i; exit } }')"
if [[ -z "${total_mib}" ]]; then
  if [[ -n "${OCR_UMA_TOTAL_MEMORY_GIB}" ]]; then
    if ! [[ "${OCR_UMA_TOTAL_MEMORY_GIB}" =~ ^[0-9]+$ ]]; then
      echo "OCR_UMA_TOTAL_MEMORY_GIB must be a whole number of GiB." >&2
      exit 1
    fi
    total_mib=$((OCR_UMA_TOTAL_MEMORY_GIB * 1024))
  else
    total_mib="$(awk '/MemTotal:/ { print int($2 / 1024); exit }' /proc/meminfo)"
  fi
  if [[ -z "${total_mib}" || "${total_mib}" -lt 4096 ]]; then
    echo "Could not determine sufficient unified system memory for GPU ${gpu_name}." >&2
    exit 1
  fi
  echo "GPU ${gpu_name} reports no dedicated VRAM; using ${total_mib} MiB unified system memory to derive the SGLang budget." >&2
fi
memory_fraction="$(TOTAL_MIB="${total_mib}" CAP_GIB="${OCR_VRAM_CAP_GIB}" python3 -c 'import os
total = float(os.environ["TOTAL_MIB"])
cap = float(os.environ["CAP_GIB"]) * 1024
print(f"{max(0.05, min((cap - 512) / total, 0.95)):.4f}")')"
echo "Serving ${OCR_MODEL_ID} with SGLang on ${gpu_name}; static memory fraction=${memory_fraction}"
exec python3 -m sglang.launch_server \
  --model-path "${OCR_MODEL_ID}" \
  --host 0.0.0.0 \
  --port 8000 \
  --mem-fraction-static "${memory_fraction}" \
  --max-running-requests "${OCR_MAX_NUM_SEQS}"
