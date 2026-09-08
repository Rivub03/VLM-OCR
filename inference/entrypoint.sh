#!/usr/bin/env bash
set -euo pipefail

: "${OCR_MODEL_ID:?OCR_MODEL_ID must be set}"
: "${OCR_VRAM_CAP_GIB:=18}"
: "${OCR_UMA_TOTAL_MEMORY_GIB:=}"
: "${OCR_KV_CACHE_MIB:=4096}"
: "${OCR_ENFORCE_EAGER:=false}"
: "${OCR_MAX_MODEL_LEN:=24576}"
: "${OCR_MAX_NUM_SEQS:=4}"
# Upper bound on image tokens the backend may send. dots.ocr's own preprocessor
# allows max_pixels=11289600, roughly 14400 tokens, which overflows the served
# context and makes vLLM reject the request outright. This is a server-side
# backstop; backend/app/preprocess.py sizes pages against the same figure.
: "${OCR_MAX_IMAGE_TOKENS:=8464}"

if ! [[ "${OCR_KV_CACHE_MIB}" =~ ^[0-9]+$ ]] || (( OCR_KV_CACHE_MIB < 256 )); then
  echo "OCR_KV_CACHE_MIB must be a whole number of at least 256 MiB." >&2
  exit 1
fi
if [[ "${OCR_ENFORCE_EAGER}" != "true" && "${OCR_ENFORCE_EAGER}" != "false" ]]; then
  echo "OCR_ENFORCE_EAGER must be true or false." >&2
  exit 1
fi

gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1)" || {
  echo "GPU is not available inside the inference container." >&2
  echo "Ensure Docker has NVIDIA Container Toolkit support, then run: docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi" >&2
  echo "nvidia-smi output: ${gpu_name}" >&2
  exit 1
}

# GB10/DGX Spark is an integrated GPU with unified system memory. NVIDIA
# intentionally returns N/A for memory.total there, even when CUDA is usable.
gpu_query="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>&1 || true)"
total_mib="$(printf '%s\n' "${gpu_query}" | awk '{ for (i = 1; i <= NF; i++) if ($i ~ /^[0-9]+$/) { print $i; exit } }')"
memory_source="nvidia-smi"
if [[ -z "${total_mib}" ]]; then
  memory_source="UMA system memory"
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
  echo "GPU ${gpu_name} reports no dedicated VRAM; using ${total_mib} MiB unified system memory to derive the vLLM budget." >&2
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

echo "Serving ${OCR_MODEL_ID} with vLLM on ${gpu_name}; KV cache=${OCR_KV_CACHE_MIB} MiB, GPU fraction=${gpu_fraction} (ignored with explicit cache), memory source=${memory_source}"

# One image token covers a 28x28 pixel cell (patch 14, spatial merge 2).
max_pixels=$((OCR_MAX_IMAGE_TOKENS * 28 * 28))

vllm_args=(
  serve "${OCR_MODEL_ID}"
  --host 0.0.0.0
  --port 8000
  --served-model-name "${OCR_MODEL_ID}"
  --trust-remote-code
  --dtype auto
  --gpu-memory-utilization "${gpu_fraction}"
  --kv-cache-memory-bytes "$((OCR_KV_CACHE_MIB * 1024 * 1024))"
  --max-model-len "${OCR_MAX_MODEL_LEN}"
  --max-num-seqs "${OCR_MAX_NUM_SEQS}"
  --max-num-batched-tokens "${OCR_MAX_MODEL_LEN}"
  --limit-mm-per-prompt '{"image": 1}'
  --mm-processor-kwargs "{\"max_pixels\": ${max_pixels}}"
)
if [[ "${OCR_ENFORCE_EAGER}" == "true" ]]; then
  vllm_args+=(--enforce-eager)
fi

exec vllm "${vllm_args[@]}"
