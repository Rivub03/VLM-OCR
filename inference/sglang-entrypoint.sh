#!/usr/bin/env bash
set -euo pipefail

: "${OCR_MODEL_ID:?OCR_MODEL_ID must be set}"
: "${OCR_VRAM_CAP_GIB:=18}"
: "${OCR_MAX_NUM_SEQS:=4}"
# nvidia-smi may report N/A during container startup; only use numeric values.
total_mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '{ for (i = 1; i <= NF; i++) if ($i ~ /^[0-9]+$/) { print $i; exit } }')"
if [[ -z "${total_mib}" ]]; then
  echo "nvidia-smi did not return a numeric GPU memory.total value." >&2
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
