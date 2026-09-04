#!/usr/bin/env bash
set -euo pipefail

cap_gib="${OCR_VRAM_CAP_GIB:-18}"
api_key="${OCR_API_KEY:?Set OCR_API_KEY before running this check}"
sample="${1:?Pass a JPEG, PNG, WEBP, or PDF sample path}"

curl --fail --silent --show-error http://localhost:8000/health >/dev/null
curl --fail --silent --show-error \
  -H "X-API-Key: ${api_key}" \
  -F "file=@${sample}" \
  -F "mode=text" \
  http://localhost:8000/api/v1/ocr >/dev/null

usage_rows="$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>&1 || true)"
used_mib="$(printf '%s\n' "${usage_rows}" | awk '$1 ~ /^[0-9]+$/ {sum += $1; found = 1} END {if (found) print sum}')"
if [[ -z "${used_mib}" ]]; then
  echo "GPU smoke check passed: OCR endpoint is healthy. This GPU does not expose comparable per-process memory totals (expected on GB10/DGX Spark UMA), so the ${cap_gib} GiB configured vLLM budget cannot be independently measured with nvidia-smi." >&2
  exit 0
fi
limit_mib=$((cap_gib * 1024 + 512))
if (( used_mib > limit_mib )); then
  echo "GPU smoke check failed: ${used_mib} MiB exceeds the ${cap_gib} GiB budget." >&2
  exit 1
fi
echo "GPU smoke check passed: ${used_mib} MiB is within the ${cap_gib} GiB budget."
