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

used_mib="$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '{sum += $1} END {print sum+0}')"
limit_mib=$((cap_gib * 1024 + 512))
if (( used_mib > limit_mib )); then
  echo "GPU smoke check failed: ${used_mib} MiB exceeds the ${cap_gib} GiB budget." >&2
  exit 1
fi
echo "GPU smoke check passed: ${used_mib} MiB is within the ${cap_gib} GiB budget."

