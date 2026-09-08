# Single-Model Full-Stack OCR

An internal, GPU-backed OCR application for printed JPEG, PNG, WEBP, and PDF documents. It combines a Next.js user interface with a FastAPI gateway and one OpenAI-compatible vision-language model. The service deliberately makes one model request per rendered page: there is no crop fan-out, consensus stage, second judge model, Redis, worker, database, or source-file persistence.

Only one model is loaded on the GPU (~8 GiB resident). NID fields that fail local validation are re-read by a small ONNX recogniser on the **CPU** — no VRAM, and it may only return a value that passes the same validator the first reading failed. Bangladesh NID backs carry an ICAO TD1 machine-readable zone whose check digits are verified, and repaired when they single out one answer; see `Architecture.md`.

## Start

1. Install Docker Compose with NVIDIA Container Toolkit on a Linux GPU host.
2. Copy `.env.example` to `.env`, replace `OCR_API_KEY` with a long random secret, and set `HF_TOKEN` to a newly issued Hugging Face read token. `.env` is ignored by Git; never put the token in Compose or source code.
3. Run `docker compose up --build`.
4. Open `http://localhost:3000`. The frontend proxies to FastAPI using the server-side key; direct API callers must supply `X-API-Key`.

The first start downloads the model and can take several minutes. The API is on port 8000 and the frontend is on port 3000.

Before starting the stack, verify that Docker itself can reach the GPU (not just the host):

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

If this fails, install/configure NVIDIA Container Toolkit on the host and restart Docker. The `inference` service explicitly requests all GPUs through Compose; it does not attempt CPU inference.

## Change model or serving engine

The one model line in `docker-compose.yml` is:

```yaml
OCR_MODEL_ID: dots-studio/dots.ocr # alternatives: datalab-to/surya-ocr-2, datalab-to/chandra-ocr-2
```

Change it to `dots-studio/dots.ocr` or `datalab-to/chandra-ocr-2`, then run `docker compose up -d --build inference`. The backend identifies the active model through `/v1/models` and applies the matching prompt/output profile: Surya uses its native full-page HTML contract, Chandra uses native HTML OCR, and dots uses text or JSON extraction prompts. Model swapping is limited to OpenAI-compatible models supported by the selected engine.

vLLM is the default. To use SGLang with the same FastAPI/frontend contract:

```bash
docker compose -f docker-compose.yml -f docker-compose.sglang.yml up --build
```

The default inference entrypoint calculates a vLLM model-executor fraction from `OCR_VRAM_CAP_GIB=18`; it reserves room for CUDA/driver overhead. On discrete GPUs it uses reported framebuffer memory. On NVIDIA GB10/DGX Spark, `nvidia-smi` correctly reports `N/A` because CPU and GPU share unified memory; the launcher detects this and derives the fraction from `/proc/meminfo` (or from optional `OCR_UMA_TOTAL_MEMORY_GIB`).

Note that `inference/entrypoint.sh` is baked into the image. Changing a launcher flag needs `docker compose up -d --build inference`; a plain `up -d` keeps the old launcher while still applying environment-variable changes.

GB10's CUDA memory view is not a reliable fraction of its shared 128 GiB RAM, so the default also sets an explicit `OCR_KV_CACHE_MIB=4096`. vLLM skips automatic cache sizing when this setting is present, avoiding a negative-cache startup failure while bounding the persistent cache to 4 GiB. The 18 GiB setting remains a target, not a strict hardware-enforced total on UMA. For more throughput after a stable deployment, increase `OCR_KV_CACHE_MIB` and `OCR_MAX_NUM_SEQS` gradually and measure memory pressure. Run `scripts/gpu-smoke-check.sh path/to/sample.png` after startup to verify end-to-end OCR.

## API

`POST /api/v1/ocr` accepts `file`, `mode` (`text`, `nid_front`, `nid_back`, or `schema`), and optional `schema`. It blocks only until bounded-concurrency processing completes, then returns page results and model/timing metadata. The browser uses `POST /api/v1/jobs`, which immediately returns a job ID and status. Poll `GET /api/v1/jobs/{id}` for `queued`, `running`, `completed`, `failed`, or `cancelled`; send `DELETE /api/v1/jobs/{id}` to cancel an in-flight request. Cancellation closes the backend request to the inference server and frees the application concurrency slot; no source file is retained.

For NID mode, camera images receive one configurable local enhancement pass — card rectification, illumination flattening, deskew, upscaling, a white border, LAB CLAHE, and mild sharpening — before the one permitted OCR request. Each stage falls back to a pass-through with a warning rather than failing. General documents and PDFs retain standard normalization and are rasterized at `MAX_PDF_DPI`.

NID structured output is intentionally limited to English fields: front `name`, `dob`, and `nid_no`; back `blood_group`, `place_of_birth`, `issue_date`, and three MRZ lines. Every field is derived from the returned transcription and is `null` with a warning when it cannot be validated; the service does not infer missing values. NID backs carry an ICAO TD1 MRZ whose check digits are verified and, where they single out one answer, repaired.

Each page result also carries:

- `layout` — blocks with `category` (`Title`, `Table`, `Picture`, `Formula`, …) and `text`, so tables and figures survive as structure rather than flattened prose. `markdown` keeps the model's HTML tables; `text` flattens them for reading.
- `field_confidence` and `field_evidence` — per field, how well supported the value is and which stage produced it (`text`, `layout`, `mrz:valid`, `rapidocr`, …), so results can be triaged rather than trusted uniformly.
- `finish_reason` — `length` means the model hit its output limit and the end of the page is missing.

All three are additive; existing callers are unaffected.

Compatibility routes from the reference service remain available: `/direct`, `/direct/base64`, `/v1/ocr/schema`, and `/v1/ocr/results/{id}`.

## Testing

There are two layers, and they answer different questions. **Unit tests** tell you whether the contract still holds. **The benchmark** tells you whether accuracy actually improved. A change to preprocessing, prompting, or the parser is not finished until both have run — several changes during development passed every unit test and still *lost* accuracy on real cards.

`Architecture.md` has the full procedure, including the A/B workflow, how to read the pipeline-health numbers, and a tuning record of what did and did not work.

### 1. Unit tests (no GPU required)

Runs in the production dependency container, with the source mounted read-only:

```bash
docker compose build backend        # only if dependencies or app code changed

docker run --rm -e PYTHONPATH=/app -e PYTHONDONTWRITEBYTECODE=1 -e OCR_API_KEY=test \
  --mount type=bind,source="$PWD/backend/app",target=/app/app,readonly \
  --mount type=bind,source="$PWD/backend/tests",target=/app/tests,readonly \
  vlm-ocr-backend pytest -q tests
```

`OCR_API_KEY` is required because settings have no default for it. Narrow a run with a path or `-k`, e.g. `pytest -q tests/test_mrz.py`. To use a local environment instead: `pip install -r backend/requirements.txt`, then `cd backend && OCR_API_KEY=test pytest -q tests`.

A GPU is not required for the CPU-only container smoke stack either:

```bash
HF_TOKEN=not-used OCR_API_KEY=local-test-key \
  docker compose -f docker-compose.yml -f docker-compose.test.yml up --build
```

### 2. Accuracy benchmark (needs the real stack running)

Keep benchmark images and ground truth **outside this repository** — they are identity documents. The evaluator calls the real HTTP API, so it measures the whole pipeline. It expects `<root>/data/ground_truth/<type>_<split>.json` alongside `<root>/data/images/<type>/<split>/`.

Bring the stack up first (`docker compose up -d`, then `docker compose ps` to confirm `inference` is healthy — first start can take several minutes).

```bash
docker run --rm --network host \
  --mount type=bind,source="$PWD/backend/scripts",target=/app/scripts,readonly \
  --mount type=bind,source=/path/to/ocr-benchmark,target=/bench,readonly \
  -v "$PWD/benchmark-results":/out \
  -e OCR_API_KEY="$(grep -E '^OCR_API_KEY=' .env | cut -d= -f2-)" \
  vlm-ocr-backend python /app/scripts/evaluate_nid.py \
    --benchmark-root /bench --type nid_front --split train \
    --concurrency 2 --output /out/front_train.json
```

Or directly, if you have the dependencies locally:

```bash
OCR_API_KEY=... python backend/scripts/evaluate_nid.py \
  --benchmark-root /path/to/ocr-benchmark --type nid_front --split train \
  --concurrency 2 --output benchmark-results/front_train.json
```

Add `--limit 40` while iterating; a full split takes roughly twenty minutes against a few for a subset. Keep `--concurrency` at or below `MAX_INFERENCE_CONCURRENCY` (default 2) — beyond that, requests just queue inside vLLM. Keep `benchmark-results/` out of Git; the rows contain transcribed identity data.

The console output reports per-field and overall exact match, false positives on blank fields, and pipeline-health counters (`truncated pages`, `request errors`). The JSON report additionally records every comparison, per-field confidence, which pipeline stage produced each value, and a breakdown of *why* each failure failed.

Do not claim a production accuracy target until validation reaches at least 99% exact match for supported non-empty English fields, blank fields produce zero false positives, and one held-out test run confirms the result. Tune only on `train`/`valid`; run `test` exactly once.

## Security and retention

FastAPI checks `X-API-Key` for every OCR/runtime route. The browser never receives that key. Input bytes are kept only in request memory, and results live in one process-local cache for one hour to support the direct job-result endpoint. Put the UI behind your internal network or a real reverse proxy before exposing it outside the organization.
