# Single-Model Full-Stack OCR

An internal, GPU-backed OCR application for printed JPEG, PNG, WEBP, and PDF documents. It combines a Next.js user interface with a FastAPI gateway and one OpenAI-compatible vision-language model. The service deliberately makes one model request per rendered page: there is no crop fan-out, consensus stage, second judge model, Redis, worker, database, or source-file persistence.

## Start

1. Install Docker Compose with NVIDIA Container Toolkit on a Linux GPU host.
2. Copy `.env.example` to `.env` and replace `OCR_API_KEY` with a long random secret.
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
OCR_MODEL_ID: datalab-to/surya-ocr-2 # alternative: dots-studio/dots.ocr
```

Change it to `dots-studio/dots.ocr`, then run `docker compose up -d --build inference`. The backend identifies the active model through `/v1/models` and applies the matching prompt/output profile. Model swapping is limited to OpenAI-compatible models supported by the selected engine.

vLLM is the default. To use SGLang with the same FastAPI/frontend contract:

```bash
docker compose -f docker-compose.yml -f docker-compose.sglang.yml up --build
```

The default inference entrypoint calculates a vLLM model-executor fraction from `OCR_VRAM_CAP_GIB=18`; it reserves room for CUDA/driver overhead. On discrete GPUs it uses reported framebuffer memory. On NVIDIA GB10/DGX Spark, `nvidia-smi` correctly reports `N/A` because CPU and GPU share unified memory; the launcher detects this and derives the fraction from `/proc/meminfo` (or from optional `OCR_UMA_TOTAL_MEMORY_GIB`).

GB10's CUDA memory view is not a reliable fraction of its shared 128 GiB RAM, so the default also sets an explicit `OCR_KV_CACHE_MIB=4096` and uses eager execution. vLLM skips automatic cache sizing when this setting is present, avoiding a negative-cache startup failure while bounding the persistent cache to 4 GiB. The 18 GiB setting remains a target, not a strict hardware-enforced total on UMA. For more throughput after a stable deployment, increase `OCR_KV_CACHE_MIB` and `OCR_MAX_NUM_SEQS` gradually and measure memory pressure. Run `scripts/gpu-smoke-check.sh path/to/sample.png` after startup to verify end-to-end OCR.

## API

`POST /api/v1/ocr` accepts `file`, `mode` (`text`, `nid_front`, `nid_back`, or `schema`), and optional `schema`. It blocks only until bounded-concurrency processing completes, then returns page results and model/timing metadata. `POST /api/v1/jobs` is a direct alias that returns a completed request ID. `GET /api/v1/jobs/{id}` reads a process-local result cache for one hour; it is not durable and clears on restart.

Compatibility routes from the reference service remain available: `/direct`, `/direct/base64`, `/v1/ocr/schema`, and `/v1/ocr/results/{id}`.

## Verification

Run backend unit tests in a Python environment with `pip install -r backend/requirements.txt` followed by `cd backend && pytest`. A GPU is not required for the CPU-only container smoke stack:

```bash
docker compose -f docker-compose.yml -f docker-compose.test.yml up --build
```

For quality evaluation, store redacted pairs of `<name>.expected.json` and `<name>.actual.json` under a fixture folder, then run `python backend/scripts/evaluate.py fixtures/`. It reports character error rate and exact field-match rate; do not claim a production accuracy target until this is measured against representative English and Bengali documents.

## Security and retention

FastAPI checks `X-API-Key` for every OCR/runtime route. The browser never receives that key. Input bytes are kept only in request memory, and results live in one process-local cache for one hour to support the direct job-result endpoint. Put the UI behind your internal network or a real reverse proxy before exposing it outside the organization.
