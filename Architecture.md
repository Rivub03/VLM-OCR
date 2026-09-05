# VLM-OCR Architecture and Design Record

## Purpose of this document

This document is the maintained context record for the VLM-OCR codebase. It is
intended for an engineer or another coding agent joining the project without
the preceding design conversations. It explains what is implemented, why it
was implemented that way, the constraints that must be preserved, and the
places to change when the system evolves.

The service is an internal, GPU-backed OCR application for printed documents.
Its primary special case is Bangladesh National ID (NID) cards, but it also
handles ordinary printed images and PDFs. The current production-oriented
default model is `dots-studio/dots.ocr` served through vLLM.

This is an architecture document, not a claim that every OCR result is
perfect. Accuracy must be measured against representative, redacted data
before a numerical production-quality claim is made.

## Current contract in one page

```text
Browser
  │  same-origin Next.js requests; browser never receives OCR_API_KEY
  ▼
Next.js frontend and server-side route proxies (frontend/)
  │  private Docker network + X-API-Key
  ▼
FastAPI gateway (backend/)
  │  validate upload → rasterize/normalise → optional NID enhancement
  │  bounded concurrent, one OpenAI-compatible request per page
  ▼
vLLM by default, or SGLang override (inference/)
  │  one selected VLM loaded on one GPU
  ▼
model transcription
  │
  ├── text/schema: return transcription and model/native fields as appropriate
  └── NID: derive strictly validated English fields from that transcription
             only; never make a second OCR call
```

The architecture deliberately has **one selected inference model** and **one
inference request per rendered page**. It does not use the old two-model OCR +
judge arrangement, OCR crop fan-out, voting/consensus, a Redis queue, a
database, or persistent upload storage.

## Problem history and the decisions it produced

### Earlier reference implementation

The supplied reference OCR microservice used a light Hunyuan OCR model and a
large Qwen judge model. It also performed multiple image views/crops and
consensus-like processing. That was a sensible response to weak Bengali OCR at
the time, but it had substantial drawbacks:

- Two loaded models required very large VRAM allocations; starts around 45 GiB
  were observed.
- More images and a judge request multiplied latency and made concurrency poor.
- A separate judge can produce a field that the original OCR text does not
  support, reducing auditability.
- The pipeline was difficult to operate as a small standalone service.

### Chosen direction

This repository was generated as a smaller full-stack replacement. The most
important decisions are:

| Decision | Why |
| --- | --- |
| One VLM per deployment | Reduces GPU memory, loading time, and operational complexity. |
| One OCR request per page | Bounds latency/cost and ensures structured NID values can be audited against a single transcription. |
| `dots-studio/dots.ocr` default | It has worked materially better than the initially selected Surya profile for the observed NID samples, especially for plain transcription/MRZ output. |
| FastAPI gateway | Keeps document validation, preprocessing, auth, postprocessing, job state, and model-server differences outside the model server. |
| Next.js frontend with server-side proxy | Provides a usable scanner UI without exposing the backend API key to the browser. |
| Strict local NID extraction | English NID fields should be deterministic consequences of OCR text, not VLM guesses. |
| Bounded in-process asynchronous jobs | The UI can poll and cancel slow work without persisting sensitive source files or operating Redis. |
| vLLM default, SGLang override | vLLM is the default OpenAI-compatible server; SGLang can be substituted without changing the app contract. |

TensorRT-LLM and NVIDIA Triton were considered in the broader requirement, but
are intentionally not supplied in this release. Adding either should be a
separate, benchmarked integration rather than a configuration name that does
not have a tested OpenAI-compatible contract.

## Repository map

```text
.
├── docker-compose.yml             # Default vLLM deployment and all service settings
├── docker-compose.sglang.yml      # Override that switches only inference serving
├── docker-compose.test.yml        # CPU-only mock inference override
├── .env.example                   # Secret/configuration template; no real secrets
├── README.md                      # Concise operator instructions
├── Architecture.md                # This deeper design record
├── backend/
│   ├── app/
│   │   ├── main.py                # FastAPI app, routes, auth, upload preparation
│   │   ├── service.py             # One-page model request and result construction
│   │   ├── profiles.py            # Model detection and model-specific prompts
│   │   ├── preprocess.py          # Safe image/PDF conversion and NID enhancement
│   │   ├── postprocess.py         # OCR response parsing and strict NID extraction
│   │   ├── jobs.py                # Ephemeral async job registry/cancellation
│   │   ├── cache.py               # One-hour process-local result cache
│   │   ├── schemas.py             # API response models
│   │   └── config.py              # Environment-backed settings
│   ├── scripts/
│   │   ├── evaluate.py            # Fixture CER and field exact-match helper
│   │   └── evaluate_nid.py        # External NID benchmark runner
│   └── tests/                     # Unit and mocked-inference regression coverage
├── frontend/
│   └── app/
│       ├── page.tsx               # Scanner/polling/results UI
│       └── api/                   # Server-side proxy routes to FastAPI
├── inference/
│   ├── entrypoint.sh              # vLLM memory-aware launcher
│   ├── sglang-entrypoint.sh       # SGLang memory-aware launcher
│   └── mock/                      # CPU-only OpenAI-compatible test server
└── scripts/gpu-smoke-check.sh     # GPU allocation and OCR smoke check
```

The `work/` directory is local dependency/tooling material, not application
architecture. Do not treat it as production source.

## Deployment topology

The normal Compose deployment has three services and a named model cache:

| Service | Image/source | Responsibilities | Exposed port |
| --- | --- | --- | --- |
| `inference` | `inference/Dockerfile`, vLLM image | Download/load one model and expose OpenAI-compatible `/v1/models` and `/v1/chat/completions`. | Internal only |
| `backend` | `backend/Dockerfile`, Python 3.12/FastAPI | Authentication, file validation, preprocessing, concurrency, postprocessing, jobs, compatibility routes. | `8000` |
| `frontend` | `frontend/Dockerfile`, Next.js | Browser UI and same-origin proxy routes. | `3000` |
| `hf-cache` | Docker named volume | Persistent Hugging Face model/cache data across inference container recreations. | n/a |

The backend waits for inference health before starting; the frontend waits for
backend health. The inference healthcheck has a long initial grace period
because a first Dots download and model startup can take minutes. Docker
networking is used throughout: there is no host network mode or hard-coded
external host IP.

### Request paths

There are two supported browser-facing patterns:

1. **Synchronous OCR:** browser → `frontend/app/api/ocr/route.ts` →
   `POST /api/v1/ocr` → completed result. This is useful for simple callers.
2. **Job OCR (the scanner UI’s normal path):** browser →
   `POST /api/jobs` → `POST /api/v1/jobs`, then the UI polls
   `GET /api/jobs/{id}`. A user can send `DELETE /api/jobs/{id}` to cancel.

The proxy obtains `BACKEND_URL` and `OCR_API_KEY` only from the frontend
container environment. It forwards the multipart body and adds `X-API-Key`.
This means JavaScript running in the browser never sees the secret.

Programmatic internal callers can call FastAPI directly with `X-API-Key`.
CORS is restricted to `FRONTEND_ORIGIN`, rather than being open to arbitrary
web origins.

## Backend API contract

### Canonical endpoints

| Route | Purpose |
| --- | --- |
| `GET /health` | Lightweight backend liveness result. |
| `GET /api/v1/runtime` | Authenticated deployed-model/profile/limit information. |
| `POST /api/v1/ocr` | Authenticated synchronous OCR. Multipart `file`, `mode`, optional `schema`. |
| `POST /api/v1/jobs` | Authenticated asynchronous job submission. |
| `GET /api/v1/jobs/{job_id}` | Authenticated job status/result retrieval. |
| `DELETE /api/v1/jobs/{job_id}` | Authenticated in-flight cancellation. |

Accepted `mode` values are `text`, `nid_front`, `nid_back`, and `schema`.
The canonical result includes a completed status, one `PageResult` per input
page, and metadata containing request ID, served model ID, inferred serving
engine, page count, and elapsed milliseconds.

### Compatibility endpoints

The reference service’s direct/schema/result patterns are kept at `/direct`,
`/direct/base64`, `/v1/ocr/schema`, and `/v1/ocr/results/{job_id}`. They are
adapters, not a second implementation. Any new behavior should be implemented
in the canonical path first, then exposed through an adapter only if needed.

## Ingestion and preprocessing

`backend/app/main.py` owns `prepare_upload`; `backend/app/preprocess.py` owns
byte-level document conversion.

### Input policy

- Supported images: JPEG, PNG, and WEBP.
- Supported document: PDF.
- Default upload limit: `MAX_UPLOAD_MIB=25`.
- Default PDF page limit: `MAX_PDF_PAGES=20`.
- Default maximum raster dimension: `MAX_PAGE_DIMENSION=2048`.
- Empty, corrupt, unsupported, over-size, over-page-limit, or malformed schema
  uploads are rejected with a useful 4xx response.

Images are opened with Pillow, EXIF orientation is applied, images are converted
to RGB, bounded to the maximum dimension, and encoded as PNG. PDF pages are
rasterized one at a time through PyMuPDF at a 2× matrix and then run through the
same normaliser. Thus inference always sees a PNG data URL, regardless of the
original accepted type.

### NID-only local enhancement

NID modes add a single local transformation to improve small camera/scan card
images before their one allowed OCR call:

1. Apply EXIF orientation and RGB normalisation.
2. Conservatively upscale until the shortest side reaches 900 pixels (subject
   to a 4× ceiling and global dimension cap).
3. Add a 20-pixel white border.
4. Convert to LAB and run CLAHE on L only, with clip limit 2.0 and an 8×8 tile
   grid.
5. Apply mild unsharp masking (amount 0.20).

The relevant Compose settings are `NID_PREPROCESS_ENABLED`,
`NID_MIN_SHORT_EDGE`, `NID_MAX_UPSCALE`, `NID_BORDER_PX`,
`NID_CLAHE_CLIP_LIMIT`, `NID_CLAHE_TILE_GRID_SIZE`, and
`NID_UNSHARP_AMOUNT`.

This is deliberately *not* perspective correction, crop fan-out, a second
contrast view, language-specific image synthesis, or an automatic retry. If
OpenCV enhancement fails, the normalised original is still sent once and a
warning is attached. General text/PDF modes do not receive this enhancement,
which avoids silently changing behavior for ordinary documents.

## Model selection and prompt adapters

### Plug-and-play model switch

The normal model switch is one Compose line:

```yaml
OCR_MODEL_ID: dots-studio/dots.ocr
```

Change that ID and restart/rebuild only the `inference` service. The backend
does not assume the Compose value. At each OCR operation it reads the served
ID from inference `/v1/models` and selects a profile in
`backend/app/profiles.py`.

Supported profiles today are:

| Profile | Recognised ID prefix | Native output expectation | Prompt strategy |
| --- | --- | --- | --- |
| Dots | `dots-studio/dots.ocr`, `rednote-hilab/dots.ocr` | text/JSON | OCR transcription for text and NID; controlled JSON prompt for schema mode. |
| Surya OCR 2 | `datalab-to/surya-ocr-2` | HTML/layout-oriented | Full-page HTML prompt for documents; block OCR contract for NID cards. |
| Chandra OCR 2 | `datalab-to/chandra-ocr-2`, `datalab-to/chandra` | HTML | Native HTML OCR prompt. |
| Generic | anything else | text | Conservative generic printed-document OCR prompt. |

The profile adapter is crucial. Sending a prompt designed for one OCR model to
another can lead to layout JSON, malformed structured output, or empty text.
Add a new model by defining a profile, native output type, tested prompt
contract, token limits, and tests in `test_profiles.py`; do not merely add its
name to a Compose comment.

### Dots behavior and NID prompts

The observed Dots model performs best on NIDs when asked to transcribe rather
than to generate a JSON object. For `text`, it uses the documented OCR task:

```text
Extract the text content from this image.
```

For NID modes, the prompt retains that task sentence and adds two narrowly
scoped constraints: transcribe all visible printed text even if a labelled field
is blank, and do not invent a value. This was added after samples with a blank
`Blood Group:` label sometimes resulted in an empty model response. It changes
only NID prompting; generic Dots document behavior remains unchanged.

No prompt should ask the model to infer Bengali data or manufacture a missing
English field. The model output is evidence, not an authoritative structured
record.

## Inference serving and memory design

### vLLM default

`inference/entrypoint.sh` launches vLLM with:

- `--trust-remote-code`, needed by the selected OCR model integrations;
- OpenAI-compatible serving on port 8000;
- a served model name identical to `OCR_MODEL_ID`;
- one image per prompt;
- bounded sequence count and model length;
- eager execution by default; and
- an explicit persistent KV-cache allocation.

Default Compose values are intentionally conservative:

| Setting | Default | Reason |
| --- | ---: | --- |
| `OCR_VRAM_CAP_GIB` | 18 | Target budget for inference server allocation. |
| `OCR_KV_CACHE_MIB` | 4096 | Avoid vLLM auto-sizing an excessive or invalid cache. |
| `OCR_MAX_MODEL_LEN` | 8192 | Enough document/OCR context while bounded. |
| `OCR_MAX_NUM_SEQS` | 2 | Lets vLLM batch a small number of requests without flooding a small OCR model. |
| Backend `MAX_INFERENCE_CONCURRENCY` | 2 | Matches the inference bound and applies backpressure before requests overload vLLM. |

The backend semaphore surrounds the actual inference HTTP call. CPU-side
validation and preprocessing happen outside that semaphore; the expensive GPU
work is what is bounded. vLLM continuous batching can therefore improve modest
parallel throughput without accepting an unbounded pile of model requests.

### NVIDIA GB10 / DGX Spark unified memory

GB10/DGX Spark can report `N/A` for `nvidia-smi --query-gpu=memory.total` even
when CUDA is working, because the GPU shares unified system memory. The launcher
does not treat this as no GPU. It first verifies that `nvidia-smi` can see a
GPU, then derives total memory from either `OCR_UMA_TOTAL_MEMORY_GIB` or
`/proc/meminfo` when dedicated VRAM is unreported. It calculates a conservative
fraction for vLLM/SGLang and still uses the explicit KV-cache cap.

`OCR_VRAM_CAP_GIB` is a useful allocation target on discrete GPUs. On UMA it is
not a hard hardware-enforced partition of the 128 GiB shared RAM. Operators
must monitor actual system/GPU pressure, start with the supplied values, and
increase cache/sequences only after a smoke test.

### SGLang override

Run Compose with `docker-compose.sglang.yml` as an additional file. It replaces
only the inference image/build while preserving the service name, model ID,
backend URL, and frontend contract. `sglang-entrypoint.sh` applies equivalent
GPU/UMA detection and uses static memory fraction plus
`--max-running-requests`.

Any future Triton or TensorRT deployment must preserve `/v1/models` and
`/v1/chat/completions`, or `OCRService` must receive a consciously designed
adapter. It is not safe to assume a different server accepts the same VLM
multimodal request JSON.

## OCR result processing

`backend/app/service.py` constructs a data URL from one rendered page, obtains
the active model, builds a profile-specific payload, and sends exactly one
`POST /v1/chat/completions`. It maps upstream timeout/4xx/5xx/invalid-result
conditions into controlled API errors.

`backend/app/postprocess.py` then:

1. removes Markdown fences;
2. converts constrained Surya/Chandra HTML to readable text;
3. stops repeated decoder loops without requesting another inference;
4. separates JSON transcription and fields when a model legitimately returns
   an object; and
5. detects Surya-like layout metadata arrays and never treats them as OCR text.

For ordinary schema mode, existing structured output may be used or a simple
deterministic label extraction is applied where appropriate. For **NID mode,
model-provided fields are always discarded**. Only raw transcription feeds the
NID parser.

## NID extraction contract

### Supported fields

NID fields are intentionally limited to English/ASCII-oriented information
that can be validated conservatively.

| NID side | Returned keys, always present | Validation policy |
| --- | --- | --- |
| Front | `name`, `dob`, `nid_no` | English name following `Name`; labelled date; labelled 10/13/17 digit NID number. |
| Back | `blood_group`, `place_of_birth`, `issue_date`, `mrz_line1`, `mrz_line2`, `mrz_line3` | Strict labelled blood type; labelled English place/date; final three valid uppercase MRZ candidates. |

Every key is returned for the selected NID side. Its value is either validated
or `null`; a missing or invalid value cannot remove unrelated fields and cannot
turn a completed OCR job into a failed job.

### Why Bengali structured fields are excluded

Raw Bengali transcription is retained in `text`/`markdown` and remains useful
for audit and future work. It is not used for NID structured fields because the
current accuracy target is reliable printed English values. Extending the
schema to Bengali must be backed by a separately measured benchmark and should
not lower the reliability of the current English-only contract.

### Label-aware extraction details

The parser normalises Bengali digits to ASCII digits before value validation.
It does not perform broad, free-floating regex extraction for NID values.

- Names are accepted from the `Name` line or the following line only when the
  candidate is an English-name-shaped value.
- Dates must be associated with their relevant label and match supported date
  forms.
- NID numbers must be associated with an NID label and have a supported length.
- Blood groups must match `A`, `B`, `AB`, or `O` plus `+`/`-`; a blank label is
  a valid absence, not a failed extraction.
- On the NID back, OCR can place `Blood Group:`, `Place of Birth:`, and
  `Issue Date:` on one line because those labels are visually horizontal. The
  parser recognizes inline known labels and slices one labelled candidate at
  the next known label. This preserves `place_of_birth` and `issue_date` when
  `Blood Group:` is intentionally blank.
- MRZ text is accepted only from the last three lines matching uppercase
  `[A-Z0-9<]` with length 20–44. Ambiguous characters are not auto-corrected.

The parser emits one warning per null field. If the model response has no text
at all, it additionally says that no transcription was available. That is an
observable model-output condition, not a reason to invent values, issue a
second inference request, or silently declare a successful extraction.

## Jobs, cancellation, retention, and privacy

`JobManager` is a deliberately small in-process registry. A job moves through
`queued`, `running`, `completed`, `failed`, or `cancelled`. It is not a durable
queue and does not retain uploaded source files.

Cancellation calls `asyncio.Task.cancel()`. That cancellation propagates into
the in-progress request coroutine, closing the backend request and releasing
the backend concurrency slot. The model server may still finish its already
accepted computation; application-level cancellation cannot reliably undo GPU
work that is already executing in vLLM.

`ResultCache` retains completed result metadata/output in process memory for
`RESULT_TTL_SECONDS` (default one hour), supporting status/result retrieval.
It is cleared on a backend restart. This design avoids storage of identity-card
source bytes at rest, but it is not a substitute for a formal data-retention or
compliance policy.

## Security model

- `OCR_API_KEY` is required by FastAPI OCR/runtime/job endpoints.
- The frontend owns the secret server-side and injects it when proxying.
- `HF_TOKEN` is provided only to `inference`, allowing authenticated model
  downloads. It must never be put in Compose source, browser code, logs, or
  Git.
- `.env` and `.env.*` are ignored; `.env.example` documents keys but holds no
  credentials.
- The supplied Hugging Face token was exposed in prior conversation context and
  should be revoked/replaced. Use a new, minimally scoped read token in the
  DGX deployment’s local `.env`.
- CORS only permits the configured frontend origin. Place the whole system
  behind an internal network/reverse proxy before any external exposure.

Required local deployment variables are at minimum:

```dotenv
OCR_API_KEY=replace-with-a-long-random-secret
FRONTEND_ORIGIN=http://localhost:3000
HF_TOKEN=replace-with-a-new-read-token
```

## Quality strategy and tests

### Automated test coverage

The backend tests cover upload/rasterisation limits, NID preprocessing and its
fallback, Bengali numeral conversion, response parsing, model-profile prompt
selection, field validation, job transitions/cancellation, expiry behavior,
and a mocked OpenAI-compatible inference call. A key regression test asserts
that NID field extraction makes only one model request and derives values from
the textual transcription rather than trusting fabricated model JSON fields.

The blank-blood-group regression test represents the realistic one-line layout:

```text
Blood Group:  Place of Birth: CHANDPUR  Issue Date: 16 Jan 2018
```

It verifies that `blood_group` is null but the other two labelled values and
MRZ lines survive.

Use the production dependency container to run backend tests without a GPU:

```bash
docker run --rm -e PYTHONPATH=/app -e PYTHONDONTWRITEBYTECODE=1 \
  --mount type=bind,source="$PWD/backend/app",target=/app/app,readonly \
  --mount type=bind,source="$PWD/backend/tests",target=/app/tests,readonly \
  vlm-ocr-backend pytest -q tests
```

`docker-compose.test.yml` replaces inference with a CPU-only mock server. It
is for API/Compose smoke coverage, not OCR quality evaluation.

### Benchmarking NID accuracy

External benchmark images and ground truth must remain outside this repository.
`backend/scripts/evaluate_nid.py` accepts the benchmark root, side, and split;
it calls the real API and records field-level exact match, null rate,
false-positive counts, MRZ results, and aggregate supported-field accuracy.

Recommended protocol:

1. Keep the held-out `test` split untouched.
2. Tune preprocessing/prompt/parser thresholds only on `train` and `valid`.
3. Require at least 99% validation exact match for supported non-empty English
   fields and zero false positives for visibly blank fields before making a
   quality claim.
4. Run the held-out test exactly once for the final report.
5. Report English/Bengali limitations separately; Bengali raw OCR is not proof
   that structured Bengali fields are production-ready.

The generic `backend/scripts/evaluate.py` compares manually paired expected and
actual JSON fixtures, producing character error rate and exact field match.

## Operational playbook

### First deployment

1. Install Docker Compose and NVIDIA Container Toolkit on the Linux GPU host.
2. Verify GPU pass-through with a CUDA container and `nvidia-smi`.
3. Copy `.env.example` to `.env`; set a strong API key and a new Hugging Face
   read token.
4. Leave Dots as the model initially:

   ```yaml
   OCR_MODEL_ID: dots-studio/dots.ocr
   ```

5. Start with `docker compose up --build`.
6. Wait through first model download/warmup; then check `/health`,
   `/api/v1/runtime` with the API key, and run `scripts/gpu-smoke-check.sh`.

### Changing model or serving engine

To evaluate another supported model, edit only `OCR_MODEL_ID`, rebuild the
inference service, and preserve the rest of the stack. The backend will detect
the active server-reported ID. Before declaring a swap successful, run the same
representative benchmark; different prompts and output formats do not imply
equivalent Bengali/NID accuracy.

To use SGLang, append the override Compose file. Do not mix vLLM and SGLang
settings blindly; retain the same small concurrency/memory posture until a
load test confirms safe headroom.

### Failure triage

| Symptom | First checks |
| --- | --- |
| Inference exits before model load | Verify GPU access inside Docker, inspect `nvidia-smi`, then inspect memory/UMA launcher logs. |
| GB10 says memory `N/A` | Expected on unified memory; ensure current launchers derive memory from `/proc/meminfo` or set `OCR_UMA_TOTAL_MEMORY_GIB`. |
| Backend 422 from upstream | Read the clipped upstream error; check profile/model compatibility and image limits. |
| Long first start | Model cache/download/custom code is warming; healthcheck allows up to 20 minutes. |
| Slow concurrent jobs | Keep `MAX_INFERENCE_CONCURRENCY` aligned with `OCR_MAX_NUM_SEQS`; increase only after monitoring memory and queueing. |
| NID fields null but text exists | Treat it as validation failure for that field, inspect raw transcription, and improve only through benchmarked changes. |
| All NID fields null and text empty | The model supplied no transcription. Capture model/inference logs and preserve the output warning; do not fabricate values. |
| Blank Blood Group hides other back fields | Ensure the current inline-label parser/prompt version is deployed; a blank blood group must remain null while other labels extract. |

## Explicit non-goals and invariants

The following are intentional constraints. A future change that breaks one
must include a written design review and measured evidence that the tradeoff is
worth it.

1. **No judge model.** Do not reintroduce Qwen or another verifier as an
   implicit second model request merely to fill a field.
2. **One OCR call per page.** No crop fan-out, retried alternate enhancement,
   ensemble, consensus, or second-pass schema request in normal production
   flow.
3. **No guessing identity data.** NID fields need source-transcription
   evidence and validation; unknown is `null`.
4. **No required field assumption.** A card may intentionally omit any field,
   especially blood group. Absence must not hide valid independent fields.
5. **No source-file persistence.** Keep upload bytes request-scoped unless a
   separately approved security/compliance design changes this.
6. **No secrets in Git.** A real `HF_TOKEN`, API key, or private endpoint is
   never documentation/example content.
7. **No unmeasured accuracy claim.** A visually pleasing sample is not a
   benchmark.

## Safe extension guide

### Add a model

1. Confirm it can run under the selected inference engine and supports the
   required multimodal API format.
2. Add a `ModelProfile` ID prefix, output type, token budget, and documented
   prompt in `profiles.py`.
3. Add unit tests proving exact profile selection and payload shape.
4. Test document, NID-front, NID-back, blank-field, PDF, and concurrent inputs.
5. Run the external benchmark before changing the default model.

### Add an NID field

1. Decide whether it is genuinely reliable enough for a structured contract.
2. Add it to the fixed front/back schema, UI preset, and evaluator map.
3. Derive it only from an explicit label/transcription evidence.
4. Define a conservative validator and null warning.
5. Add positive, blank, malformed, and adversarial regression tests.
6. Re-run benchmark gates. Never add Bengali structured fields just because raw
   Bengali text appears plausible in a few screenshots.

### Change preprocessing

Preprocessing changes can improve one scan while damaging another. Keep changes
NID-scoped unless there is independent document evidence; preserve a normalised
fallback and warning; never create an extra inference view. Tune numerical
values only through train/validation measurements, then reserve the held-out
split for final reporting.

### Replace ephemeral jobs with durable queueing

Only do this when a requirement calls for durable, cross-restart, multi-worker
processing. It will require a data retention/security design for file storage,
idempotency, cancellation semantics across workers, explicit queue backpressure,
and observability. Do not add Redis merely because an earlier plan mentioned a
queue; the current direct bounded-concurrency architecture is intentional.

## Change chronology

The recent Git history reflects the development sequence and is useful context
when debugging an older deployment:

1. The initial service replaced the heavy OCR + judge architecture with a
   single-model FastAPI/Next.js stack.
2. vLLM startup was hardened for GPU access and then GB10 unified memory, where
   `nvidia-smi` reports dedicated memory as `N/A`.
3. Explicit KV cache, eager mode, conservative sequence/model limits, and a
   longer Dots warmup period were added to avoid memory/startup failures.
4. Model profile adapters were added for Surya, Chandra, and Dots; the active
   model is discovered rather than hard-coded in backend code.
5. Surya’s layout-only output on NID cards led to the Dots default after
   observed document/NID trials.
6. NID-specific CLAHE/upscale/border/sharpening and strict English-only field
   extraction replaced VLM JSON extraction for NID modes.
7. The most recent NID-back hardening treats blank blood groups as legitimate,
   protects adjacent same-row labelled fields, and asks Dots to transcribe the
   rest of a card even if a label has no value.

When reviewing an incident, compare the deployed commit with this chronology.
Older image caches or remote clones may not contain the current GB10, Dots,
NID-preprocessing, or blank-field behavior.

## Maintainer checklist

Before merging/deploying OCR changes:

- [ ] Run backend tests.
- [ ] Validate `docker compose config` with non-secret placeholder values.
- [ ] Build backend/frontend images when dependencies or UI changed.
- [ ] Smoke test the selected GPU engine with an ordinary document and both NID
      sides, including a blank blood-group NID.
- [ ] Confirm only one `/v1/chat/completions` call is made per page.
- [ ] Confirm NID output contains only the fixed side-specific keys and that
      unknown values are `null` with warnings.
- [ ] Confirm raw transcription is retained for audit.
- [ ] Keep `.env`, caches, benchmark source documents, and credentials out of
      Git.
- [ ] Evaluate on train/validation before a held-out test run.

