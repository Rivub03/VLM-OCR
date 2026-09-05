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
  │  (rectify → flatten illumination → deskew → resize → CLAHE → border)
  │  → size every page against the model's context budget
  │  bounded concurrent, one OpenAI-compatible request per page
  ▼
vLLM by default, or SGLang override (inference/)
  │  one selected VLM loaded on one GPU (~8 GiB resident)
  ▼
model transcription (layout blocks: bbox + category + text)
  │
  ├── text/schema: return transcription, layout, and fields as appropriate
  └── NID: derive strictly validated English fields from that transcription
  │        only, matching labels to values geometrically
  ▼
CPU reconciliation (backend/app/verify.py) — only for fields that failed
  │  validation. Re-reads crops with a small ONNX engine. No GPU, no VLM.
  ▼
result (fields + per-field confidence + evidence)
```

The architecture deliberately has **one selected inference model** and **one
inference request per rendered page**. It does not use the old two-model OCR +
judge arrangement, OCR crop fan-out, voting/consensus, a Redis queue, a
database, or persistent upload storage.

The reconciliation step is not a judge and not a second VLM: it is a ~15 MB
ONNX recogniser on the CPU, invoked only when a field failed its own validator,
and it may only supply a value that passes that same validator.

It is further restricted to fields whose validator can genuinely *confirm* a
reading — the MRZ check digits, an NID length of 10/13/17, a parseable date, a
literal blood group. `name` and `place_of_birth` are excluded, because their
validator only asks "does this look like letters" and cannot tell a good reading
from a bad one. See `RECONCILABLE_FIELDS` in `verify.py`.

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
│   │   ├── preprocess.py          # Image/PDF conversion, NID enhancement, token budget
│   │   ├── postprocess.py         # OCR response parsing and layout recovery
│   │   ├── nid.py                 # Spatial + textual NID field extraction
│   │   ├── mrz.py                 # TD1 check digits and constrained repair
│   │   ├── verify.py              # CPU-only reconciliation of failed fields
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

### Context-budget sizing (all modes)

Image tokens are `ceil(width / 28) * ceil(height / 28)`: dots.ocr uses a
Qwen2-VL processor with patch size 14 and a 2×2 spatial merge, so one token
covers a 28-pixel square. The model's own `preprocessor_config.json` permits
`max_pixels: 11289600`, about 14,400 tokens — far past any sane served context.

`preprocess.fit_to_token_budget` scales every rendered page down until its token
cost fits `MAX_MODEL_LEN - TOKEN_BUDGET_MARGIN`, and `entrypoint.sh` passes the
same figure to vLLM as `--mm-processor-kwargs '{"max_pixels": ...}'`. Both must
move together. Without this, a large page produces a request longer than the
context, vLLM answers 400, and the backend turns that into a 422 that fails the
page.

### NID-only local enhancement

NID modes apply an ordered chain of local transformations before their one
allowed OCR call. Each stage degrades to a pass-through with a warning, so a
difficult photograph still reaches the model:

1. Apply EXIF orientation and RGB normalisation.
2. **Rectify the card.** Detect the largest convex quadrilateral and warp it
   flat, but only when the result has an ID-1 aspect ratio (1.45–1.75; the
   standard is 1.586) and covers at least 25% of the frame. A card photographed
   on its side is rotated upright by the same warp. When no confident quad is
   found — a flatbed scan has none — the image passes through unchanged with a
   warning saying so.
3. **Flatten illumination.** Divide the LAB `L` channel by a heavily smoothed
   estimate of its own background. This removes shadow gradients and brings
   specular glare down to the level of clean laminate, so the CLAHE step below
   amplifies strokes rather than the edge of a highlight.
4. **Deskew** using the dominant text angle, but only when rectification
   declined and the angle is under 15°.
5. Resize toward a 1600-pixel long edge with Lanczos.
6. Add a 20-pixel white border.
7. Run CLAHE on `L` only, clip limit 2.0, 8×8 tiles.
8. Apply mild unsharp masking (amount 0.20).

Geometry is corrected before lighting and lighting before contrast, because
each stage assumes the previous one has run.

The relevant Compose settings are `NID_PREPROCESS_ENABLED`,
`NID_RECTIFY_ENABLED`, `NID_ILLUMINATION_ENABLED`, `NID_DESKEW_ENABLED`,
`NID_TARGET_LONG_EDGE`, `NID_MIN_SHORT_EDGE`, `NID_MAX_UPSCALE`,
`NID_BORDER_PX`, `NID_CLAHE_CLIP_LIMIT`, `NID_CLAHE_TILE_GRID_SIZE`, and
`NID_UNSHARP_AMOUNT`.

This is deliberately *not* crop fan-out, a second contrast view,
language-specific image synthesis, or an automatic retry of the model request.
General text/PDF modes do not receive this enhancement, which avoids silently
changing behavior for ordinary documents; PDFs are instead rasterized at
`MAX_PDF_DPI` (default 200) rather than a fixed 2× matrix, which was 144 dpi
and thin for small print.

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

### Dots behavior and prompts

dots.ocr ships a fixed prompt dictionary and was fine-tuned against those exact
strings. `profiles.py` reproduces them verbatim, keyed by their upstream names:

| Upstream key | Used for | Returns |
| --- | --- | --- |
| `prompt_layout_all_en` | documents, PDFs, and both NID modes | one JSON object per block: `bbox`, `category`, `text` |
| `prompt_ocr` | plain `text` mode when `DOTS_LAYOUT_PROMPT_ENABLED=false` | a flat transcription |
| `prompt_grounding_ocr` | re-reading one region | text inside a supplied box |

**Do not "improve" the wording.** An earlier revision appended two sentences to
`prompt_ocr` for NID modes, so that a blank `Blood Group:` label would not
suppress the rest of a card's transcription. For a model this size, drifting off
the trained contract is itself a plausible cause of the empty and looping
responses that change was trying to fix. Blank fields are handled in
postprocessing instead, where they cost nothing.

The layout task is what makes tables, formulas, figures and reading order
recoverable: it formats tables as HTML, formulas as LaTeX, omits text for
`Picture` blocks, and sorts everything into human reading order. `prompt_ocr`
returns none of that structure. For NID cards the same output supplies per-block
boxes, which is what lets fields be matched to their printed labels
geometrically instead of by assuming the value follows the label on the next
line.

Decoding uses `temperature: 0.0` with `repetition_penalty: 1.05`. Greedy
decoding on a degraded photograph can fall into a repeat loop that consumes the
whole token budget; a light penalty suppresses that without making the
transcription non-deterministic.

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
| `OCR_VRAM_CAP_GIB` | 18 | Target budget for inference server allocation. Actual resident use is ~8 GiB; vLLM reports ~57 GiB free at startup on this host. |
| `OCR_KV_CACHE_MIB` | 4096 | Avoid vLLM auto-sizing an excessive or invalid cache. Holds ~149,800 tokens. |
| `OCR_MAX_MODEL_LEN` | 24576 | The model supports 131,072 positions. At 8192, a full-page image landed within ~700 tokens of the limit, where vLLM answers 400 and the page fails. The KV cache already allocated covers 2 sequences at this length several times over, so raising it is free. |
| `OCR_MAX_IMAGE_TOKENS` | 8464 | Server-side backstop passed as `--mm-processor-kwargs '{"max_pixels": ...}'`. Must track the backend's own budget. |
| `OCR_ENFORCE_EAGER` | false | Eager mode was a memory workaround. With this much headroom it only costs decode throughput; CUDA graphs are worth several times on latency. Set back to `true` if startup memory regresses. |
| `OCR_MAX_NUM_SEQS` | 2 | Lets vLLM batch a small number of requests without flooding a small OCR model. |
| Backend `MAX_INFERENCE_CONCURRENCY` | 2 | Matches the inference bound and applies backpressure before requests overload vLLM. |

`OCR_MAX_MODEL_LEN`, `OCR_MAX_IMAGE_TOKENS`, `MAX_MODEL_LEN`, and
`TOKEN_BUDGET_MARGIN` are one decision expressed in four places. Changing one
without the others reintroduces the context-overflow failure.

Note that `entrypoint.sh` is baked into the inference image. Changing a flag
there requires `docker compose up -d --build inference`; a plain `up -d` silently
keeps the old launcher while still picking up environment-variable changes.

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
2. converts constrained Surya/Chandra whole-page HTML to readable text;
3. stops repeated decoder loops without requesting another inference — on
   **every** branch, not only the HTML one, so a dots.ocr loop is caught too;
4. separates JSON transcription and fields when a model legitimately returns
   an object;
5. detects Surya-like layout metadata arrays and never treats them as OCR text;
   and
6. builds `PageResult.layout` from either output shape (see below).

### Two layout output shapes

The layout prompt asks for a JSON object per block. Checkpoints differ in
whether they comply, so both are handled:

| Shape | Emitted by | Result |
| --- | --- | --- |
| `[{bbox, category, text}, ...]` | upstream `rednote-hilab/dots.ocr` | blocks **with** boxes; the NID extractor can match labels to values geometrically |
| Markdown with embedded `<table>` HTML | the served `dots-studio/dots.ocr` | blocks **without** boxes, recovered from the Markdown structure |

The served `dots-studio/dots.ocr` returns the second shape. Its structure is
still real and worth keeping — headings, HTML tables, captions — which is why
`blocks_from_markdown` reads categories back out of the text. But there are no
boxes, so **spatial NID extraction does not currently activate on this
checkpoint** and the text-based parser does the work. Do not delete that path.

`text` and `markdown` are deliberately different. `text` flattens a table to
pipe-separated rows for reading; `markdown` preserves the HTML the model
emitted, because that markup is the structure a document consumer wants.

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

- Names are accepted next to a `Name` label only when the candidate is an
  English-name-shaped value.
- Dates must be associated with their relevant label and match supported date
  forms.
- NID numbers must be associated with an NID label and have a supported length.
- Blood groups must match `A`, `B`, `AB`, or `O` plus `+`/`-`; a blank label is
  a valid absence, not a failed extraction.
- MRZ text is verified rather than merely pattern-matched. See below.

**Labels are matched as the model transcribes them, not as they are printed.**
This distinction was worth roughly seventeen points on the benchmark and is
easy to undo by "tidying" the patterns:

- `NID No.` comes back as **`NIC No.`** on most cards — the D reads as a C at
  card resolution. `LABEL_PATTERNS` therefore accepts `NID`/`NIC` and the
  common digit-letter confusions. Widening a *label* is safe because the value
  still has to satisfy its own validator, so a mislabelled match cannot emit an
  invalid number.
- The cards are bilingual and the model often emits `নাম Name` on one row, or
  only the Bengali label above an English value. Bengali labels are therefore
  matched too. This does not make the parser a Bengali extractor: the label is
  used only to *locate* an ASCII value, and the validators are unchanged.
- Labels are matched anywhere in a row rather than at its start. A start anchor
  rejected every bilingual row, Markdown table pipe, and run-on line, taking the
  value with it.
- A value is sliced at the next known label, in either direction, and an English
  name candidate additionally ends where another script or a digit begins — so
  `MD ALMAS নাম Date of Birth 11 Feb 1983` yields `MD ALMAS`.
- `নাম` is also the label for a parent's name, so a row carrying `পিতা`/`মাতা`
  (or `father`/`mother`) is skipped for the cardholder's name.
- On the NID back, OCR can place `Blood Group:`, `Place of Birth:`, and
  `Issue Date:` on one line because those labels are visually horizontal. The
  same next-label slicing preserves `place_of_birth` and `issue_date` when
  `Blood Group:` is intentionally blank.

The parser emits one warning per null field. If the model response has no text
at all, it additionally says that no transcription was available. That is an
observable model-output condition, not a reason to invent values or silently
declare a successful extraction.

Every field also carries a confidence and an evidence record naming the stage
that produced it (`layout` geometry, flat `text`, `mrz:valid`, `mrz:repaired`,
or `rapidocr`). These are what the benchmark uses to attribute a regression to a
specific stage.

### MRZ verification and repair (`backend/app/mrz.py`)

Bangladesh NID backs carry an ICAO 9303 **TD1** machine-readable zone: three
lines of exactly thirty characters. It is self-checking, and this is the highest
-value accuracy mechanism in the system because it costs nothing to run.

Three check digits are verified with the standard 7-3-1 weighting
(`<`=0, digits=value, `A`=10…`Z`=35):

| Check | Position | Covers |
| --- | --- | --- |
| Birth date | line 2, position 7 | line 2 positions 1–6 |
| Expiry date | line 2, position 15 | line 2 positions 9–14 |
| **Composite** | line 2, position 30 | `L1[6..30] + L2[1..7] + L2[9..15] + L2[19..29]` — fifty characters across both lines |

The composite digit is what makes this powerful: one arithmetic test covers
nearly the whole zone. Both sample cards used to develop this verify cleanly.

**The line 1 document-number check digit (position 15) is deliberately not
enforced.** On real Bangladesh cards it is `<` filler and does not satisfy the
standard algorithm over positions 6–14. Enforcing it would reject every valid
card.

A **structural** rule runs alongside the digits: numeric fields must contain
digits, the sex field must be `M`/`F`/`<`, nationality must be alphabetic. This
is not redundant. `G` has value 16, so misreading `640118` as `G40118` shifts
the weighted sum by exactly 70 and the printed check digit still matches; only
the field-shape rule catches it. A structural violation also localises the error
to one position, which is what makes the repair below decidable.

**Repair** substitutes within a fixed OCR glyph-confusion set
(`O↔0 I↔1 S↔5 B↔8 Z↔2 G↔6 U↔V K↔X`), constrained by field type, up to two
simultaneous edits, and accepts a result **only when exactly one candidate
satisfies every check**. Filler `<` is never a substitution source: cards print
long runs of it, and admitting each one as an editable position makes almost any
repair look ambiguous, suppressing the corrections that matter.

Expect repair to succeed only sometimes. A single mod-10 digit detects errors
reliably but cannot always localise them — many different single edits shift the
sum by the same amount. That is by design: when the digits do not decide, the
status is `unverified` and the field becomes a candidate for CPU reconciliation
rather than a guess.

Rows are also normalised to thirty characters before checking. Padding position
matters: rows one and three end in filler so length is corrected at the tail,
while row two ends with the composite check digit, so a dropped character is
restored inside the optional-data run just before it.

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

Run the whole suite in the production dependency container. No GPU, no local
Python environment, and the app/tests are mounted read-only so a run cannot
modify the tree:

```bash
docker run --rm -e PYTHONPATH=/app -e PYTHONDONTWRITEBYTECODE=1 -e OCR_API_KEY=test \
  --mount type=bind,source="$PWD/backend/app",target=/app/app,readonly \
  --mount type=bind,source="$PWD/backend/tests",target=/app/tests,readonly \
  vlm-ocr-backend pytest -q tests
```

`OCR_API_KEY` is required because `config.Settings` has no default for it. Add
`-k <expr>` or a path to narrow the run, e.g. `pytest -q tests/test_mrz.py`.
If the image is stale, rebuild first with `docker compose build backend`.

| File | Covers |
| --- | --- |
| `test_preprocess.py` | Upload/rasterisation limits, image-token cost, the context-budget guard, card rectification and its fallbacks, PDF DPI. |
| `test_profiles.py` | Model-profile selection, the verbatim dots.ocr prompt strings, and that NID modes are never budgeted below plain text. |
| `test_postprocess.py` | Response parsing, Bengali numeral conversion, NID field validation, blank-blood-group handling. |
| `test_layout.py` | Layout-block recovery from both output shapes, table/formula rendering, spatial and textual field matching. |
| `test_mrz.py` | TD1 check digits against two real cards, normalisation, repair, and the refusals. |
| `test_verify.py` | Reconciliation rules: what may replace a failed field, and on what evidence. |
| `test_service.py` | Per-page failure isolation, retry policy, `finish_reason`, per-mode prompt selection. |
| `test_jobs.py` | Job transitions, cancellation, expiry. |

Two of these encode findings that are easy to regress and expensive to
rediscover, so read them before changing the code they guard:

- `test_mrz.py` pins the check-digit arithmetic against two **real** cards whose
  digits were computed by hand before the implementation existed, and asserts
  that the line-1 document check digit is *not* enforced.
- `test_verify.py::test_free_text_fields_are_never_supplied_by_the_second_engine`
  encodes a measured regression: the CPU recogniser drops spaces in wide-set
  capitals, and `name`'s validator is too weak to reject the result.

The blank-blood-group regression test represents the realistic one-line layout:

```text
Blood Group:  Place of Birth: CHANDPUR  Issue Date: 16 Jan 2018
```

It verifies that `blood_group` is null but the other two labelled values and
MRZ lines survive.

`docker-compose.test.yml` replaces inference with a CPU-only mock server. It
is for API/Compose smoke coverage, not OCR quality evaluation.

Unit tests cannot tell you whether accuracy improved. Every prompt,
preprocessing, and parser change in the tuning record below passed the suite
both before and after — several of them *lost* accuracy. Use the benchmark.

### Benchmarking NID accuracy

There are two independent test layers and they answer different questions.
Unit tests answer *did I break the contract*; the benchmark answers *did
accuracy actually improve*. A change to preprocessing, prompting, or the parser
is not finished until both have run, because every one of those knobs can look
correct in a unit test and lose accuracy on real cards.

#### The benchmark corpus

External benchmark images and ground truth must remain **outside this
repository**; they are identity documents. On the current development host the
corpus lives at `/home/rivu/Repositories/vlm-playground/ocr-benchmark`, which is
also the repository that produced the earlier two-model baseline.

`evaluate_nid.py` expects exactly this layout, and `--benchmark-root` points at
the directory containing `data/`:

```text
<benchmark-root>/
└── data/
    ├── ground_truth/
    │   ├── nid_front_train.json      # {"documents": {"<image>.png": {"fields": {...}}}}
    │   ├── nid_front_valid.json
    │   ├── nid_front_test.json
    │   └── nid_back_*.json
    └── images/
        ├── nid_front/{train,valid,test}/
        └── nid_back/{train,valid,test}/
```

Ground-truth field names differ from the API's field names on purpose; the
mapping is `FIELD_MAP` in `evaluate_nid.py`:

| API field | Ground-truth key |
| --- | --- |
| `name` | `name_en` |
| `dob` | `dob` |
| `nid_no` | `nid_number` |

**What is actually populated today.** Only `nid_front_train.json` has ground
truth (262 documents). The other five files exist but are empty, so
`nid_front` / `train` is the only runnable evaluation. The images for every
split and both sides are present (front 263/56/56, back 255/55/55), so filling
in a ground-truth file is all that is needed to extend coverage.

#### Scoring rules

Scoring deliberately mirrors the earlier benchmark's `evaluation/metrics.py`, so
numbers from the two systems are directly comparable:

- names and dates: whitespace collapsed, compared case-insensitively, exact
  equality — **punctuation is significant**, so `MST. SHARMIN` does not match
  `MST SHARMIN`;
- NID numbers: reduced to digits, then exact equality;
- a field whose ground truth is blank is excluded from the accuracy denominator
  and scored only for false positives.

`near_miss` classifies a *failed* comparison (`punctuation_only`, `spacing_only`,
`honorific_only`, `one_character`, `missing`, `different`). It is a triage aid
and is **never** counted as a match — it exists so a ground-truth transcription
convention can be told apart from a genuine misread.

#### Running an evaluation

The stack must be up and healthy first (`docker compose ps`). The evaluator
calls the real HTTP API, so it measures the whole pipeline, not a library.

```bash
# Container form; no local Python environment needed. --network host lets the
# container reach the backend published on the host's port 8000.
docker run --rm --network host \
  --mount type=bind,source="$PWD/backend/scripts",target=/app/scripts,readonly \
  --mount type=bind,source=/home/rivu/Repositories/vlm-playground/ocr-benchmark,target=/bench,readonly \
  -v "$PWD/benchmark-results":/out \
  -e OCR_API_KEY="$(grep -E '^OCR_API_KEY=' .env | cut -d= -f2-)" \
  vlm-ocr-backend python /app/scripts/evaluate_nid.py \
    --benchmark-root /bench --type nid_front --split train \
    --concurrency 2 --output /out/front_train.json
```

Run it directly instead if you have the dependencies locally:

```bash
export OCR_API_KEY=...
python backend/scripts/evaluate_nid.py \
  --benchmark-root /home/rivu/Repositories/vlm-playground/ocr-benchmark \
  --type nid_front --split train --concurrency 2 \
  --output benchmark-results/front_train.json
```

| Flag | Meaning |
| --- | --- |
| `--benchmark-root` | Directory containing `data/`. |
| `--type` | `nid_front` or `nid_back`. |
| `--split` | `train`, `valid`, or `test`. |
| `--limit N` | Only the first N documents. Use for quick iteration; `--limit 40` takes a few minutes against ~20 for the full split. |
| `--concurrency` | Parallel requests. **Keep at or below `MAX_INFERENCE_CONCURRENCY`** (default 2); higher only queues inside vLLM and distorts timing. |
| `--base-url` | Point at an alternative backend, which is how A/B runs work (below). |
| `--output` | JSON report. The directory is created if absent. |

Keep `benchmark-results/` out of Git: rows contain transcribed identity data.

The console summary is deliberately shaped like the earlier benchmark's table:

```text
=== nid_front / train ===
field             total  correct  accuracy
name                 40       29    72.50%
dob                  40       37    92.50%
nid_no               40       38    95.00%
OVERALL             120      104    86.67%

false positives on blank fields: 0
truncated pages: 14   request errors: 0
failure breakdown: {'missing': 11, 'different': 2, ...}
```

The JSON report holds much more: `per_field` (with `sources`, `mean_confidence`
and `near_misses`), `rows` (every comparison), and `diagnostics` (per-document
`finish_reason`, `layout_blocks`, `mrz_status`, warnings).

#### Reading the pipeline-health numbers

These are not accuracy, and they are usually what to fix first:

| Number | What a bad value means |
| --- | --- |
| `truncated pages` | The model ran to its token limit, almost always a repetition loop. Measured: pages that truncate score ~76% against ~92% for clean ones. Raise `NID_FRONT_REPETITION_PENALTY` (see the tuning record below). |
| `request_errors` | HTTP failures. Any non-zero value invalidates the run. |
| `pages_with_layout` | Zero on cards is expected — cards use the plain OCR task. Zero on documents means the layout prompt is not being honoured. |
| `mrz_status` (back only) | Tally of `valid` / `repaired` / `unverified` / `absent`. |
| `per_field[...].sources` | Which stage produced each value: `text`, `layout`, `mrz`, `rapidocr`. A sudden shift toward `rapidocr` means the first pass is degrading. |

#### A/B testing a configuration change

Nearly every tuning knob is an environment variable, so a comparison does not
need a rebuild. Run a second backend on another port against the *same*
inference server and point the evaluator at it. This is how the tuning record
below was produced.

```bash
KEY=$(grep -E '^OCR_API_KEY=' .env | cut -d= -f2-)
docker run -d --name ab-backend --network vlm-ocr_default -p 8001:8000 \
  -e OCR_API_KEY="$KEY" -e INFERENCE_BASE_URL=http://inference:8000 \
  -e NID_FRONT_REPETITION_PENALTY=1.15 \
  vlm-ocr-backend

# ... then add --base-url http://localhost:8001 to the evaluator command.
docker rm -f ab-backend      # afterwards
```

Change one variable at a time, keep `--limit` and the split fixed between arms,
and remember that a 40-document run is 120 field comparisons — a one-point
difference is roughly one field and is not a result.

#### Protocol

1. Keep the held-out `test` split untouched.
2. Tune preprocessing, prompt, and parser thresholds only on `train` and
   `valid`.
3. Require at least 99% exact match for supported non-empty English fields and
   zero false positives for visibly blank fields before making a quality claim.
4. Run the held-out test exactly once, for the final report.
5. Report English and Bengali limitations separately; Bengali raw OCR is not
   proof that structured Bengali fields are production-ready.

The generic `backend/scripts/evaluate.py` compares manually paired expected and
actual JSON fixtures, producing character error rate and exact field match.

### Tuning record

Every number below is `nid_front` / `train`, measured through the HTTP API. The
point of keeping this is that several changes which looked obviously correct
lost accuracy, and the reasons are not recoverable from the code alone.

**Baseline to beat.** The earlier two-model system (lightweight OCR + Qwen
judge, ~45 GiB) on the same 262 documents, from
`ocr-benchmark/notebooks/accuracy_analysis.ipynb`:

| Run | Date of Birth | NID Number | Name (EN) | Overall |
| --- | ---: | ---: | ---: | ---: |
| sequential 1 | 98.85% | 97.33% | 90.08% | 95.42% |
| sequential 2 | 98.85% | 83.59% | 90.46% | 90.97% |
| sequential 3 | 98.85% | 98.09% | 90.46% | 95.80% |
| parallel 1 | 95.80% | 95.42% | 87.40% | 92.88% |
| parallel 2 | 96.18% | 95.42% | 87.40% | 93.00% |

Note the spread: the same system scored 90.97% and 95.80% on identical inputs.
Run-to-run variance of several points is normal here, so a small difference
between two configurations is not a result.

**What moved the number** (40-document subset, so 120 field comparisons —
roughly 0.8 points per field — unless marked *full split*):

| Change | Overall | Note |
| --- | ---: | --- |
| Layout prompt on cards | 66.7% | Starting point. 8/24 pages truncating. |
| → plain OCR prompt on cards | 70.8% | The layout task collapses a card into run-on lines and loops on dense ones. |
| → restrict CPU reconciliation | 66.7% | Correct change, *lower* score: it stopped a wrong-but-plausible name from being emitted. |
| → label-matching fixes | 86.7% | The single biggest win. See below. |
| → repetition penalty 1.15 | 88.3% | Truncated pages 14 → 1. |
| → **repetition penalty 1.30** | **90.8%** | Truncated pages → 0. Adopted. |
| → repetition penalty 1.50 | 85.0% | Worse, and truncation returns (16/40). Not monotonic. |
| penalty 1.30, preprocessing off | 87.5% | Preprocessing helps *once looping is fixed*. |
| *full split*, all of the above | **86.8%** | 262 documents. The subset was optimistic by 4 points. |
| *full split*, + name-row fixes | **87.5%** | name 76.3%, dob 93.5%, nid_no 92.8%. Adopted. |
| *full split*, + structural name fallback | 87.4% | **Tried and rejected — see below.** |

**Rejected: unlabelled name recovery.** 52 of the remaining failures are fields
where no labelled candidate validated, so an obvious next step is to fall back
to the card's fixed layout — take the first all-caps Latin row above the date of
birth, since the cardholder's name is printed there and a parent's name below
it. That was implemented, gated behind a flag, and measured: **87.40% against
87.53% without it.** No gain.

The run also shows why a small difference proves nothing here. `dob` and
`nid_no` moved by 0.4 points *between the two runs* even though the fallback
cannot touch those fields — that is the model's own nondeterminism, and it is
the same size as the effect being measured.

The code was removed rather than left switched off. It relaxes *where* a value
may come from, which is exactly the rule in **Explicit non-goals** #3, and a
contract relaxation that buys nothing measurable is a liability rather than a
dormant feature. Anyone reaching for this idea again should expect the label
patterns, not the search region, to be where the remaining `missing` cases live.

**Current standing.** 87.5% against a baseline that ranged 91.0–95.8%. The
remaining gap is roughly 5 points on `dob` and `nid_no` and 14 points on `name`.
That is a real deficit against a two-model system and should be stated as such;
the trade is ~8 GiB of VRAM against ~45 GiB, one model instead of two, and a
result that is auditable field-by-field.

Where the 98 remaining failures sit on the full split:

| Bucket | Count | Nature |
| --- | ---: | --- |
| `missing` | 52 | No labelled candidate validated. The real remaining work. |
| `different` | 16 | Model misread the name outright. |
| `one_character` | 14 | Genuine single-character OCR errors. |
| `punctuation_only` | 11 | **Ground-truth convention, not a defect.** |
| `spacing_only` / `honorific_only` | 5 | Mostly convention. |

The `punctuation_only` bucket is unwinnable and should not be optimised
against: the ground truth is itself inconsistent about the honorific period,
containing both `MST MONJELA KHATUN` and `MST. ROTNA KHATUN`. No parser
satisfies both. The earlier benchmark's error lists show it losing the same
cards, so the comparison remains fair.

Findings worth keeping:

1. **The label patterns were the bottleneck, not the model.** The model
   transcribes `NID No.` as **`NIC No.`** on most cards — the D reads as a C at
   card resolution — and a pattern requiring a literal "NID" dropped the number
   entirely. Cards are also bilingual: the model frequently emits `নাম Name`
   on one row, or only the Bengali label above an English value, both of which a
   start-anchored match rejects. Widening the label patterns and matching them
   anywhere in a row took `nid_no` from 79% to 95% and `name` from 33% to 72%.
   Widening a *label* is safe; the value still has to satisfy its validator.
2. **Repetition penalty is the lever against truncation, and it is not
   monotonic.** 1.30 eliminated looping; 1.50 brought it back worse than 1.05.
   Pages that truncate score ~76% against ~92% for clean ones, so this is worth
   more than it sounds.
3. **Measure interactions, not knobs in isolation.** Preprocessing measured as
   *harmful* at penalty 1.05 (66.7% vs 70.8% with it off) and *helpful* at 1.30
   (90.8% vs 87.5%). The first reading was confounded by looping. Re-check a
   rejected change after fixing something upstream of it.
4. **A better validator can lower the score and still be right.** Restricting
   reconciliation to fields whose validator can confirm a reading cost ~4 points
   because it replaced confidently-wrong names with nulls. The contract prefers
   null; that trade is deliberate and must not be "fixed" by re-admitting them.
5. **Some remaining failures are ground-truth convention, not OCR.** The
   `punctuation_only` and `honorific_only` buckets are cases like `MST. SHARMIN`
   against a ground truth of `MST SHARMIN`. The earlier benchmark counted these
   as errors too, so the comparison is fair, but they are not model defects and
   chasing them with parser rules would be overfitting.
6. **The model is not deterministic across calls** even at `temperature: 0.0`,
   because continuous batching changes the numerics. Do not conclude anything
   from a single document.

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
| One page fails and the whole job fails | Should no longer happen: `service.process` gathers with `return_exceptions=True` and only raises when *every* page failed. Check `metadata.failed_pages` and the per-page warning. |
| `finish_reason` is `length` | The output budget was exhausted before the page finished. On an NID back this removes the MRZ, which is printed last. Raise `nid_max_tokens` in `profiles.py`. |
| Upstream 400 / backend 422 mentioning context length | The image token cost exceeded the served context. Compare `metadata.image_tokens` against `MAX_MODEL_LEN - TOKEN_BUDGET_MARGIN`, and confirm the inference image was rebuilt so `--mm-processor-kwargs` is actually on the command line. |
| MRZ lines null but visible on the card | Check the `mrz` evidence source. `unverified` means the check digits failed and repair was ambiguous. Confirm reconciliation is enabled and that the rendered page is at least ~1600px on its long edge; the CPU recogniser cannot read a 30-character line much below that. |
| Reconciliation never runs | `NID_VERIFY_ENABLED`, or the backend image predates `rapidocr-onnxruntime`. The import failure is logged once and degrades quietly. |

## Explicit non-goals and invariants

The following are intentional constraints. A future change that breaks one
must include a written design review and measured evidence that the tradeoff is
worth it.

1. **No judge model.** Do not reintroduce Qwen or another verifier as an
   implicit second model request merely to fill a field. The GPU holds exactly
   one model.
2. **One VLM call per page.** No crop fan-out, retried alternate enhancement,
   ensemble, consensus, or second-pass schema request in normal production flow.
   *Amended:* a page that fails for a transient reason (timeout, upstream 5xx)
   is retried up to `UPSTREAM_MAX_ATTEMPTS`; a 4xx never is, because a contract
   error repeats identically. And a field that fails its validator may be
   re-read by the **CPU** engine in `verify.py`. That engine is a ~15 MB ONNX
   recogniser, costs no VRAM, runs only on failure, and may only return a value
   that passes the same validator the first reading failed.
3. **No guessing identity data.** NID fields need source-transcription
   evidence and validation; unknown is `null`. This is unchanged and
   load-bearing. Checksum-driven MRZ repair does not weaken it: it is
   arithmetic verification against digits printed on the card, it substitutes
   only within a fixed glyph-confusion set, and it accepts a correction **only
   when the check digits single out one answer**. Two candidate readings that
   both satisfy the digits are evidence for neither, and the field stays null.
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
7. NID-back hardening treated blank blood groups as legitimate, protected
   adjacent same-row labelled fields, and asked Dots to transcribe the rest of a
   card even if a label has no value.
8. The accuracy pass on `dev/optimize-accuracy` addressed whole-job failures and
   the extraction ceiling together:
   - a failing page no longer discards the pages that succeeded, transient
     upstream errors are retried, and `finish_reason` is surfaced;
   - image size is bounded against the served context, and `max_model_len` was
     raised to 24576 with eager mode disabled;
   - NID modes were budgeted *above* plain text rather than below it, which had
     been truncating the MRZ;
   - prompting moved to dots.ocr's verbatim layout contract, recovering tables,
     captions and reading order for documents;
   - NID preprocessing gained card rectification, illumination flattening and
     deskew, each with a pass-through fallback;
   - the TD1 MRZ is now checksum-verified and repaired; and
   - a CPU-only ONNX recogniser re-reads fields that fail validation.
9. The same pass was then measured against the external NID front train split
   for the first time, which corrected several of its own assumptions: the
   layout prompt was moved off cards, CPU reconciliation was restricted to
   fields whose validator can confirm a reading, the label patterns were widened
   to what the model actually transcribes (`NIC No.`, bilingual label rows), and
   the repetition penalty was tuned. See the tuning record under *Quality
   strategy and tests*.

When reviewing an incident, compare the deployed commit with this chronology.
Older image caches or remote clones may not contain the current GB10, Dots,
NID-preprocessing, or blank-field behavior.

## Maintainer checklist

Before merging/deploying OCR changes:

- [ ] Run backend tests.
- [ ] Run the NID benchmark on `train` **before and after** the change and
      compare. A unit-test pass is not evidence that accuracy held; several
      changes in the tuning record passed the suite and lost points.
- [ ] Check the pipeline-health counters, not just accuracy: `truncated pages`
      and `request_errors` should both be zero.
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

