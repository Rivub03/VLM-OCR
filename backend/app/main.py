import base64
import json
import secrets
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .cache import ResultCache
from .config import Settings, get_settings
from .jobs import JobManager
from .preprocess import DocumentError, NidPreprocessOptions, render_document
from .profiles import profile_for
from .schemas import JobResponse, OCRResult, RuntimeResponse
from .service import OCRService, UpstreamError


def settings_dependency() -> Settings:
    return get_settings()


def require_api_key(x_api_key: str | None = Header(default=None), settings: Settings = Depends(settings_dependency)) -> None:
    if not x_api_key or not secrets.compare_digest(x_api_key, settings.ocr_api_key):
        raise HTTPException(status_code=401, detail="A valid X-API-Key is required.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.ocr = OCRService(settings)
    app.state.results = ResultCache(settings.result_ttl_seconds)
    app.state.jobs = JobManager(app.state.results)
    try:
        yield
    finally:
        await app.state.jobs.shutdown()
        await app.state.ocr.aclose()


app = FastAPI(title="Single-Model OCR Service", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[get_settings().frontend_origin],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["X-API-Key", "Content-Type"],
)


async def prepare_upload(file: UploadFile, mode: str, schema_raw: str | None) -> tuple[str, list, dict | None]:
    settings = get_settings()
    content = await file.read()
    if len(content) > settings.max_upload_mib * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Uploads are limited to {settings.max_upload_mib} MiB.")
    schema: dict | None = None
    if mode == "schema":
        if not schema_raw:
            raise HTTPException(status_code=422, detail="A JSON schema is required in schema mode.")
        try:
            schema = json.loads(schema_raw)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=422, detail="The extraction schema is not valid JSON.") from exc
        if not isinstance(schema, dict):
            raise HTTPException(status_code=422, detail="The extraction schema must be a JSON object.")
    try:
        nid_options = NidPreprocessOptions(
            enabled=settings.nid_preprocess_enabled,
            rectify=settings.nid_rectify_enabled,
            illumination=settings.nid_illumination_enabled,
            deskew=settings.nid_deskew_enabled,
            target_long_edge=settings.nid_target_long_edge,
            min_short_edge=settings.nid_min_short_edge,
            max_upscale=settings.nid_max_upscale,
            border_px=settings.nid_border_px,
            clahe_clip_limit=settings.nid_clahe_clip_limit,
            clahe_tile_grid_size=settings.nid_clahe_tile_grid_size,
            unsharp_amount=settings.nid_unsharp_amount,
        )
        pages = render_document(
            content,
            file.content_type,
            settings.max_pdf_pages,
            settings.max_page_dimension,
            nid_mode=mode.startswith("nid_"),
            nid_options=nid_options,
            token_budget=settings.image_token_budget,
            patch_size=settings.vision_patch_size,
            merge_size=settings.vision_merge_size,
            pdf_dpi=settings.max_pdf_dpi,
        )
    except DocumentError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    request_id = str(uuid.uuid4())
    return request_id, pages, schema


async def process_upload(file: UploadFile, mode: str, schema_raw: str | None, request: Request) -> tuple[str, OCRResult]:
    request_id, pages, schema = await prepare_upload(file, mode, schema_raw)
    try:
        result = await request.app.state.ocr.process(request_id, pages, mode, schema)
    except UpstreamError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    request.app.state.results.put(request_id, result)
    return request_id, result


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v1/runtime", response_model=RuntimeResponse, dependencies=[Depends(require_api_key)])
async def runtime(request: Request) -> RuntimeResponse:
    try:
        active = await request.app.state.ocr.runtime()
    except UpstreamError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    settings = get_settings()
    return RuntimeResponse(
        model=active.model,
        profile=profile_for(active.model).name,
        serving_engine=active.engine,
        max_inference_concurrency=settings.max_inference_concurrency,
        max_upload_mib=settings.max_upload_mib,
        max_pdf_pages=settings.max_pdf_pages,
    )


@app.post("/api/v1/ocr", response_model=OCRResult, dependencies=[Depends(require_api_key)])
async def ocr(
    request: Request,
    file: UploadFile = File(...),
    mode: str = Form("text", pattern="^(text|nid_front|nid_back|schema)$"),
    schema: str | None = Form(default=None),
) -> OCRResult:
    _, result = await process_upload(file, mode, schema, request)
    return result


@app.post("/api/v1/jobs", response_model=JobResponse, dependencies=[Depends(require_api_key)])
async def create_job(
    request: Request,
    file: UploadFile = File(...),
    mode: str = Form("text", pattern="^(text|nid_front|nid_back|schema)$"),
    schema: str | None = Form(default=None),
) -> JobResponse:
    job_id, pages, schema_object = await prepare_upload(file, mode, schema)

    async def work() -> OCRResult:
        return await request.app.state.ocr.process(job_id, pages, mode, schema_object)

    return request.app.state.jobs.start(job_id, work)


@app.get("/api/v1/jobs/{job_id}", response_model=JobResponse, dependencies=[Depends(require_api_key)])
async def get_job(request: Request, job_id: str) -> JobResponse:
    try:
        return request.app.state.jobs.response(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found or expired. Jobs are not persisted across restarts.") from exc


@app.delete("/api/v1/jobs/{job_id}", response_model=JobResponse, dependencies=[Depends(require_api_key)])
async def cancel_job(request: Request, job_id: str) -> JobResponse:
    try:
        return request.app.state.jobs.cancel(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found or expired.") from exc


# Compatibility adapters for the supplied service's public routes.
@app.post("/direct", dependencies=[Depends(require_api_key)])
async def legacy_direct(request: Request, file: UploadFile = File(...), type: str = Form(...)) -> JSONResponse:
    if type not in {"nid_front", "nid_back"}:
        raise HTTPException(status_code=422, detail="type must be nid_front or nid_back")
    _, result = await process_upload(file, type, None, request)
    fields = result.result[0].fields or {}
    return JSONResponse({"message": "Request successful", "status": 200, "data": fields})


@app.post("/direct/base64", dependencies=[Depends(require_api_key)])
async def legacy_base64(request: Request, body: dict) -> JSONResponse:
    encoded = str(body.get("image_base64", "")).split(",")[-1]
    try:
        content = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=422, detail="image_base64 is invalid") from exc
    document_type = body.get("type")
    if document_type not in {"nid_front", "nid_back"}:
        raise HTTPException(status_code=422, detail="type must be nid_front or nid_back")
    upload = UploadFile(filename="document.png", file=__import__("io").BytesIO(content), headers={"content-type": "image/png"})
    _, result = await process_upload(upload, document_type, None, request)
    return JSONResponse({"message": "Request successful", "status": 200, "data": result.result[0].fields or {}})


@app.post("/v1/ocr/schema", dependencies=[Depends(require_api_key)])
async def legacy_schema(request: Request, file: UploadFile = File(...), schema: str | None = Form(default=None)) -> JobResponse:
    mode = "schema" if schema else "text"
    job_id, result = await process_upload(file, mode, schema, request)
    return JobResponse(job_id=job_id, status="completed", result=result)


@app.get("/v1/ocr/results/{job_id}", dependencies=[Depends(require_api_key)])
async def legacy_result(request: Request, job_id: str) -> dict:
    result = request.app.state.results.get(job_id)
    if not result:
        return {"status": "expired", "result": None}
    return {"status": "completed", "result": result.model_dump()}
