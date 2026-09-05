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
from .preprocess import DocumentError, render_document
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
    yield


app = FastAPI(title="Single-Model OCR Service", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[get_settings().frontend_origin],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)


async def process_upload(file: UploadFile, mode: str, schema_raw: str | None, request: Request) -> tuple[str, OCRResult]:
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
        pages = render_document(content, file.content_type, settings.max_pdf_pages, settings.max_page_dimension)
    except DocumentError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    request_id = str(uuid.uuid4())
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
async def direct_job(
    request: Request,
    file: UploadFile = File(...),
    mode: str = Form("text", pattern="^(text|nid_front|nid_back|schema)$"),
    schema: str | None = Form(default=None),
) -> JobResponse:
    job_id, result = await process_upload(file, mode, schema, request)
    return JobResponse(job_id=job_id, result=result)


@app.get("/api/v1/jobs/{job_id}", response_model=JobResponse, dependencies=[Depends(require_api_key)])
async def get_job(request: Request, job_id: str) -> JobResponse:
    result = request.app.state.results.get(job_id)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found or expired. Direct results are not persisted across restarts.")
    return JobResponse(job_id=job_id, result=result)


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
    return JobResponse(job_id=job_id, result=result)


@app.get("/v1/ocr/results/{job_id}", dependencies=[Depends(require_api_key)])
async def legacy_result(request: Request, job_id: str) -> dict:
    result = request.app.state.results.get(job_id)
    if not result:
        return {"status": "expired", "result": None}
    return {"status": "completed", "result": result.model_dump()}
