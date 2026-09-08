from typing import Any, Literal
from pydantic import BaseModel, Field


DocumentMode = Literal["text", "nid_front", "nid_back", "schema"]


class LayoutBlockModel(BaseModel):
    """One layout element as reported by the model's layout task."""

    category: str
    text: str = ""
    bbox: list[float] | None = None


class PageResult(BaseModel):
    page_number: int
    text: str
    markdown: str
    fields: dict[str, Any] | None = None
    warnings: list[str] = Field(default_factory=list)
    # Additive: absent for models and prompts that return no layout information.
    layout: list[LayoutBlockModel] | None = None
    # How well each extracted field is supported by the transcription, so a
    # caller can triage rather than treating every value as equally certain.
    field_confidence: dict[str, float] | None = None
    field_evidence: dict[str, Any] | None = None
    finish_reason: str | None = None


class OCRMetadata(BaseModel):
    request_id: str
    model: str
    serving_engine: str
    page_count: int
    elapsed_ms: int
    image_tokens: int | None = None
    failed_pages: int = 0


class OCRResult(BaseModel):
    status: Literal["completed"] = "completed"
    result: list[PageResult]
    metadata: OCRMetadata


class JobResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    result: OCRResult | None = None
    error: str | None = None


class RuntimeResponse(BaseModel):
    model: str
    profile: str
    serving_engine: str
    max_inference_concurrency: int
    max_upload_mib: int
    max_pdf_pages: int
