from typing import Any, Literal
from pydantic import BaseModel, Field


DocumentMode = Literal["text", "nid_front", "nid_back", "schema"]


class PageResult(BaseModel):
    page_number: int
    text: str
    markdown: str
    fields: dict[str, Any] | None = None
    warnings: list[str] = Field(default_factory=list)


class OCRMetadata(BaseModel):
    request_id: str
    model: str
    serving_engine: str
    page_count: int
    elapsed_ms: int


class OCRResult(BaseModel):
    status: Literal["completed"] = "completed"
    result: list[PageResult]
    metadata: OCRMetadata


class JobResponse(BaseModel):
    job_id: str
    status: Literal["completed"] = "completed"
    result: OCRResult


class RuntimeResponse(BaseModel):
    model: str
    profile: str
    serving_engine: str
    max_inference_concurrency: int
    max_upload_mib: int
    max_pdf_pages: int

