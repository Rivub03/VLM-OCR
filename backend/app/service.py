import asyncio
import base64
import time
from dataclasses import dataclass
from typing import Any

import httpx

from .config import Settings
from .postprocess import normalise_fields, parse_response
from .preprocess import RenderedPage
from .profiles import make_payload, profile_for
from .schemas import OCRMetadata, OCRResult, PageResult


class UpstreamError(RuntimeError):
    def __init__(self, message: str, status_code: int = 503):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class InferenceRuntime:
    model: str
    engine: str


class OCRService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.semaphore = asyncio.Semaphore(settings.max_inference_concurrency)

    async def runtime(self) -> InferenceRuntime:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.settings.inference_base_url.rstrip('/')}/v1/models")
                response.raise_for_status()
                entries = response.json().get("data", [])
                model = entries[0].get("id") if entries else "unknown"
                return InferenceRuntime(model=model, engine="sglang" if "sglang" in response.headers.get("server", "").lower() else "vllm")
        except httpx.HTTPError as exc:
            raise UpstreamError("The OCR inference server is unavailable.") from exc

    async def _one_page(self, page: RenderedPage, model: str, mode: str, schema: dict[str, Any] | None) -> PageResult:
        data_url = "data:image/png;base64," + base64.b64encode(page.content).decode("ascii")
        payload = make_payload(model, data_url, mode, schema)
        async with self.semaphore:
            try:
                async with httpx.AsyncClient(timeout=self.settings.upstream_timeout_seconds) as client:
                    response = await client.post(f"{self.settings.inference_base_url.rstrip('/')}/v1/chat/completions", json=payload)
                    response.raise_for_status()
            except httpx.TimeoutException as exc:
                raise UpstreamError("The OCR inference server timed out.") from exc
            except httpx.HTTPStatusError as exc:
                detail = exc.response.text.strip().replace("\n", " ")[:500]
                if 400 <= exc.response.status_code < 500:
                    raise UpstreamError(
                        f"The OCR inference server rejected the request: {detail or 'invalid inference request.'}",
                        status_code=422,
                    ) from exc
                raise UpstreamError("The OCR inference server failed while processing the request.") from exc
            except httpx.HTTPError as exc:
                raise UpstreamError("The OCR inference server rejected the request.") from exc
        try:
            output = response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise UpstreamError("The OCR inference server returned an invalid response.") from exc
        profile = profile_for(model)
        text, fields, warnings = parse_response(str(output))
        if mode.startswith("nid_") and not fields:
            warnings.append("Structured NID fields were derived from the single OCR transcription.")
        if mode == "schema" and profile.native_output == "html" and not fields:
            warnings.append("This model uses native HTML OCR; custom fields were derived deterministically from the single transcription.")
        fields = normalise_fields(fields, text, mode, schema)
        return PageResult(page_number=page.number, text=text, markdown=text, fields=fields, warnings=warnings)

    async def process(self, request_id: str, pages: list[RenderedPage], mode: str, schema: dict[str, Any] | None) -> OCRResult:
        started = time.perf_counter()
        runtime = await self.runtime()
        results = await asyncio.gather(*(self._one_page(page, runtime.model, mode, schema) for page in pages))
        metadata = OCRMetadata(
            request_id=request_id,
            model=runtime.model,
            serving_engine=runtime.engine,
            page_count=len(results),
            elapsed_ms=round((time.perf_counter() - started) * 1000),
        )
        return OCRResult(result=results, metadata=metadata)
