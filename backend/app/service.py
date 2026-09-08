import asyncio
import base64
import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

from .config import Settings
from .nid import extract_nid
from .postprocess import normalise_fields, parse_response
from .preprocess import RenderedPage, image_token_cost
from .profiles import make_payload, max_tokens_for, profile_for
from .schemas import LayoutBlockModel, OCRMetadata, OCRResult, PageResult
from .verify import Reconciler


logger = logging.getLogger(__name__)


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
        # One pooled client for the process. Building a client per page reopened
        # a connection for every request and discarded the pool each time.
        self._client = httpx.AsyncClient(timeout=settings.upstream_timeout_seconds)
        self._reconciler = Reconciler(settings)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def runtime(self) -> InferenceRuntime:
        try:
            response = await self._client.get(
                f"{self.settings.inference_base_url.rstrip('/')}/v1/models", timeout=5.0,
            )
            response.raise_for_status()
            entries = response.json().get("data", [])
            model = entries[0].get("id") if entries else "unknown"
            engine = "sglang" if "sglang" in response.headers.get("server", "").lower() else "vllm"
            return InferenceRuntime(model=model, engine=engine)
        except httpx.HTTPError as exc:
            raise UpstreamError("The OCR inference server is unavailable.") from exc

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """One bounded inference call, retried only for transient conditions."""
        url = f"{self.settings.inference_base_url.rstrip('/')}/v1/chat/completions"
        attempts = max(1, self.settings.upstream_max_attempts)
        last: Exception | None = None
        for attempt in range(attempts):
            async with self.semaphore:
                try:
                    response = await self._client.post(url, json=payload)
                    response.raise_for_status()
                    return response.json()
                except httpx.HTTPStatusError as exc:
                    detail = exc.response.text.strip().replace("\n", " ")[:500]
                    if 400 <= exc.response.status_code < 500:
                        # A 4xx is a contract error: the same request will be
                        # rejected identically, so retrying only wastes time.
                        raise UpstreamError(
                            f"The OCR inference server rejected the request: {detail or 'invalid inference request.'}",
                            status_code=422,
                        ) from exc
                    last = exc
                except (httpx.TimeoutException, httpx.HTTPError) as exc:
                    last = exc
            if attempt + 1 < attempts:
                await asyncio.sleep(self.settings.upstream_retry_backoff_seconds * (2 ** attempt))
                logger.warning("Retrying inference request after %s (attempt %d/%d)", type(last).__name__, attempt + 2, attempts)
        if isinstance(last, httpx.TimeoutException):
            raise UpstreamError("The OCR inference server timed out.") from last
        raise UpstreamError("The OCR inference server failed while processing the request.") from last

    def _repetition_penalty(self, mode: str) -> float:
        if mode == "nid_front":
            return self.settings.nid_front_repetition_penalty
        if mode == "nid_back":
            return self.settings.nid_back_repetition_penalty
        return self.settings.repetition_penalty

    async def _one_page(self, page: RenderedPage, model: str, mode: str, schema: dict[str, Any] | None) -> PageResult:
        data_url = "data:image/png;base64," + base64.b64encode(page.content).decode("ascii")
        profile = profile_for(model)
        is_nid = mode.startswith("nid_")
        payload = make_payload(
            model, data_url, mode, schema,
            use_layout_prompt=(
                self.settings.nid_layout_prompt_enabled if is_nid
                else self.settings.dots_layout_prompt_enabled
            ),
            repetition_penalty=self._repetition_penalty(mode),
            temperature=self.settings.nid_temperature if is_nid else 0.0,
        )
        tokens = image_token_cost(
            page.width or 0, page.height or 0,
            self.settings.vision_patch_size, self.settings.vision_merge_size,
        )
        logger.info(
            "ocr page=%s mode=%s model=%s size=%sx%s image_tokens=%s max_tokens=%s rectified=%s",
            page.number, mode, model, page.width, page.height, tokens,
            max_tokens_for(profile, mode), page.rectified,
        )
        body = await self._post(payload)
        try:
            choice = body["choices"][0]
            output = choice["message"]["content"]
            finish_reason = choice.get("finish_reason")
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise UpstreamError("The OCR inference server returned an invalid response.") from exc

        parsed = parse_response(str(output))
        text, fields, warnings = parsed.text, parsed.fields, [*page.warnings, *parsed.warnings]
        markdown = parsed.markdown or text

        if finish_reason == "length":
            warnings.append(
                "The model reached its output limit before finishing this page, so the end of the "
                "transcription is missing. On an NID back this removes the MRZ, which is printed last."
            )

        confidence: dict[str, float] | None = None
        evidence: dict[str, Any] | None = None
        if mode.startswith("nid_"):
            # NID fields are never accepted directly from a VLM JSON response.
            # Every emitted value must be located and validated in its raw OCR.
            extraction = extract_nid(text, mode, parsed.layout)
            extraction = await self._reconciler.reconcile(extraction, page, mode)
            fields, confidence, evidence = extraction.fields, extraction.confidence, extraction.evidence
            warnings.extend(extraction.warnings)
            if not text.strip():
                warnings.append("The OCR model returned no transcription. Blank NID fields are supported, but no other card text was available to validate this page.")
            warnings.append("NID fields were derived from the single OCR transcription.")
        else:
            if mode == "schema" and profile.native_output == "html" and not fields:
                warnings.append("This model uses native HTML OCR; custom fields were derived deterministically from the single transcription.")
            fields = normalise_fields(fields, text, mode, schema)

        return PageResult(
            page_number=page.number,
            text=text,
            markdown=markdown,
            fields=fields,
            warnings=warnings,
            layout=[LayoutBlockModel(**block.as_dict()) for block in parsed.layout] if parsed.layout else None,
            field_confidence=confidence,
            field_evidence=evidence,
            finish_reason=finish_reason,
        )

    async def process(self, request_id: str, pages: list[RenderedPage], mode: str, schema: dict[str, Any] | None) -> OCRResult:
        started = time.perf_counter()
        runtime = await self.runtime()
        outcomes = await asyncio.gather(
            *(self._one_page(page, runtime.model, mode, schema) for page in pages),
            return_exceptions=True,
        )

        results: list[PageResult] = []
        failures: list[BaseException] = []
        for page, outcome in zip(pages, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                if isinstance(outcome, asyncio.CancelledError):
                    raise outcome
                # One unreadable page must not discard the pages that succeeded.
                failures.append(outcome)
                logger.warning("Page %s failed: %s", page.number, outcome)
                results.append(PageResult(
                    page_number=page.number, text="", markdown="", fields=None,
                    warnings=[*page.warnings, f"This page could not be processed: {outcome}"],
                ))
            else:
                results.append(outcome)

        if failures and len(failures) == len(pages):
            first = failures[0]
            raise first if isinstance(first, UpstreamError) else UpstreamError(str(first))

        metadata = OCRMetadata(
            request_id=request_id,
            model=runtime.model,
            serving_engine=runtime.engine,
            page_count=len(results),
            elapsed_ms=round((time.perf_counter() - started) * 1000),
            image_tokens=sum(
                image_token_cost(page.width or 0, page.height or 0,
                                 self.settings.vision_patch_size, self.settings.vision_merge_size)
                for page in pages
            ) or None,
            failed_pages=len(failures),
        )
        return OCRResult(result=results, metadata=metadata)
