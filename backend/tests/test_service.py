import asyncio
from types import SimpleNamespace

import httpx
import pytest

from app.preprocess import RenderedPage
from app.profiles import DOTS_PROMPT_LAYOUT_ALL_EN, DOTS_PROMPT_OCR
from app.service import OCRService, UpstreamError


def settings(**overrides) -> SimpleNamespace:
    base = dict(
        max_inference_concurrency=1,
        inference_base_url="http://inference:8000",
        upstream_timeout_seconds=10,
        upstream_max_attempts=3,
        upstream_retry_backoff_seconds=0.0,
        dots_layout_prompt_enabled=True,
        nid_layout_prompt_enabled=False,
        repetition_penalty=1.05,
        nid_front_repetition_penalty=1.30,
        nid_back_repetition_penalty=1.05,
        nid_temperature=0.0,
        vision_patch_size=14,
        vision_merge_size=2,
        nid_verify_enabled=False,
        nid_second_pass_enabled=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload
        self.headers = {}

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self.payload


class FakeClient:
    """Stands in for the pooled httpx client held by OCRService."""

    def __init__(self, content="", *, fail_pages=(), finish_reason="stop"):
        self.posts = 0
        self.content = content
        self.fail_pages = set(fail_pages)
        self.finish_reason = finish_reason

    async def get(self, url, **kwargs):
        return FakeResponse({"data": [{"id": "dots-studio/dots.ocr"}]})

    async def post(self, url, json=None, **kwargs):
        self.posts += 1
        if self.posts in self.fail_pages:
            raise httpx.ConnectError("boom")
        return FakeResponse({"choices": [{
            "message": {"content": self.content},
            "finish_reason": self.finish_reason,
        }]})

    async def aclose(self):
        return None


def build(client, **overrides) -> OCRService:
    service = OCRService(settings(**overrides))
    service._client = client
    return service


NID_FRONT_TEXT = "Name\nMST. KOHINUR BEGUM\nDate of Birth 28 Oct 1983\nNID No. 370 809 0620"


def test_nid_uses_one_request_and_derives_fields_from_transcription() -> None:
    async def scenario() -> None:
        # A fabricated model field must not bypass NID text validation.
        client = FakeClient(NID_FRONT_TEXT)
        result = await build(client).process("request", [RenderedPage(1, b"image")], "nid_front", None)
        assert client.posts == 1
        assert result.result[0].fields == {
            "name": "MST. KOHINUR BEGUM", "dob": "28 Oct 1983", "nid_no": "3708090620",
        }
        assert result.result[0].field_confidence["name"] > 0

    asyncio.run(scenario())


def test_one_failing_page_does_not_discard_the_pages_that_succeeded() -> None:
    """A single unreadable page used to fail the whole multi-page document."""
    async def scenario() -> None:
        client = FakeClient(NID_FRONT_TEXT, fail_pages={2})
        pages = [RenderedPage(index, b"image") for index in (1, 2, 3)]
        result = await build(client, upstream_max_attempts=1).process("request", pages, "text", None)
        assert [page.page_number for page in result.result] == [1, 2, 3]
        assert result.metadata.failed_pages == 1
        assert result.result[0].text and result.result[2].text
        assert result.result[1].text == ""
        assert any("could not be processed" in warning for warning in result.result[1].warnings)

    asyncio.run(scenario())


def test_every_page_failing_still_raises() -> None:
    async def scenario() -> None:
        client = FakeClient(NID_FRONT_TEXT, fail_pages={1, 2})
        pages = [RenderedPage(index, b"image") for index in (1, 2)]
        with pytest.raises(UpstreamError):
            await build(client, upstream_max_attempts=1).process("request", pages, "text", None)

    asyncio.run(scenario())


def test_transient_failures_are_retried_and_client_errors_are_not() -> None:
    async def scenario() -> None:
        transient = FakeClient(NID_FRONT_TEXT, fail_pages={1, 2})
        result = await build(transient).process("request", [RenderedPage(1, b"image")], "text", None)
        assert transient.posts == 3
        assert result.result[0].text

        class RejectingClient(FakeClient):
            async def post(self, url, json=None, **kwargs):
                self.posts += 1
                response = httpx.Response(400, text="bad request", request=httpx.Request("POST", url))
                raise httpx.HTTPStatusError("bad", request=response.request, response=response)

        rejecting = RejectingClient()
        with pytest.raises(UpstreamError) as caught:
            await build(rejecting).process("request", [RenderedPage(1, b"image")], "text", None)
        # A contract error repeats identically; retrying it only wastes time.
        assert rejecting.posts == 1
        assert caught.value.status_code == 422

    asyncio.run(scenario())


def test_cards_use_the_plain_ocr_task_while_documents_use_layout() -> None:
    """Measured on the NID front train split: layout degrades cards badly."""
    async def scenario() -> None:
        sent: list[str] = []

        class Recording(FakeClient):
            async def post(self, url, json=None, **kwargs):
                sent.append(json["messages"][0]["content"][0]["text"])
                return await super().post(url, json=json, **kwargs)

        client = Recording(NID_FRONT_TEXT)
        service = build(client)
        await service.process("r", [RenderedPage(1, b"i")], "nid_front", None)
        await service.process("r", [RenderedPage(1, b"i")], "text", None)
        assert sent[0] == DOTS_PROMPT_OCR
        assert sent[1] == DOTS_PROMPT_LAYOUT_ALL_EN

    asyncio.run(scenario())


def test_truncated_output_is_reported_rather_than_silently_returned() -> None:
    async def scenario() -> None:
        client = FakeClient(NID_FRONT_TEXT, finish_reason="length")
        result = await build(client).process("request", [RenderedPage(1, b"image")], "nid_back", None)
        assert result.result[0].finish_reason == "length"
        assert any("output limit" in warning for warning in result.result[0].warnings)

    asyncio.run(scenario())


def test_layout_response_populates_the_layout_field() -> None:
    async def scenario() -> None:
        content = (
            '[{"bbox": [0, 0, 100, 20], "category": "Title", "text": "Invoice"},'
            ' {"bbox": [0, 30, 200, 90], "category": "Table", "text": "<table><tr><td>A</td></tr></table>"},'
            ' {"bbox": [0, 100, 50, 150], "category": "Picture"}]'
        )
        result = await build(FakeClient(content)).process("request", [RenderedPage(1, b"image")], "text", None)
        page = result.result[0]
        assert [block.category for block in page.layout] == ["Title", "Table", "Picture"]
        assert "# Invoice" in page.markdown
        assert "<table>" in page.markdown
        assert "Invoice" in page.text

    asyncio.run(scenario())
