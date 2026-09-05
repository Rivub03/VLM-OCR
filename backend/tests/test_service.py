import asyncio
from types import SimpleNamespace

from app.preprocess import RenderedPage
from app.service import OCRService


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload
        self.headers = {}

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self.payload


class FakeClient:
    posts = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def get(self, url):
        return FakeResponse({"data": [{"id": "dots-studio/dots.ocr"}]})

    async def post(self, url, json):
        self.__class__.posts += 1
        # A fabricated model field must not bypass NID text validation.
        return FakeResponse({"choices": [{"message": {"content": "Name\nMST. KOHINUR BEGUM\nDate of Birth 28 Oct 1983\nNID No. 370 809 0620"}}]})


def test_nid_uses_one_request_and_derives_fields_from_transcription(monkeypatch) -> None:
    async def scenario() -> None:
        FakeClient.posts = 0
        monkeypatch.setattr("app.service.httpx.AsyncClient", lambda **kwargs: FakeClient())
        settings = SimpleNamespace(
            max_inference_concurrency=1,
            inference_base_url="http://inference:8000",
            upstream_timeout_seconds=10,
        )
        result = await OCRService(settings).process("request", [RenderedPage(1, b"image")], "nid_front", None)
        assert FakeClient.posts == 1
        assert result.result[0].fields == {"name": "MST. KOHINUR BEGUM", "dob": "28 Oct 1983", "nid_no": "3708090620"}

    asyncio.run(scenario())
