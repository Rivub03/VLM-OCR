from fastapi import FastAPI

app = FastAPI()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def models():
    return {"data": [{"id": "datalab-to/surya-ocr-2"}]}


@app.post("/v1/chat/completions")
async def complete(payload: dict):
    prompt = payload["messages"][0]["content"][0]["text"]
    content = '{"text":"Mock OCR result", "fields":{"nid_no":"1234567890"}}' if "JSON" in prompt else "# Mock OCR result\n\nThis response came from the CPU-only test server."
    return {"choices": [{"message": {"content": content}}]}

