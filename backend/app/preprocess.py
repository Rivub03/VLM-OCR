import io
from dataclasses import dataclass

import fitz
from PIL import Image, ImageOps, UnidentifiedImageError


SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
SUPPORTED_TYPES = SUPPORTED_IMAGE_TYPES | {"application/pdf"}


class DocumentError(ValueError):
    pass


@dataclass
class RenderedPage:
    number: int
    content: bytes
    media_type: str = "image/png"


def _normalise_image(content: bytes, max_dimension: int) -> bytes:
    try:
        with Image.open(io.BytesIO(content)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            if max(image.size) > max_dimension:
                image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
            output = io.BytesIO()
            image.save(output, format="PNG", optimize=True)
            return output.getvalue()
    except (UnidentifiedImageError, OSError) as exc:
        raise DocumentError("The uploaded image is corrupt or unsupported.") from exc


def render_document(content: bytes, content_type: str | None, max_pages: int, max_dimension: int) -> list[RenderedPage]:
    if not content:
        raise DocumentError("The uploaded file is empty.")
    content_type = (content_type or "").split(";", 1)[0].lower()
    if content_type not in SUPPORTED_TYPES:
        raise DocumentError("Use a PDF, JPEG, PNG, or WEBP document.")
    if content_type != "application/pdf":
        return [RenderedPage(number=1, content=_normalise_image(content, max_dimension))]

    try:
        document = fitz.open(stream=content, filetype="pdf")
    except (fitz.FileDataError, RuntimeError) as exc:
        raise DocumentError("The uploaded PDF cannot be opened.") from exc
    try:
        if document.page_count == 0:
            raise DocumentError("The PDF has no pages.")
        if document.page_count > max_pages:
            raise DocumentError(f"PDFs are limited to {max_pages} pages.")
        pages: list[RenderedPage] = []
        for index, page in enumerate(document):
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            png = pixmap.tobytes("png")
            pages.append(RenderedPage(index + 1, _normalise_image(png, max_dimension)))
        return pages
    finally:
        document.close()

