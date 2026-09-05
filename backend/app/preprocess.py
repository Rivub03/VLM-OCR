import io
from dataclasses import dataclass, field

import cv2
import fitz
import numpy as np
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
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class NidPreprocessOptions:
    enabled: bool = True
    min_short_edge: int = 900
    max_upscale: float = 4.0
    border_px: int = 20
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8
    unsharp_amount: float = 0.20


def _enhance_nid(image: Image.Image, max_dimension: int, options: NidPreprocessOptions) -> Image.Image:
    """Enhance one NID image locally without creating another OCR view."""
    bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    height, width = bgr.shape[:2]
    if min(height, width) < options.min_short_edge:
        scale = min(options.max_upscale, options.min_short_edge / min(height, width), max_dimension / max(height, width))
        if scale > 1.0:
            bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if options.border_px:
        bgr = cv2.copyMakeBorder(
            bgr, options.border_px, options.border_px, options.border_px, options.border_px,
            cv2.BORDER_CONSTANT, value=(255, 255, 255),
        )
    height, width = bgr.shape[:2]
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=options.clahe_clip_limit,
        tileGridSize=(options.clahe_tile_grid_size, options.clahe_tile_grid_size),
    )
    bgr = cv2.cvtColor(cv2.merge((clahe.apply(lightness), a_channel, b_channel)), cv2.COLOR_LAB2BGR)
    if options.unsharp_amount:
        blurred = cv2.GaussianBlur(bgr, (0, 0), 1.0)
        bgr = cv2.addWeighted(bgr, 1.0 + options.unsharp_amount, blurred, -options.unsharp_amount, 0)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _normalise_image(
    content: bytes,
    max_dimension: int,
    nid_mode: bool = False,
    nid_options: NidPreprocessOptions | None = None,
) -> tuple[bytes, list[str]]:
    try:
        with Image.open(io.BytesIO(content)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            if max(image.size) > max_dimension:
                image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
            warnings: list[str] = []
            options = nid_options or NidPreprocessOptions()
            if nid_mode and options.enabled:
                try:
                    image = _enhance_nid(image, max_dimension, options)
                except Exception:
                    warnings.append("NID contrast enhancement was unavailable; standard image normalization was used.")
            output = io.BytesIO()
            image.save(output, format="PNG", optimize=True)
            return output.getvalue(), warnings
    except (UnidentifiedImageError, OSError) as exc:
        raise DocumentError("The uploaded image is corrupt or unsupported.") from exc


def render_document(
    content: bytes,
    content_type: str | None,
    max_pages: int,
    max_dimension: int,
    nid_mode: bool = False,
    nid_options: NidPreprocessOptions | None = None,
) -> list[RenderedPage]:
    if not content:
        raise DocumentError("The uploaded file is empty.")
    content_type = (content_type or "").split(";", 1)[0].lower()
    if content_type not in SUPPORTED_TYPES:
        raise DocumentError("Use a PDF, JPEG, PNG, or WEBP document.")
    if content_type != "application/pdf":
        normalised, warnings = _normalise_image(content, max_dimension, nid_mode, nid_options)
        return [RenderedPage(number=1, content=normalised, warnings=warnings)]

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
            normalised, warnings = _normalise_image(png, max_dimension, nid_mode, nid_options)
            pages.append(RenderedPage(index + 1, normalised, warnings=warnings))
        return pages
    finally:
        document.close()
