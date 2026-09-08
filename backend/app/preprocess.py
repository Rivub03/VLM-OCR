import io
import logging
import math
from dataclasses import dataclass, field

import cv2
import fitz
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
SUPPORTED_TYPES = SUPPORTED_IMAGE_TYPES | {"application/pdf"}

# ISO/IEC 7810 ID-1 (the NID card format) is 85.60 x 53.98 mm.
ID1_ASPECT_RATIO = 85.60 / 53.98  # 1.5857
ID1_ASPECT_MIN = 1.45
ID1_ASPECT_MAX = 1.75
# A detected quadrilateral smaller than this fraction of the frame is background
# clutter rather than the card the photograph was taken of.
MIN_CARD_AREA_RATIO = 0.25


class DocumentError(ValueError):
    pass


@dataclass
class RenderedPage:
    number: int
    content: bytes
    media_type: str = "image/png"
    warnings: list[str] = field(default_factory=list)
    # Retained in-memory for the request only, so that a field which fails
    # validation can be re-read from the *enhanced* pixels without a second
    # upload or a second full-page model request.  Never persisted.
    width: int = 0
    height: int = 0
    rectified: bool = False


@dataclass(frozen=True)
class NidPreprocessOptions:
    enabled: bool = True
    rectify: bool = True
    illumination: bool = True
    deskew: bool = True
    target_long_edge: int = 1600
    min_short_edge: int = 900
    max_upscale: float = 4.0
    border_px: int = 20
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8
    unsharp_amount: float = 0.20


def image_token_cost(width: int, height: int, patch_size: int = 14, merge_size: int = 2) -> int:
    """Language tokens the vision encoder will emit for an image of this size.

    dots.ocr uses a Qwen2-VL image processor: the encoder splits the image into
    `patch_size` patches and then merges `merge_size` x `merge_size` of them into
    one language token, so one token covers a (patch_size * merge_size) square.
    The model's own preprocessor will happily build ~14k tokens from a large page
    (its `max_pixels` is 11289600), which overflows the served context and makes
    vLLM reject the whole request.  Sizing images against this figure is what
    keeps a page from failing outright.
    """
    cell = max(1, patch_size * merge_size)
    return math.ceil(width / cell) * math.ceil(height / cell)


def fit_to_token_budget(
    image: Image.Image,
    budget: int | None,
    patch_size: int = 14,
    merge_size: int = 2,
) -> tuple[Image.Image, bool]:
    """Shrink an image until its encoded token cost fits the context budget."""
    if not budget or budget <= 0:
        return image, False
    width, height = image.size
    if image_token_cost(width, height, patch_size, merge_size) <= budget:
        return image, False
    cell = max(1, patch_size * merge_size)
    # Solve for the pixel area that yields `budget` cells, then step down until
    # the per-axis ceilings agree.  The loop terminates because each pass shrinks
    # the longest edge by at least one cell.
    scale = math.sqrt((budget * cell * cell) / float(width * height))
    for _ in range(16):
        width = max(cell, int(image.width * scale))
        height = max(cell, int(image.height * scale))
        if image_token_cost(width, height, patch_size, merge_size) <= budget:
            break
        scale *= 0.97
    return image.resize((width, height), Image.Resampling.LANCZOS), True


def _order_quad(points: np.ndarray) -> np.ndarray:
    """Order four corners as top-left, top-right, bottom-right, bottom-left."""
    ordered = np.zeros((4, 2), dtype=np.float32)
    total = points.sum(axis=1)
    diff = np.diff(points, axis=1).ravel()
    ordered[0] = points[np.argmin(total)]
    ordered[2] = points[np.argmax(total)]
    ordered[1] = points[np.argmin(diff)]
    ordered[3] = points[np.argmax(diff)]
    return ordered


def _find_card_quad(bgr: np.ndarray) -> np.ndarray | None:
    """Locate the card's four corners, or return None when unsure.

    Detection runs on a downscaled copy so that a phone photograph costs a few
    milliseconds.  Returning None is a normal outcome: a flatbed scan has no
    card boundary to find, and guessing one would damage a good image.
    """
    height, width = bgr.shape[:2]
    longest = max(height, width)
    scale = 1000.0 / longest if longest > 1000 else 1.0
    small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else bgr
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = float(small.shape[0] * small.shape[1])
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        if cv2.contourArea(contour) < frame_area * MIN_CARD_AREA_RATIO:
            break
        perimeter = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approximation) == 4 and cv2.isContourConvex(approximation):
            return _order_quad(approximation.reshape(4, 2).astype(np.float32)) / scale
    return None


def _rectify_card(bgr: np.ndarray, target_long_edge: int) -> np.ndarray | None:
    """Flatten a photographed card, or return None when the quad is not card-shaped."""
    quad = _find_card_quad(bgr)
    if quad is None:
        return None
    top_left, top_right, bottom_right, bottom_left = quad
    edge_width = (np.linalg.norm(top_right - top_left) + np.linalg.norm(bottom_right - bottom_left)) / 2
    edge_height = (np.linalg.norm(bottom_left - top_left) + np.linalg.norm(bottom_right - top_right)) / 2
    if edge_width < 1 or edge_height < 1:
        return None
    aspect = edge_width / edge_height
    if ID1_ASPECT_MIN <= (1.0 / aspect) <= ID1_ASPECT_MAX:
        # The card was photographed on its side. Rotating the corner order warps
        # it straight to landscape in the same single operation.
        quad = np.roll(quad, 1, axis=0)
        aspect = 1.0 / aspect
    if not ID1_ASPECT_MIN <= aspect <= ID1_ASPECT_MAX:
        return None
    width = int(target_long_edge)
    height = int(round(target_long_edge / ID1_ASPECT_RATIO))
    destination = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), destination)
    return cv2.warpPerspective(bgr, matrix, (width, height), flags=cv2.INTER_LANCZOS4, borderValue=(255, 255, 255))


def _is_already_card_shaped(bgr: np.ndarray) -> bool:
    """True when the frame is the card itself, so there is nothing to rectify."""
    height, width = bgr.shape[:2]
    if height <= 0 or width <= 0:
        return False
    aspect = width / height
    return ID1_ASPECT_MIN <= aspect <= ID1_ASPECT_MAX or ID1_ASPECT_MIN <= 1 / aspect <= ID1_ASPECT_MAX


def _deskew(bgr: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    """Straighten a mildly rotated card using the dominant text angle."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    coordinates = cv2.findNonZero(mask)
    if coordinates is None:
        return bgr
    angle = cv2.minAreaRect(coordinates)[-1]
    if angle > 45:
        angle -= 90
    if abs(angle) < 0.2 or abs(angle) > max_angle:
        return bgr
    height, width = bgr.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(
        bgr, matrix, (width, height),
        flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE,
    )


def _flatten_illumination(bgr: np.ndarray) -> np.ndarray:
    """Remove shadow gradients and flatten specular glare on the L channel.

    Dividing lightness by a heavily smoothed estimate of its own background
    normalises uneven lighting across the card. A blown-out highlight ends up at
    the same level as clean laminate instead of dominating the contrast range,
    so CLAHE downstream amplifies real strokes rather than the glare edge.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    kernel_size = max(15, (min(lightness.shape) // 20) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    background = cv2.morphologyEx(lightness, cv2.MORPH_CLOSE, kernel)
    background = cv2.GaussianBlur(background, (0, 0), max(1.0, kernel_size / 3.0))
    background = np.maximum(background, 1).astype(np.float32)
    flattened = np.clip(lightness.astype(np.float32) / background * 200.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge((flattened, a_channel, b_channel)), cv2.COLOR_LAB2BGR)


def _resize_for_ocr(bgr: np.ndarray, target_long_edge: int, max_dimension: int, max_upscale: float) -> np.ndarray:
    """Scale toward the OCR working resolution with text-preserving resampling."""
    height, width = bgr.shape[:2]
    longest = max(height, width)
    if longest <= 0:
        return bgr
    scale = min(target_long_edge / longest, max_upscale, max_dimension / longest)
    if abs(scale - 1.0) < 0.02:
        return bgr
    interpolation = cv2.INTER_LANCZOS4 if scale > 1.0 else cv2.INTER_AREA
    return cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=interpolation)


def _enhance_nid(image: Image.Image, max_dimension: int, options: NidPreprocessOptions) -> tuple[Image.Image, list[str], bool]:
    """Enhance one NID image locally without creating another OCR view.

    Every stage degrades to a pass-through so that a difficult photograph still
    reaches the model.  The chain order matters: geometry is corrected before
    lighting, and lighting before contrast, so each stage sees the output the
    next one expects.
    """
    bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    warnings: list[str] = []
    rectified = False

    if options.rectify:
        try:
            warped = _rectify_card(bgr, options.target_long_edge)
        except cv2.error:
            warped = None
        if warped is not None:
            bgr, rectified = warped, True
        elif _is_already_card_shaped(bgr):
            # A cropped upload or flatbed scan is the card and nothing else, so
            # there is no outline to find and nothing to correct. Warning about
            # it would be noise on the most common well-formed input.
            pass
        else:
            warnings.append(
                "No ID-1 card outline was detected, so the image was enhanced without perspective correction. "
                "Photographing the whole card against a contrasting background improves accuracy."
            )

    if options.deskew and not rectified:
        try:
            bgr = _deskew(bgr)
        except cv2.error:
            warnings.append("Deskew was unavailable for this image.")

    if options.illumination:
        try:
            bgr = _flatten_illumination(bgr)
        except cv2.error:
            warnings.append("Illumination flattening was unavailable for this image.")

    if not rectified:
        bgr = _resize_for_ocr(bgr, options.target_long_edge, max_dimension, options.max_upscale)

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
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), warnings, rectified


def _normalise_image(
    content: bytes,
    max_dimension: int,
    nid_mode: bool = False,
    nid_options: NidPreprocessOptions | None = None,
    token_budget: int | None = None,
    patch_size: int = 14,
    merge_size: int = 2,
) -> tuple[bytes, list[str], tuple[int, int], bool]:
    try:
        with Image.open(io.BytesIO(content)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            if max(image.size) > max_dimension:
                image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
            warnings: list[str] = []
            rectified = False
            options = nid_options or NidPreprocessOptions()
            if nid_mode and options.enabled:
                try:
                    image, enhancement_warnings, rectified = _enhance_nid(image, max_dimension, options)
                    warnings.extend(enhancement_warnings)
                except Exception:
                    logger.warning("NID enhancement failed; falling back to standard normalization", exc_info=True)
                    warnings.append("NID contrast enhancement was unavailable; standard image normalization was used.")
            image, shrunk = fit_to_token_budget(image, token_budget, patch_size, merge_size)
            if shrunk:
                warnings.append(
                    "The page was reduced so its encoded size fits the model's context window. "
                    "Very large pages may lose fine detail."
                )
            output = io.BytesIO()
            image.save(output, format="PNG", optimize=True)
            return output.getvalue(), warnings, image.size, rectified
    except (UnidentifiedImageError, OSError) as exc:
        raise DocumentError("The uploaded image is corrupt or unsupported.") from exc


def render_document(
    content: bytes,
    content_type: str | None,
    max_pages: int,
    max_dimension: int,
    nid_mode: bool = False,
    nid_options: NidPreprocessOptions | None = None,
    token_budget: int | None = None,
    patch_size: int = 14,
    merge_size: int = 2,
    pdf_dpi: int = 200,
) -> list[RenderedPage]:
    if not content:
        raise DocumentError("The uploaded file is empty.")
    content_type = (content_type or "").split(";", 1)[0].lower()
    if content_type not in SUPPORTED_TYPES:
        raise DocumentError("Use a PDF, JPEG, PNG, or WEBP document.")
    if content_type != "application/pdf":
        normalised, warnings, size, rectified = _normalise_image(
            content, max_dimension, nid_mode, nid_options, token_budget, patch_size, merge_size,
        )
        return [RenderedPage(1, normalised, warnings=warnings, width=size[0], height=size[1], rectified=rectified)]

    try:
        document = fitz.open(stream=content, filetype="pdf")
    except (fitz.FileDataError, RuntimeError) as exc:
        raise DocumentError("The uploaded PDF cannot be opened.") from exc
    try:
        if document.page_count == 0:
            raise DocumentError("The PDF has no pages.")
        if document.page_count > max_pages:
            raise DocumentError(f"PDFs are limited to {max_pages} pages.")
        # PDF user space is 72 dpi, so the zoom factor is the target dpi over 72.
        # A fixed 2x matrix rendered every document at 144 dpi, which is thin for
        # small print; the token budget below caps the result either way.
        zoom = max(1.0, pdf_dpi / 72.0)
        pages: list[RenderedPage] = []
        for index, page in enumerate(document):
            pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            png = pixmap.tobytes("png")
            normalised, warnings, size, rectified = _normalise_image(
                png, max_dimension, nid_mode, nid_options, token_budget, patch_size, merge_size,
            )
            pages.append(RenderedPage(index + 1, normalised, warnings=warnings, width=size[0], height=size[1], rectified=rectified))
        return pages
    finally:
        document.close()
