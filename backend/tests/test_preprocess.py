import io

import fitz
from PIL import Image

from app.preprocess import DocumentError, render_document


def image_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (20, 20), "white").save(output, format="PNG")
    return output.getvalue()


def test_image_is_normalised_to_png() -> None:
    pages = render_document(image_bytes(), "image/png", max_pages=2, max_dimension=128)
    assert len(pages) == 1
    assert pages[0].content.startswith(b"\x89PNG")


def test_image_is_downscaled_to_dimension_limit() -> None:
    output = io.BytesIO()
    Image.new("RGB", (4000, 1000), "white").save(output, format="PNG")
    pages = render_document(output.getvalue(), "image/png", max_pages=2, max_dimension=2048)
    with Image.open(io.BytesIO(pages[0].content)) as result:
        assert result.size == (2048, 512)


def test_small_nid_image_is_upscaled_for_the_vision_encoder() -> None:
    output = io.BytesIO()
    Image.new("RGB", (400, 300), "white").save(output, format="PNG")
    pages = render_document(output.getvalue(), "image/png", max_pages=2, max_dimension=2048, nid_mode=True)
    with Image.open(io.BytesIO(pages[0].content)) as result:
        assert result.size == (1200, 900)


def test_pdf_is_rasterized_per_page() -> None:
    document = fitz.open()
    document.new_page()
    document.new_page()
    pages = render_document(document.tobytes(), "application/pdf", max_pages=2, max_dimension=256)
    assert [page.number for page in pages] == [1, 2]


def test_pdf_page_limit_is_enforced() -> None:
    document = fitz.open()
    document.new_page()
    document.new_page()
    try:
        render_document(document.tobytes(), "application/pdf", max_pages=1, max_dimension=256)
    except DocumentError as exc:
        assert "limited" in str(exc)
    else:
        raise AssertionError("Expected a page-limit error")
