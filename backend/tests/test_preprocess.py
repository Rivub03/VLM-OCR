import io

import fitz
import numpy as np
from PIL import Image

from app.preprocess import (
    DocumentError,
    NidPreprocessOptions,
    fit_to_token_budget,
    image_token_cost,
    render_document,
)


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
        # Sized toward the 1600px working resolution (capped by max_upscale=4),
        # plus the 20px white border on each edge.
        assert result.size == (1640, 1240)
        assert pages[0].width == 1640


def test_non_nid_image_does_not_receive_nid_border_or_clahe() -> None:
    output = io.BytesIO()
    Image.new("RGB", (400, 300), "white").save(output, format="PNG")
    pages = render_document(output.getvalue(), "image/png", max_pages=2, max_dimension=2048, nid_mode=False)
    with Image.open(io.BytesIO(pages[0].content)) as result:
        assert result.size == (400, 300)


def test_nid_enhancement_falls_back_to_standard_normalization(monkeypatch) -> None:
    output = io.BytesIO()
    Image.new("RGB", (400, 300), "white").save(output, format="PNG")
    monkeypatch.setattr("app.preprocess._enhance_nid", lambda *args: (_ for _ in ()).throw(RuntimeError("CV failure")))
    pages = render_document(
        output.getvalue(), "image/png", max_pages=2, max_dimension=2048, nid_mode=True, nid_options=NidPreprocessOptions(),
    )
    assert pages[0].warnings
    assert pages[0].content.startswith(b"\x89PNG")


def test_pdf_is_rasterized_per_page() -> None:
    document = fitz.open()
    document.new_page()
    document.new_page()
    pages = render_document(document.tobytes(), "application/pdf", max_pages=2, max_dimension=256)
    assert [page.number for page in pages] == [1, 2]


def test_image_token_cost_matches_the_vision_encoder_geometry() -> None:
    """One token per 28x28 cell: patch size 14 with a 2x2 spatial merge."""
    assert image_token_cost(28, 28) == 1
    assert image_token_cost(2048, 1248) == 74 * 45
    # Partial cells still consume a whole token.
    assert image_token_cost(29, 29) == 4


def test_oversized_pages_are_shrunk_to_fit_the_context_window() -> None:
    """An unbounded image overflows max_model_len and vLLM rejects the page."""
    image = Image.new("RGB", (4000, 4000), "white")
    fitted, shrunk = fit_to_token_budget(image, budget=2000)
    assert shrunk
    assert image_token_cost(*fitted.size) <= 2000
    untouched, unchanged = fit_to_token_budget(Image.new("RGB", (280, 280)), budget=2000)
    assert not unchanged and untouched.size == (280, 280)


def test_token_budget_is_applied_end_to_end_with_a_warning() -> None:
    output = io.BytesIO()
    Image.new("RGB", (2000, 2000), "white").save(output, format="PNG")
    pages = render_document(
        output.getvalue(), "image/png", max_pages=2, max_dimension=2560, token_budget=1000,
    )
    assert image_token_cost(pages[0].width, pages[0].height) <= 1000
    assert any("context window" in warning for warning in pages[0].warnings)


def _card_photo() -> bytes:
    """A synthetic ID-1 card photographed at an angle on a dark surface."""
    import cv2

    card = np.full((540, 856, 3), 250, dtype=np.uint8)
    cv2.putText(card, "NID No. 911 616 1184", (40, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    scene = np.full((1200, 1600, 3), 30, dtype=np.uint8)
    source = np.float32([[0, 0], [856, 0], [856, 540], [0, 540]])
    destination = np.float32([[280, 210], [1330, 300], [1270, 940], [220, 840]])
    warped = cv2.warpPerspective(card, cv2.getPerspectiveTransform(source, destination), (1600, 1200))
    mask = cv2.warpPerspective(np.ones((540, 856, 3), np.uint8), cv2.getPerspectiveTransform(source, destination), (1600, 1200))
    scene = np.where(mask > 0, warped, scene)
    return cv2.imencode(".png", scene)[1].tobytes()


def test_a_photographed_card_is_rectified_to_id1_proportions() -> None:
    pages = render_document(
        _card_photo(), "image/png", max_pages=1, max_dimension=2560, nid_mode=True,
        nid_options=NidPreprocessOptions(target_long_edge=1600),
    )
    assert pages[0].rectified
    with Image.open(io.BytesIO(pages[0].content)) as result:
        width, height = result.size
    # 1600x1009 warp plus the 20px border on each edge.
    assert (width, height) == (1640, 1049)
    assert not any("No ID-1 card outline" in warning for warning in pages[0].warnings)


def test_a_photo_with_no_detectable_card_is_not_warped_and_says_so() -> None:
    """Guessing a quad on an image that has none would damage a good scan."""
    output = io.BytesIO()
    Image.new("RGB", (900, 900), "white").save(output, format="PNG")
    pages = render_document(
        output.getvalue(), "image/png", max_pages=1, max_dimension=2560, nid_mode=True,
    )
    assert not pages[0].rectified
    assert any("No ID-1 card outline" in warning for warning in pages[0].warnings)
    assert pages[0].content.startswith(b"\x89PNG")


def test_an_already_cropped_card_is_not_warned_about() -> None:
    """A scan or cropped upload is the card, so there is nothing to correct."""
    output = io.BytesIO()
    Image.new("RGB", (1010, 636), "white").save(output, format="PNG")  # ID-1 proportions
    pages = render_document(
        output.getvalue(), "image/png", max_pages=1, max_dimension=2560, nid_mode=True,
    )
    assert not any("No ID-1 card outline" in warning for warning in pages[0].warnings)


def test_rectification_can_be_disabled_without_losing_the_rest() -> None:
    pages = render_document(
        _card_photo(), "image/png", max_pages=1, max_dimension=2560, nid_mode=True,
        nid_options=NidPreprocessOptions(rectify=False),
    )
    assert not pages[0].rectified
    assert pages[0].content.startswith(b"\x89PNG")


def test_pdf_is_rendered_at_the_requested_dpi() -> None:
    document = fitz.open()
    document.new_page(width=612, height=792)  # US Letter at 72 dpi
    pages = render_document(document.tobytes(), "application/pdf", max_pages=1, max_dimension=4000, pdf_dpi=200)
    # 612pt at 200 dpi is 1700px, versus 1224px under the old fixed 2x matrix.
    assert pages[0].width == 1700


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
