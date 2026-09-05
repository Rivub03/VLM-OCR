from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(case_sensitive=False)

    ocr_api_key: str
    frontend_origin: str = "http://localhost:3000"
    inference_base_url: str = "http://inference:8000"
    max_inference_concurrency: int = 4
    max_upload_mib: int = 25
    max_pdf_pages: int = 20
    max_page_dimension: int = 2560
    max_pdf_dpi: int = 200
    result_ttl_seconds: int = 3600
    upstream_timeout_seconds: float = 300.0

    # The served model's context window.  It must track OCR_MAX_MODEL_LEN in
    # inference/entrypoint.sh: the backend sizes images so that image tokens plus
    # generated tokens cannot exceed it.  Drift between the two silently
    # reintroduces the upstream 400 that fails a whole page.
    max_model_len: int = 24576
    # Reserve room for the generated tokens, the prompt, the chat template, and
    # vLLM's own bookkeeping before deciding how many image tokens a page may
    # cost. It must exceed the largest per-mode max_tokens in profiles.py.
    token_budget_margin: int = 6144
    # dots.ocr / Qwen2-VL vision geometry: a 14px patch with a 2x2 spatial merge
    # makes one language token per 28x28 pixel cell.  Sourced from the model's
    # preprocessor_config.json (patch_size 14, merge_size 2).
    vision_patch_size: int = 14
    vision_merge_size: int = 2

    upstream_max_attempts: int = 3
    upstream_retry_backoff_seconds: float = 0.5

    # dots.ocr's layout task returns categories, HTML tables and LaTeX formulas;
    # its plain OCR task returns an unstructured string. Layout is the default
    # for documents, where that structure is the point.
    dots_layout_prompt_enabled: bool = True
    # Cards are different. Measured on the benchmark's NID front train split,
    # the layout task collapses a card into run-on lines and loops to the token
    # limit on dense ones, so identity cards use the plain OCR task. Enable this
    # only to reproduce that comparison.
    nid_layout_prompt_enabled: bool = False
    # Last-resort name recovery from the card's fixed layout when no printed
    # label anchored a candidate. Weaker evidence than a label, so it is opt-in
    # and reported with a lower confidence and a distinct evidence source.
    nid_name_structural_fallback: bool = False
    # Anti-repetition sampling, tuned per document kind rather than globally.
    #
    # Cards loop far more readily than documents. On the NID front train split,
    # raising the penalty took pages that ran to the token limit from 14/40 to
    # 0/40 and overall accuracy from 86.7% to 90.8%.
    #
    # The back deliberately keeps the low value. Its MRZ contains long runs of
    # `<` filler that are *legitimately* repetitive, and a token-level penalty
    # shortens them: at 1.30 an observed filler run dropped from 9 characters to
    # 6. There is no NID back ground truth to tune against, so it stays
    # conservative and the MRZ check digits catch what the model gets wrong.
    repetition_penalty: float = 1.05
    nid_front_repetition_penalty: float = 1.30
    nid_back_repetition_penalty: float = 1.05
    nid_temperature: float = 0.0

    nid_preprocess_enabled: bool = True
    nid_rectify_enabled: bool = True
    nid_illumination_enabled: bool = True
    nid_deskew_enabled: bool = True
    nid_target_long_edge: int = 1600
    nid_min_short_edge: int = 900
    nid_max_upscale: float = 4.0
    nid_border_px: int = 20
    nid_clahe_clip_limit: float = 2.0
    nid_clahe_tile_grid_size: int = 8
    nid_unsharp_amount: float = 0.20

    # CPU-only reconciliation engine.  It never loads a model onto the GPU; it
    # re-reads only the crops whose fields failed local validation.
    nid_verify_enabled: bool = True
    nid_verify_max_workers: int = 2
    # A second dots.ocr grounding request for still-failing fields. Off until a
    # benchmark shows the extra latency is earned.
    nid_second_pass_enabled: bool = False

    @property
    def image_token_budget(self) -> int:
        """Image tokens a single page may cost before generation is accounted for."""
        return max(256, self.max_model_len - self.token_budget_margin)


@lru_cache
def get_settings() -> Settings:
    return Settings()
