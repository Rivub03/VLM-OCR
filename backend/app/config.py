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
    max_page_dimension: int = 2048
    result_ttl_seconds: int = 3600
    upstream_timeout_seconds: float = 300.0
    nid_preprocess_enabled: bool = True
    nid_min_short_edge: int = 900
    nid_max_upscale: float = 4.0
    nid_border_px: int = 20
    nid_clahe_clip_limit: float = 2.0
    nid_clahe_tile_grid_size: int = 8
    nid_unsharp_amount: float = 0.20


@lru_cache
def get_settings() -> Settings:
    return Settings()
