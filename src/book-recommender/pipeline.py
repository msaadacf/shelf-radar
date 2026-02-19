from __future__ import annotations

from typing import List

from .config import AppConfig
from .vlm_extract import load_qwen2_vl, extract_candidates
from .google_books import resolve_candidates
from .recommender import rank_books, Recommendation


def run_pipeline(image_path: str, preference_text: str, cfg: AppConfig) -> List[Recommendation]:
    model, processor = load_qwen2_vl(cfg.model_id)

    candidates = extract_candidates(
        image_path=image_path,
        model=model,
        processor=processor,
        max_items=cfg.max_candidates,
        max_new_tokens=cfg.max_new_tokens,
    )

    resolved = resolve_candidates(
        candidates=candidates,
        max_results=cfg.google_max_results,
        threshold=cfg.match_threshold,
    )

    recs = rank_books(
        books=resolved,
        preference_text=preference_text,
        top_n=cfg.top_n,
    )
    return recs
