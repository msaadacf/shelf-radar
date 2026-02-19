from dataclasses import dataclass

@dataclass(frozen=True)
class AppConfig:
    model_id: str = "Qwen/Qwen2-VL-2B-Instruct"
    max_candidates: int = 30
    max_new_tokens: int = 600
    google_max_results: int = 5
    match_threshold: int = 70
    top_n: int = 5
