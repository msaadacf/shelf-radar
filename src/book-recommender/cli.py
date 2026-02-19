from __future__ import annotations

import argparse

from .config import AppConfig
from .pipeline import run_pipeline


def main() -> None:
    p = argparse.ArgumentParser(
        description="Bookshelf OCR + recommendation demo using Qwen2-VL and Google Books."
    )
    p.add_argument(
        "--image",
        required=True,
        help="Path to a bookshelf image (e.g., assets/sample_bookshelf.jpg)",
    )
    p.add_argument(
        "--prefs",
        required=True,
        help="Free-text reading preferences (genres/themes/authors/pacing).",
    )
    p.add_argument(
        "--topn",
        type=int,
        default=3,
        help="Number of recommendations to print (default: 3).",
    )
    args = p.parse_args()

    cfg = AppConfig(top_n=args.topn)
    recs = run_pipeline(args.image, args.prefs, cfg)

    if not recs:
        print("No recommendations (no books resolved).")
        return

    for k, r in enumerate(recs, 1):
        authors = ", ".join(r.authors) if r.authors else "Unknown author"
        print(f"\n{k}) {r.title} — {authors}")
        print(f"   similarity: {r.similarity:.3f} | match_score: {r.match_score}")
        if r.categories:
            print(f"   categories: {', '.join(r.categories[:4])}")
        if r.average_rating is not None:
            rc = r.ratings_count if r.ratings_count is not None else "?"
            print(f"   rating: {r.average_rating} ({rc} ratings)")
        if r.info_link:
            print(f"   link: {r.info_link}")


if __name__ == "__main__":
    main()
