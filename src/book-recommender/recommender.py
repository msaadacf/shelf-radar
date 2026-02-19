from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from .google_books import ResolvedBook


@dataclass(frozen=True)
class Recommendation:
    title: str
    authors: List[str]
    categories: List[str]
    similarity: float
    match_score: int
    average_rating: float | None
    ratings_count: int | None
    info_link: str | None


def _book_text(b: ResolvedBook) -> str:
    parts = [
        b.title,
        " ".join(b.authors),
        " ".join(b.categories),
        b.description,
    ]
    return "\n".join([p for p in parts if p])


def rank_books(
    books: List[ResolvedBook],
    preference_text: str,
    top_n: int = 5,
    embedder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> List[Recommendation]:
    if not books:
        return []

    embedder = SentenceTransformer(embedder_name)

    book_texts = [_book_text(b) for b in books]
    book_vecs = embedder.encode(book_texts, normalize_embeddings=True)
    pref_vec = embedder.encode([preference_text], normalize_embeddings=True)[0]

    sims = book_vecs @ pref_vec
    idx = np.argsort(-sims)

    out: List[Recommendation] = []
    for i in idx[:top_n]:
        b = books[int(i)]
        out.append(
            Recommendation(
                title=b.title,
                authors=b.authors,
                categories=b.categories,
                similarity=float(sims[int(i)]),
                match_score=b.match_score,
                average_rating=b.average_rating,
                ratings_count=b.ratings_count,
                info_link=b.info_link,
            )
        )
    return out
