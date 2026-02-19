from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
from rapidfuzz import fuzz


@dataclass(frozen=True)
class ResolvedBook:
    query_title: str
    match_score: int
    title: str
    authors: List[str]
    categories: List[str]
    description: str
    published_date: str
    average_rating: Optional[float]
    ratings_count: Optional[int]
    info_link: Optional[str]


def normalize_title(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s:,'-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def google_books_search(title: str, author: Optional[str], max_results: int) -> List[Dict[str, Any]]:
    q = f'intitle:"{title}"'
    if author:
        q += f'+inauthor:"{author}"'
    url = "https://www.googleapis.com/books/v1/volumes"
    r = requests.get(url, params={"q": q, "maxResults": max_results, "printType": "books"}, timeout=20)
    r.raise_for_status()
    return r.json().get("items", []) or []


def pick_best_match(title_norm: str, items: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], int]:
    best = None
    best_score = -1
    for it in items:
        vi = it.get("volumeInfo", {}) or {}
        t = normalize_title(vi.get("title", "") or "")
        score = int(fuzz.token_set_ratio(title_norm, t))
        if score > best_score:
            best_score = score
            best = it
    return best, best_score


def resolve_candidates(
    candidates,
    max_results: int = 5,
    threshold: int = 70,
) -> List[ResolvedBook]:
    resolved: List[ResolvedBook] = []

    for c in candidates:
        items = google_books_search(c.title, c.author, max_results=max_results)
        best, score = pick_best_match(normalize_title(c.title), items)
        if not best or score < threshold:
            continue

        vi = best.get("volumeInfo", {}) or {}
        resolved.append(
            ResolvedBook(
                query_title=c.title,
                match_score=score,
                title=vi.get("title", "") or "",
                authors=vi.get("authors", []) or [],
                categories=vi.get("categories", []) or [],
                description=vi.get("description", "") or "",
                published_date=vi.get("publishedDate", "") or "",
                average_rating=vi.get("averageRating", None),
                ratings_count=vi.get("ratingsCount", None),
                info_link=vi.get("infoLink", None),
            )
        )

    return resolved
