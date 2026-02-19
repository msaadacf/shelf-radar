# shelf-radar

## Goal
Extract book titles from a bookshelf photo using an open source vision language model, resolve them to canonical metadata using Google Books, then rank the detected books against a user’s text reading preferences using embedding similarity.

## Pipeline
1. Load an image containing a lot of books
2. VLM extracts book titles as JSON
3. Normalize, deduplicate book candidates
4. Resolve candidates via Google Books search AND fuzzy matching
5. Then embed each resolved book (title/authors/categories/description)
6. Embed user preference text (you can add your own preferences here if you wish to do so)
7. Rank by cosine similarity and display the books that you are most likely to enjoy




#### Why I chose a Vision Language Model (VLM) instead of classical OCR?
Book pictures commonly have glare (biggest issue), curved text, partial occlusion, stylized fonts, non-horizontal orientation, and missing words. Since I wanted to ship something really quick and useful, using a classical OCR was not the right option because it often returns fragmented tokens with low recall. A VLM can directly infer complete titles from partial visual cues and return structured results suitable for resolution later in the pipeline.

#### Why Google Books resolution?
Text extracted from images often comes out noisy and ambiguous. If you restrict text too much, failure points increase. So, Google Books provides canonical titles and authors plus descriptions and categories of the given book queries. This turns the not-so-perfect spine text into clean metadata that allows for better ranking. Moreover, Google Books also offers clickable references for books, so that's great!

#### Why cosine similarity instead of training a recommender?
A trained recommender needs user item interaction logs (ratings, clicks, dwell time) and an evaluation loop. For a fast working demo of this project's concept, embeddings with cosine similarity provide good relevance with minimal infrastructure, is deterministic, and is easy to explain and reproduce. However, if this project is to be scaled at a massive level, training a recommender would definitely be the right choice.

## Run
```bash
pip install -r requirements.txt
python -m bookshelf_recommender.cli --image assets/sample_bookshelf.jpg --prefs "literary fiction, Paris, slow-paced, character-driven" --topn 3
```
## How to include the sample bookshelf image
- Put one image in the `assets/` folder (keep it reasonably sized, e.g., < 2–5 MB).
- If the image is large, git will still handle it; avoid multiple huge images.

## What to keep as notebook
Keep `notebooks/demo.ipynb` as a thin wrapper:
- one cell for installs (optional),
- one cell that calls the pipeline functions,
- one cell that prints results.
The `.py` package remains the canonical implementation.










