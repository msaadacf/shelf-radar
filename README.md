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

## Run
```bash
pip install -r requirements.txt
python -m bookshelf_recommender.cli --image assets/sample_bookshelf.jpg --prefs "literary fiction, Paris, slow-paced, character-driven" --topn 5

## How to include the sample bookshelf image
- Put one image in the `assets/` folder (keep it reasonably sized, e.g., < 2–5 MB).
- If the image is large, git will still handle it; avoid multiple huge images.

## What to keep as notebook
Keep `notebooks/demo.ipynb` as a thin wrapper:
- one cell for installs (optional),
- one cell that calls the pipeline functions,
- one cell that prints results.
The `.py` package remains the canonical implementation.
::contentReference[oaicite:0]{index=0}






