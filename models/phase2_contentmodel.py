# models/phase2_contentmodel.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_VECT = None
_MATRIX = None
_MOVIES = None

def _load_movies(root: Path) -> pd.DataFrame:
    # Prefer prepared CSV from Phase 1
    p1 = root / "data" / "ml-1m" / "prepared" / "movies.csv"
    if not p1.exists():
        # Fallback: try to quickly generate via phase1
        from .phase1_dataprep import prepare_ml1m
        prepare_ml1m(root / "data" / "ml-1m")
    return pd.read_csv(root / "data" / "ml-1m" / "prepared" / "movies.csv")

def _ensure_model(root: Path):
    global _VECT, _MATRIX, _MOVIES
    if _MATRIX is not None and _MOVIES is not None:
        return
    _MOVIES = _load_movies(root)
    corpus = _MOVIES["tokens"].fillna("")
    _VECT = TfidfVectorizer(min_df=2, stop_words="english")
    _MATRIX = _VECT.fit_transform(corpus)

def get_content_recommendations(movie_title: str, top_n: int = 10) -> pd.DataFrame:
    """
    Returns DataFrame: ['title','score'] best matches to the input title.
    """
    root = Path(__file__).resolve().parents[1]
    _ensure_model(root)
    movies = _MOVIES

    if not movie_title:
        return pd.DataFrame(columns=["title", "score"])

    # Find the closest title index (simple exact/contains match; can be improved)
    matches = movies[movies["title"].str.contains(movie_title, case=False, na=False)]
    if matches.empty:
        # try fuzzy-like fallback: pick highest TF-IDF similarity to the query tokens
        q_vec = _VECT.transform([movie_title.lower()])
        sims = cosine_similarity(q_vec, _MATRIX).ravel()
        idx = int(np.argmax(sims))
    else:
        idx = matches.index[0]

    sims = cosine_similarity(_MATRIX[idx], _MATRIX).ravel()
    order = np.argsort(-sims)

    recs = (
        movies.iloc[order][["title"]]
        .assign(score=sims[order])
        .iloc[1 : top_n + 1]  # skip the query itself
        .reset_index(drop=True)
    )
    return recs
