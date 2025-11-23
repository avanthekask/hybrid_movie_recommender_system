# models/phase3_collabfiltering.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

_MOVIES = None
_RATINGS = None
_ITEM_SIM = None
_UI = None

def _load_base(root: Path):
    movies_p = root / "data" / "ml-1m" / "prepared" / "movies.csv"
    ratings_p = root / "data" / "ml-1m" / "prepared" / "ratings.csv"
    if not (movies_p.exists() and ratings_p.exists()):
        from .phase1_dataprep import prepare_ml1m
        prepare_ml1m(root / "data" / "ml-1m")
    movies = pd.read_csv(movies_p)
    ratings = pd.read_csv(ratings_p)
    return movies, ratings

def _ensure_item_model(root: Path):
    global _MOVIES, _RATINGS, _UI, _ITEM_SIM
    if _ITEM_SIM is not None:
        return
    _MOVIES, _RATINGS = _load_base(root)
    ui = _RATINGS.pivot_table(index="userId", columns="movieId", values="rating")
    ui = ui.fillna(0.0)
    _UI = ui
    # cosine similarity between items (movies)
    _ITEM_SIM = cosine_similarity(ui.T)  # item x item
    # store mapping
    _ITEM_SIM = pd.DataFrame(_ITEM_SIM, index=ui.columns, columns=ui.columns)

def get_collab_recommendations(user_id: str | int, top_n: int = 10) -> pd.DataFrame:
    """
    For a given user, recommend top-N movies they haven't rated,
    using item-based CF (weighted sum of similarities).
    """
    root = Path(__file__).resolve().parents[1]
    _ensure_item_model(root)
    user_id = int(user_id)
    if user_id not in _UI.index:
        return pd.DataFrame(columns=["title", "score"])

    user_ratings = _UI.loc[user_id]
    rated = user_ratings[user_ratings > 0]
    if rated.empty:
        return pd.DataFrame(columns=["title", "score"])

    # Score each unseen item by similarity to the user's rated items
    scores = {}
    for m_id, r in rated.items():
        sims = _ITEM_SIM[m_id]
        for target_id, sim in sims.items():
            if user_ratings[target_id] == 0 and sim > 0:
                scores[target_id] = scores.get(target_id, 0.0) + sim * r

    if not scores:
        return pd.DataFrame(columns=["title", "score"])

    s = pd.Series(scores).sort_values(ascending=False).head(top_n)
    movies_lookup = _MOVIES.set_index("movieId")["title"]
    out = pd.DataFrame({
        "title": [movies_lookup.get(mid, str(mid)) for mid in s.index],
        "score": s.values
    })
    return out
