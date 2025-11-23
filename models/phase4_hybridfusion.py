# models/phase4_hybridfusion.py
from __future__ import annotations
import pandas as pd
from pathlib import Path

# Import from sibling modules
from .phase2_contentmodel import get_content_recommendations
from .phase3_collabfiltering import get_collab_recommendations

def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["title", "score"])
    df = df.copy()
    if "score" not in df.columns:
        df["score"] = 1.0
    # min-max to [0,1]
    mn, mx = df["score"].min(), df["score"].max()
    if mx > mn:
        df["score"] = (df["score"] - mn) / (mx - mn)
    else:
        df["score"] = 1.0
    return df[["title", "score"]]

def get_hybrid_recommendations(movie_title: str | None = None,
                               user_id: str | int | None = None,
                               top_n: int = 10) -> pd.DataFrame:
    """
    Combines content and collaborative lists by normalized score average.
    Works if you pass either movie_title, user_id, or both.
    """
    parts = []

    if movie_title:
        c = _normalize(get_content_recommendations(movie_title, top_n=top_n*3))
        c = c.groupby("title", as_index=False)["score"].max()
        parts.append(c)

    if user_id is not None and str(user_id).strip() != "":
        u = _normalize(get_collab_recommendations(user_id, top_n=top_n*3))
        u = u.groupby("title", as_index=False)["score"].max()
        parts.append(u)

    if not parts:
        return pd.DataFrame(columns=["title", "score"])

    fused = parts[0]
    for p in parts[1:]:
        fused = fused.merge(p, on="title", how="outer", suffixes=("", "_2"))
        fused["score_2"] = fused["score_2"].fillna(fused["score"])
        fused["score"] = (fused["score"].fillna(0) + fused["score_2"]) / 2.0
        fused = fused[["title", "score"]]

    fused = fused.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)
    return fused
