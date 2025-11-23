# models/phase1_dataprep.py
import os
from pathlib import Path
import pandas as pd

ML1M_MOVIES = "movies.dat"
ML1M_RATINGS = "ratings.dat"
ML1M_USERS = "users.dat"

def _read_ml1m_movies(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        header=None,
        names=["movieId", "title", "genres"],
        encoding="latin-1"
    )
    df["movieId"] = df["movieId"].astype(int)
    return df


def _read_ml1m_ratings(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        header=None,
        names=["userId", "movieId", "rating", "timestamp"],
        encoding="latin-1"
    )
    df["userId"] = df["userId"].astype(int)
    df["movieId"] = df["movieId"].astype(int)
    df["rating"] = df["rating"].astype(float)
    return df


def _read_ml1m_users(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        header=None,
        names=["userId", "gender", "age", "occupation", "zip"],
        encoding="latin-1"
    )
    df["userId"] = df["userId"].astype(int)
    return df


def prepare_ml1m(data_dir: str | Path):
    data_dir = Path(data_dir)
    movies_path = data_dir / ML1M_MOVIES
    ratings_path = data_dir / ML1M_RATINGS
    users_path = data_dir / ML1M_USERS

    if not movies_path.exists() or not ratings_path.exists():
        raise FileNotFoundError(
            f"Could not find ml-1m files in {data_dir}. "
            "Expected movies.dat and ratings.dat."
        )

    movies = _read_ml1m_movies(movies_path)
    ratings = _read_ml1m_ratings(ratings_path)
    users = _read_ml1m_users(users_path) if users_path.exists() else pd.DataFrame()

    movies["tokens"] = (movies["title"].fillna("") + " " + movies["genres"].fillna("")).str.lower()

    (data_dir / "prepared").mkdir(exist_ok=True, parents=True)
    movies.to_csv(data_dir / "prepared" / "movies.csv", index=False)
    ratings.to_csv(data_dir / "prepared" / "ratings.csv", index=False)
    if not users.empty:
        users.to_csv(data_dir / "prepared" / "users.csv", index=False)

    return movies, ratings, users

def main():
    root = Path(__file__).resolve().parents[1]
    ml_dir = root / "data" / "ml-1m"
    movies, ratings, users = prepare_ml1m(ml_dir)
    print("Prepared:", len(movies), "movies;", len(ratings), "ratings;", len(users), "users")

if __name__ == "__main__":
    main()
