import sys, os, re
from pathlib import Path
import requests
import streamlit as st
import json, hashlib
import pandas as pd

# ======================================================
# PATH SETUP
# ======================================================
PROJECT_ROOT = Path(os.getcwd())
MODELS_PATH = PROJECT_ROOT / "models"

for path in [PROJECT_ROOT, MODELS_PATH]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# ======================================================
# DATA FOLDERS
# ======================================================
ROOT = PROJECT_ROOT
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
USERS_PATH = DATA_DIR / "users.json"
HISTORY_PATH = DATA_DIR / "history.json"

# ======================================================
# TMDB API CONFIG
# ======================================================
TMDB_API_KEY = "0a97342470f867d2c5a9f6b317ac58f8"

@st.cache_data(show_spinner=False)
def fetch_poster(title: str) -> str:
    """
    Improved TMDB poster fetcher:
    - Cleans and normalizes movie titles
    - Attempts fuzzy year-based matching
    - Falls back gracefully to placeholder if not found
    """
    try:
        # 1️⃣ Clean movie title (remove year, commas, trailing "The")
        clean_title = re.sub(r"\s*\([^)]*\)", "", title)  # remove (1995)
        clean_title = re.sub(r",\s*The$", "", clean_title, flags=re.IGNORECASE)
        clean_title = clean_title.strip()

        # 2️⃣ Extract year if available
        match = re.search(r"\((\d{4})\)", title)
        year = match.group(1) if match else None

        # 3️⃣ Search TMDB
        url = "https://api.themoviedb.org/3/search/movie"
        params = {"api_key": TMDB_API_KEY, "query": clean_title}
        res = requests.get(url, params=params, timeout=5)

        if res.status_code == 200:
            data = res.json()
            results = data.get("results", [])

            # 4️⃣ Retry with alternate title forms if needed
            if not results:
                alt_title = clean_title.replace("The ", "").replace(",", "").replace("'", "")
                params = {"api_key": TMDB_API_KEY, "query": alt_title}
                res = requests.get(url, params=params, timeout=5)
                data = res.json()
                results = data.get("results", [])

            # 5️⃣ Try year-based filtering
            if results:
                if year:
                    for r in results:
                        release_date = r.get("release_date", "")
                        if release_date.startswith(year) and r.get("poster_path"):
                            return f"https://image.tmdb.org/t/p/w500{r['poster_path']}"

                if results[0].get("poster_path"):
                    return f"https://image.tmdb.org/t/p/w500{results[0]['poster_path']}"

    except Exception as e:
        print(f"TMDB error for '{title}': {e}")

    # 6️⃣ Fallback placeholder
    return "https://via.placeholder.com/200x300?text=No+Poster"

# ======================================================
# POSTER RENDER FUNCTION
# ======================================================
def _render_results(df: pd.DataFrame):
    """
    Renders movie recommendations with poster images.
    If a poster can't be loaded, shows a gray 'No Image Available' box.
    """
    if df is None or df.empty:
        st.error("No results. Check that your phase functions return data.")
        return

    st.success(f"Found {len(df)} recommendations")

    with st.spinner("Loading posters..."):
        cols = st.columns(5)
        for i, (_, row) in enumerate(df.iterrows()):
            col = cols[i % 5]
            with col:
                poster_url = fetch_poster(row["title"])

                # Check if poster URL is valid or placeholder
                if "placeholder.com" in poster_url or not poster_url:
                    # Display fallback box
                    st.markdown(
                        f"""
                        <div style='
                            width: 200px;
                            height: 300px;
                            background-color:  #1e1e1e;
                            color: white;
                            display: flex;
                            justify-content: center;
                            align-items: center;
                            border-radius: 10px;
                            font-size: 14px;
                            border: 1px solid #444;;
                            margin-bottom: 8px;
                        '>
                            No Image Available
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    # Show actual poster image
                    st.image(poster_url, use_container_width=True)

                # Movie title and score text
                st.markdown(
                    f"""
                    <div style='text-align:center; margin-top:5px;'>
                        🎬 <b>{row['title']}</b><br>
                        ⭐ <i>Score:</i> {row.get('score', 0):.3f}
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# ======================================================
# AUTH HELPERS
# ======================================================
def _sha(txt: str) -> str:
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()

def _load_json(path: Path, default):
    if not path.exists():
        path.write_text(json.dumps(default, indent=2))
        return default
    try:
        return json.loads(path.read_text() or json.dumps(default))
    except Exception:
        return default

def _save_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2))

def register_user(username: str, password: str) -> bool:
    users = _load_json(USERS_PATH, {})
    if username in users:
        return False
    users[username] = _sha(password)
    _save_json(USERS_PATH, users)
    return True

def validate_login(username: str, password: str) -> bool:
    users = _load_json(USERS_PATH, {})
    return users.get(username) == _sha(password)

def add_history(username: str, payload: dict):
    hist = _load_json(HISTORY_PATH, {})
    hist.setdefault(username, []).append(payload)
    _save_json(HISTORY_PATH, hist)

def get_history(username: str):
    hist = _load_json(HISTORY_PATH, {})
    return hist.get(username, [])

# ======================================================
# MODEL IMPORTS
# ======================================================
def _try_import(module_path):
    try:
        mod = __import__(module_path, fromlist=["*"])
        return mod
    except Exception as e:
        print(f"Import failed for {module_path}: {e}")
        return None

content_mod = _try_import("models.phase2_contentmodel")
collab_mod  = _try_import("models.phase3_collabfiltering")
hybrid_mod  = _try_import("models.phase4_hybridfusion")

# ======================================================
# ADAPTERS FOR RECOMMENDER PHASES
# ======================================================
def _call_first(mod, names, *args, **kwargs):
    if not mod:
        return None
    for name in names:
        fn = getattr(mod, name, None)
        if callable(fn):
            try:
                return fn(*args, **kwargs)
            except Exception:
                continue
    return None

def recommend_content(movie_title: str, top_n: int = 10):
    result = _call_first(
        content_mod,
        ["get_content_recommendations", "recommend_content", "recommend_movies_content"],
        movie_title, top_n=top_n,
    )
    return _normalize_result(result)

def recommend_collab(user_id: str, top_n: int = 10):
    result = _call_first(
        collab_mod,
        ["get_collab_recommendations", "recommend_collab", "recommend_movies_collaborative"],
        user_id, top_n=top_n,
    )
    return _normalize_result(result)

def recommend_hybrid(movie_title: str = None, user_id: str = None, top_n: int = 10):
    result = _call_first(
        hybrid_mod,
        ["get_hybrid_recommendations", "recommend_hybrid", "hybrid_fusion"],
        movie_title, user_id, top_n=top_n,
    )
    return _normalize_result(result)

def _normalize_result(result):
    if result is None:
        return pd.DataFrame(columns=["rank", "title", "score"])
    if isinstance(result, pd.DataFrame):
        df = result.copy()
        cols = {c.lower(): c for c in df.columns}
        title_col = cols.get("title") or cols.get("movie") or cols.get("name")
        score_col = cols.get("score") or cols.get("similarity") or cols.get("rating")
        if title_col is None:
            df = df.reset_index().rename(columns={"index": "title"})
            title_col = "title"
        if score_col is None:
            df["score"] = None
            score_col = "score"
        df = df.rename(columns={title_col: "title", score_col: "score"})
        df = df[["title", "score"]]
        df.insert(0, "rank", range(1, len(df) + 1))
        return df
    if isinstance(result, (list, tuple)):
        rows = []
        for i, item in enumerate(result, start=1):
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                rows.append({"rank": i, "title": item[0], "score": item[1]})
            elif isinstance(item, dict) and "title" in item:
                rows.append({"rank": i, "title": item["title"], "score": item.get("score")})
            else:
                rows.append({"rank": i, "title": str(item), "score": None})
        return pd.DataFrame(rows)
    return pd.DataFrame(columns=["rank", "title", "score"])

# ======================================================
# UI CONFIG + MAIN TABS
# ======================================================
st.set_page_config(page_title="Personalized Movie Recommender", page_icon="🎬", layout="wide")

if "user" not in st.session_state:
    st.session_state.user = None

with st.sidebar:
    st.header("🎟️ Account")
    if st.session_state.user:
        st.success(f"Logged in as **{st.session_state.user}**")
        if st.button("Log out"):
            st.session_state.user = None
            st.rerun()
    else:
        tab_login, tab_signup = st.tabs(["Login", "Sign up"])
        with tab_login:
            u = st.text_input("Username", key="login_u")
            p = st.text_input("Password", type="password", key="login_p")
            if st.button("Sign in"):
                if validate_login(u, p):
                    st.session_state.user = u
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
        with tab_signup:
            u2 = st.text_input("New username", key="reg_u")
            p2 = st.text_input("New password", type="password", key="reg_p")
            if st.button("Create account"):
                if not u2 or not p2:
                    st.warning("Enter both username and password.")
                elif register_user(u2, p2):
                    st.success("Account created. Please log in.")
                else:
                    st.error("Username already exists.")

st.title("🎬 Personalized Movie Recommender System")

if not st.session_state.user:
    st.info("Please log in from the sidebar to use the recommender.")
    st.stop()

tab_rec, tab_hist, tab_info = st.tabs(["Get Recommendations", "History", "About"])

with tab_rec:
    st.subheader("Choose a method")
    method = st.radio("Method", ["Content-based", "Collaborative", "Hybrid"], horizontal=True)

    col1, col2, col3 = st.columns([1.5, 1, 1])
    with col1:
        movie_title = st.text_input("Favorite movie (for Content/Hybrid)", placeholder="e.g., The Dark Knight")
    with col2:
        user_id = st.text_input("User ID (for Collaborative/Hybrid)", placeholder="e.g., 42")
    with col3:
        top_n = st.number_input("Top N", min_value=5, max_value=50, value=10, step=1)

    if st.button("Recommend"):
        if method == "Content-based":
            if not movie_title:
                st.warning("Enter a movie title.")
            else:
                df = recommend_content(movie_title, top_n=top_n)
                add_history(st.session_state.user, {"method": "content", "movie_title": movie_title})
                _render_results(df)
        elif method == "Collaborative":
            if not user_id:
                st.warning("Enter a user ID.")
            else:
                df = recommend_collab(user_id, top_n=top_n)
                add_history(st.session_state.user, {"method": "collab", "user_id": user_id})
                _render_results(df)
        else:  # Hybrid
            if not (movie_title or user_id):
                st.warning("Enter at least a movie title or a user ID.")
            else:
                df = recommend_hybrid(movie_title, user_id, top_n)
                add_history(st.session_state.user, {"method": "hybrid", "movie_title": movie_title, "user_id": user_id})
                _render_results(df)

with tab_hist:
    st.subheader("Your recent requests")
    hist = list(reversed(get_history(st.session_state.user)))
    if not hist:
        st.info("No history yet.")
    else:
        st.table(pd.DataFrame(hist))

with tab_info:
    st.markdown("""
**How this app connects to your phases**

- `phase2_contentmodel.py` → Content-based recommendations  
- `phase3_collabfiltering.py` → Collaborative filtering  
- `phase4_hybridfusion.py` → Hybrid recommendation  
    """)
