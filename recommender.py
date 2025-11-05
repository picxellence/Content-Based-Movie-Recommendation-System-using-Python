import ast 
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")
df = movies.merge(credits , on="title")
df = df[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]]

def safe_eval(x):
    try:
        return ast.literal_eval(x)
    except Exception:
        return []

def extract_names(list_of_dicts, key="name", top_k=None):
    if not isinstance(list_of_dicts, list):
        return []
    names = [d.get(key, "") for d in list_of_dicts if isinstance(d, dict)]
    return names[:top_k] if top_k else names

def get_director(crew_list):
    if not isinstance(crew_list, list):
        return []
    for d in crew_list:
        if isinstance(d, dict) and d.get("job") == "Director":
            return [d.get("name", "")]
    return []


df["overview"] = df["overview"].fillna("").str.lower()

for col in ["genres", "keywords", "cast", "crew"]:
    df[col] = df[col].apply(safe_eval)

df["genres"]   = df["genres"].apply(lambda x: extract_names(x))
df["keywords"] = df["keywords"].apply(lambda x: extract_names(x))
df["cast"]     = df["cast"].apply(lambda x: extract_names(x, top_k=3))
df["crew"]     = df["crew"].apply(get_director)

def clean_list(lst):
    return [s.replace(" ", "").lower() for s in lst if isinstance(s, str)]

for col in ["genres", "keywords", "cast", "crew"]:
    df[col] = df[col].apply(clean_list)

df["tags"] = (
    df["overview"] + " " +
    df["genres"].apply(lambda x: " ".join(x)) + " " +
    df["keywords"].apply(lambda x: " ".join(x)) + " " +
    df["cast"].apply(lambda x: " ".join(x)) + " " +
    df["crew"].apply(lambda x: " ".join(x))
)

data = df[["movie_id", "title", "tags"]].reset_index(drop=True)

cv = CountVectorizer(max_features=10000, stop_words="english")
vectors = cv.fit_transform(data["tags"]).toarray()
similarity = cosine_similarity(vectors)

#Recommender function
def recommend(title, top_n=5):
    # case-insensitive exact match, then fallback to contains
    titles = data["title"].tolist()
    idx = None
    for i, t in enumerate(titles):
        if t.strip().lower() == title.strip().lower():
            idx = i
            break
    if idx is None:
        q = title.strip().lower()
        for i, t in enumerate(titles):
            if q in t.lower():
                idx = i
                break
    if idx is None:
        return []

    scores = similarity[idx]
    # argsort descending, skip self
    order = np.argsort(scores)[::-1]
    out = []
    for i in order:
        if i == idx: 
            continue
        out.append(titles[i])
        if len(out) == top_n:
            break
    return out



