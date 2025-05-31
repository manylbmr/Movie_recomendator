"""
File to handle the movie recommendation logic.
"""

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import data_processor

# Load datasets
#/Users/whiiz/Code/Movie_Recommendation/
# movies = pd.read_csv("movies_with_wikipedia_intro.csv")
movies = pd.read_csv("dataset/movies_with_genres_and_intro.csv", quotechar='"', escapechar='\\', on_bad_lines='skip')
ratings = pd.read_csv("dataset/rating.csv")
tags = pd.read_csv("dataset/tag.csv")

# Fill missing tags
tags['tag'] = tags['tag'].fillna('')
tags_agg = tags.groupby("movieId")["tag"].apply(lambda x: " ".join(x.astype(str))).reset_index()
movies = pd.merge(movies, tags_agg, on="movieId", how="left")
movies['tag'] = movies['tag'].fillna('')

# Construct text column including description if available
movies['description'] = movies['wikipedia_intro'].apply(lambda x: x if pd.notna(x) and x != "-" else "")
movies['text'] = (
    movies['title'] + ' ' +
    movies['genres'].str.replace('|', ' ', regex=False) + ' ' +
    movies['tag'] + ' ' +
    movies['description']
)

# Encode movie texts
model = SentenceTransformer('all-MiniLM-L6-v2')
movies['embedding'] = list(model.encode(movies['text'].tolist(), show_progress_bar=True))

# Average ratings
avg_ratings = ratings.groupby("movieId")["rating"].mean().reset_index()
avg_ratings.rename(columns={"rating": "avg_rating"}, inplace=True)
movies = pd.merge(movies, avg_ratings, on="movieId", how="left")


def get_average_rating_per_genre(user_id):
    user_ratings = ratings[ratings["userId"] == user_id]
    merged = pd.merge(user_ratings, movies, on="movieId", how="left")
    merged["genres"] = merged["genres"].fillna("")
    merged = merged.assign(genre=merged["genres"].str.split("|")).explode("genre")
    avg_per_genre = merged.groupby("genre")["rating"].mean().reset_index()
    avg_per_genre.rename(columns={"rating": "avg_rating"}, inplace=True)
    return avg_per_genre.sort_values(by="avg_rating", ascending=False)


def get_rated_movies(user_id):
    user_rated = ratings[ratings['userId'] == user_id]
    merged = pd.merge(user_rated, movies, on="movieId", how="left")
    return merged[['title', 'genres', 'rating']].sort_values(by='rating', ascending=False)


def get_user_profile(user_id):
    user_rated = ratings[ratings["userId"] == user_id]
    vectors = []
    weights = []

    for _, row in user_rated.iterrows():
        movie = movies[movies["movieId"] == row["movieId"]]
        if not movie.empty:
            vectors.append(movie.iloc[0]["embedding"])
            weights.append(row["rating"])
    if not vectors:
        return None
    return np.average(vectors, axis=0, weights=weights)


def hybrid_recommendation(user_id, query_text, top_n=10, w_query=0.6, w_user=0.3, w_rating=0.1):
    query_vec = model.encode([query_text])[0]
    user_vec = get_user_profile(user_id)

    embeddings = np.stack(movies["embedding"].values)
    sim_to_query = cosine_similarity([query_vec], embeddings)[0]
    sim_to_user = cosine_similarity([user_vec], embeddings)[0] if user_vec is not None else np.zeros(len(sim_to_query))
    rating_scaled = (movies["avg_rating"].fillna(0) / 5)

    movies["sim_query"] = sim_to_query
    movies["sim_user"] = sim_to_user
    movies["rating_scaled"] = rating_scaled

    movies["final_score"] = (
        w_query * sim_to_query +
        w_user * sim_to_user +
        w_rating * rating_scaled
    )

    top_movies = movies.sort_values(by="final_score", ascending=False).head(top_n).copy()

    top_movies["explanation"] = (
        f"Score = {int(w_query*100)}% descripción + {int(w_user*100)}% perfil + {int(w_rating*100)}% rating global."
    )

    return top_movies[[
        "movieId", "title", "genres", "avg_rating", "final_score",
        "sim_query", "sim_user", "rating_scaled", "explanation",
        "wikipedia_intro", "wikipedia_link"
    ]]
