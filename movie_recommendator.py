import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# Load datasets
movies = pd.read_csv("archive/movie.csv")
ratings = pd.read_csv("archive/rating.csv")
tags = pd.read_csv("archive/tag.csv")  # opcional

tags['tag'] = tags['tag'].fillna('')
tags_agg = tags.groupby("movieId")["tag"].apply(lambda x: " ".join(x.astype(str))).reset_index()
movies = pd.merge(movies, tags_agg, on="movieId", how="left")
movies['tag'] = movies['tag'].fillna('')
movies['text'] = movies['title'] + ' ' + movies['genres'].str.replace('|', ' ') + ' ' + movies['tag']

model = SentenceTransformer('all-MiniLM-L6-v2')
movies['embedding'] = list(model.encode(movies['text'].tolist(), show_progress_bar=True))

avg_ratings = ratings.groupby("movieId")["rating"].mean().reset_index()
avg_ratings.rename(columns={"rating": "avg_rating"}, inplace=True)
movies = pd.merge(movies, avg_ratings, on="movieId", how="left")



def get_average_rating_per_genre(user_id):
    """
    Returns a DataFrame with the average rating per genre for a given user.
    """
    # Películas valoradas por el usuario
    user_ratings = ratings[ratings["userId"] == user_id]
    merged = pd.merge(user_ratings, movies, on="movieId", how="left")

    # Separar los géneros y expandir las filas
    merged["genres"] = merged["genres"].fillna("")
    merged = merged.assign(genre=merged["genres"].str.split("|")).explode("genre")

    # Calcular el promedio por género
    avg_per_genre = merged.groupby("genre")["rating"].mean().reset_index()
    avg_per_genre.rename(columns={"rating": "avg_rating"}, inplace=True)

    return avg_per_genre.sort_values(by="avg_rating", ascending=False)


def get_rated_movies(user_id):
    """
    Returns a DataFrame of movies rated by a specific user.
    """
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
            weights.append(row["rating"])  # rating como peso

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

    # Guardar componentes por separado
    movies["sim_query"] = sim_to_query
    movies["sim_user"] = sim_to_user
    movies["rating_scaled"] = rating_scaled

    # Score final con pesos
    movies["final_score"] = (
        w_query * sim_to_query +
        w_user * sim_to_user +
        w_rating * rating_scaled
    )

    top_movies = movies.sort_values(by="final_score", ascending=False).head(top_n).copy()

    top_movies["explanation"] = (
        f"Score = {int(w_query*100)}% descripción + {int(w_user*100)}% perfil + {int(w_rating*100)}% rating global."
    )

    return top_movies[["title", "genres", "avg_rating", "final_score", "sim_query", "sim_user", "rating_scaled", "explanation"]]

# def hybrid_recommendation(user_id, query_text, top_n=10, w_query=0.6, w_user=0.3, w_rating=0.1):
#     query_vec = model.encode([query_text])[0]
#     user_vec = get_user_profile(user_id)

#     # Similitud con descripción del usuario
#     embeddings = np.stack(movies["embedding"].values)
#     sim_to_query = cosine_similarity([query_vec], embeddings)[0]

#     # Similitud con perfil del usuario (si hay)
#     if user_vec is not None:
#         sim_to_user = cosine_similarity([user_vec], embeddings)[0]
#     else:
#         sim_to_user = np.zeros(len(sim_to_query))  # sin peso si no hay datos

#     # Score combinado: 60% descripción, 30% perfil, 10% rating global
#     sim_score = (
#         w_query * sim_to_query +
#         w_user * sim_to_user +
#         w_rating * (movies["avg_rating"].fillna(0) / 5)  # normalizamos el rating global
#     )

#     movies["final_score"] = sim_score
#     top_movies = movies.sort_values(by="final_score", ascending=False).head(top_n)

#     top_movies["explanation"] = "Recommended based on your description, your past ratings, and global popularity."
#     return top_movies[["title", "genres", "avg_rating", "final_score", "explanation"]]


# # Combine tags por película
# tags_agg = tags.groupby("movieId")["tag"].apply(lambda x: " ".join(x)).reset_index()
# movies = pd.merge(movies, tags_agg, on="movieId", how="left")
# movies["tag"] = movies["tag"].fillna("")

# # Preparamos el texto para el embedding
# movies["text"] = movies["title"] + " " + movies["genres"].str.replace('|', ' ') + " " + movies["tag"]

# # Embedding del texto de cada película
# model = SentenceTransformer("all-MiniLM-L6-v2")
# movies["embedding"] = list(model.encode(movies["text"].tolist(), show_progress_bar=False))

# # Calcular rating promedio de cada película
# avg_ratings = ratings.groupby("movieId")["rating"].mean().reset_index()
# avg_ratings.rename(columns={"rating": "avg_rating"}, inplace=True)
# movies = pd.merge(movies, avg_ratings, on="movieId", how="left")