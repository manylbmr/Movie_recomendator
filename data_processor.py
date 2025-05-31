"""
Data processing module for the MovieLens dataset.
This module handles the loading and processing of various datasets used in the MovieLens recommendation system.
"""
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load the pre-trained SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")



""" Dataset file paths"""
MOVIES_FILE = "dataset/movies_with_genres_and_intro.csv"
RATINGS_FILE = "dataset/rating.csv"
USERS_FILE = "dataset/users.csv"
TAGS_FILE = "dataset/tag.csv"
GENOME_SCORES_FILE = "dataset/genome_scores.csv"
GENOME_TAGS_FILE = "dataset/genome_tags.csv"



""" Load datasets"""
MOVIES_DF = pd.read_csv(MOVIES_FILE, quotechar='"', escapechar='\\', on_bad_lines='skip')
RATINGS_DF = pd.read_csv(RATINGS_FILE)
USERS_DF = pd.read_csv(USERS_FILE)
TAGS_DF = pd.read_csv(TAGS_FILE)
GENOME_SCORES_DF = pd.read_csv(GENOME_SCORES_FILE)
GENOME_TAGS_DF = pd.read_csv(GENOME_TAGS_FILE)



""" 
Function to transform and prepare data for recommendations 
"""
@st.cache_data(ttl=3600, show_spinner=False, persist=True)
def transform_data():
    # Load datasets
    movies = MOVIES_DF.set_index('movieId')
    ratings = RATINGS_DF
    users = USERS_DF
    genome_scores = GENOME_SCORES_DF
    genome_tags = GENOME_TAGS_DF

    # Build movie-tag relevance matrix (movieId x tagId)
    movie_tag_matrix = genome_scores.pivot(index='movieId', columns='tagId', values='relevance').fillna(0)
    tag_map = dict(zip(genome_tags['tagId'], genome_tags['tag']))
    movie_tag_matrix.columns = [tag_map[int(tagId)] for tagId in movie_tag_matrix.columns]

    # Merge movie info into the matrix for easy access
    movie_features = movies[['clean_title', 'year', 'genres', 'wikipedia_intro']].copy()
    movie_features = movie_features.join(movie_tag_matrix, how='left').fillna(0)

    # Align movies and tag matrix to only common movieIds and same order
    common_ids = movie_features.index.intersection(movie_tag_matrix.index)
    movie_features = movie_features.loc[common_ids].copy()
    movie_tag_matrix = movie_tag_matrix.loc[common_ids]
    movie_vectors = movie_tag_matrix.values  # numpy array for fast similarity

    return {
        "movies": movie_features,
        "ratings": ratings,
        "users": users,
        "tag_map": tag_map,
        "movie_tag_matrix": movie_tag_matrix,
        "movie_vectors": movie_vectors,
        "common_ids": common_ids
    }



def movie_recommendation(user_id, user_prompt, data, top_n=10, weights=None):
    """
    Recommend movies for a user based on their history and a prompt.
    Returns a DataFrame with score breakdowns.
    """
    movies = data["movies"]
    ratings = data["ratings"]
    tag_map = data["tag_map"]
    movie_tag_matrix = data["movie_tag_matrix"]

    if weights is None:
        weights = {
            "prompt": 0.4,
            "user_profile": 0.4,
            "genre": 0.1,
            "year": 0.1
        }

    # --- 1. User profile vector (average of tag vectors for highly rated movies) ---
    user_ratings = ratings[ratings["userId"] == user_id]
    high_rated = user_ratings[user_ratings["rating"] >= 4.0]
    if not high_rated.empty:
        user_movie_vectors = movie_tag_matrix.loc[high_rated["movieId"]].values
        user_profile_vec = np.average(user_movie_vectors, axis=0, weights=high_rated["rating"])
    else:
        user_profile_vec = np.zeros(movie_tag_matrix.shape[1])

    # --- 2. Prompt embedding in tag space (simple: match tags in prompt) ---
    prompt_vec = np.zeros(movie_tag_matrix.shape[1])
    prompt_words = set(user_prompt.lower().split())
    tag_list = [tag.lower() for tag in movie_tag_matrix.columns]
    for i, tag in enumerate(tag_list):
        if any(word in tag for word in prompt_words):
            prompt_vec[i] = 1.0
    if np.sum(prompt_vec) > 0:
        prompt_vec /= np.linalg.norm(prompt_vec)

    # --- 3. Similarity scores ---
    movie_vectors = movie_tag_matrix.values
    sim_prompt = cosine_similarity([prompt_vec], movie_vectors)[0]
    sim_user = cosine_similarity([user_profile_vec], movie_vectors)[0]

    # --- 4. Genre and year preference ---
    def genre_score(movie_genres):
        if high_rated.empty:
            return 0
        user_genres = "|".join(movies.loc[high_rated["movieId"]]["genres"].values)
        user_genres = set(user_genres.split("|"))
        movie_genres = set(movie_genres.split("|"))
        return len(user_genres & movie_genres) / len(user_genres | movie_genres)

    if not high_rated.empty:
        fav_year = int(np.average(movies.loc[high_rated["movieId"]]["year"], weights=high_rated["rating"]))
    else:
        fav_year = None

    def year_score(movie_year):
        if fav_year is None or pd.isna(movie_year):
            return 0
        return 1 - (abs(int(movie_year) - fav_year) / 50)

    # --- 5. Combine scores ---
    
    movies = movies.copy()
    movies["score_prompt"] = sim_prompt
    movies["score_user"] = sim_user
    movies["score_genre"] = movies["genres"].apply(genre_score)
    movies["score_year"] = movies["year"].apply(year_score)

    movies["final_score"] = (
        weights["prompt"] * movies["score_prompt"] +
        weights["user_profile"] * movies["score_user"] +
        weights["genre"] * movies["score_genre"] +
        weights["year"] * movies["score_year"]
    )

    result = movies.sort_values("final_score", ascending=False).head(top_n).copy()
    result["score_breakdown"] = result.apply(
        lambda row: {
            "prompt": row["score_prompt"],
            "user_profile": row["score_user"],
            "genre": row["score_genre"],
            "year": row["score_year"],
            "weights": weights
        }, axis=1
    )

    return result.reset_index()[[
        "movieId", "clean_title", "year", "genres", "final_score",
        "score_prompt", "score_user", "score_genre", "score_year", "score_breakdown"
    ]]


# data = transform_data()
# print(data)

test = movie_recommendation(
    user_id=1,
    user_prompt="A thrilling action movie with a die hard style",
    data = transform_data()
    )

pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)        # No line width limit



print("MovieId", test["movieId"].iloc[0])  
print("Title", test["clean_title"].iloc[0])  
print("Year", test["year"].iloc[0]) 
print("Genres", test["genres"].iloc[0])  
print("final score", test["final_score"].iloc[0]) 
print("score prompt", test["score_prompt"].iloc[0])  
print("score user", test["score_user"].iloc[0])  
print("score genre", test["score_genre"].iloc[0])  
print("score year", test["score_year"].iloc[0])  
print("score breakdown", test["score_breakdown"].iloc[0]) 


