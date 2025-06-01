import data_processor as dp
import pandas as pd

def get_likes_dislikes_by_age():
    feedback = pd.read_csv(dp.FEEDBACK_FILE)
    users = dp.USERS_DF
    merged = feedback.merge(users, left_on="userId", right_on="user_id", how="left")
    merged["like"] = merged["feedback"].apply(lambda x: 1 if x >= 4 else 0)
    merged["dislike"] = merged["feedback"].apply(lambda x: 1 if x <= 2 else 0)
    return merged.groupby("age")[["like", "dislike"]].sum().reset_index()

def get_likes_dislikes_by_occupation():
    feedback = pd.read_csv(dp.FEEDBACK_FILE)
    users = dp.USERS_DF
    merged = feedback.merge(users, left_on="userId", right_on="user_id", how="left")
    merged["like"] = merged["feedback"].apply(lambda x: 1 if x >= 4 else 0)
    merged["dislike"] = merged["feedback"].apply(lambda x: 1 if x <= 2 else 0)
    return merged.groupby("occupation")[["like", "dislike"]].sum().reset_index()

def get_likes_dislikes_by_genre():
    feedback = pd.read_csv(dp.FEEDBACK_FILE)
    movies = dp.MOVIES_DF
    merged = feedback.merge(movies, left_on="movieId", right_on="movieId", how="left")
    merged["like"] = merged["feedback"].apply(lambda x: 1 if x >= 4 else 0)
    merged["dislike"] = merged["feedback"].apply(lambda x: 1 if x <= 2 else 0)
    merged["genres"] = merged["genres"].fillna("")
    merged_genres = merged.assign(genre=merged["genres"].str.split("|")).explode("genre")
    genre_stats = merged_genres.groupby("genre")[["like", "dislike"]].sum().reset_index()
    genre_stats = genre_stats[genre_stats["genre"] != ""]
    return genre_stats