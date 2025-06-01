"""
Movie Recommender App using Streamlit
Visualization file
"""

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import seaborn as sns

# local imports
import data_processor as dp
import stats_data as stats


# Initialize session state variables
if "recommendations" not in st.session_state:
    st.session_state["recommendations"] = pd.DataFrame()
if "genre_ratings" not in st.session_state:
    st.session_state["genre_ratings"] = None
if "page" not in st.session_state:
    st.session_state["page"] = 0

from movie_recommendator import (
    get_rated_movies,
    get_average_rating_per_genre,
    hybrid_recommendation
)

# Page title
st.title("🎬 Movie Recommender")

# User input
user_id = st.number_input("Enter your User ID", min_value=1, max_value=1000, value=1)
query = st.text_area("Describe what you feel like watching (e.g., a sci-fi thriller with suspense):")


# Settings section

default_weights = {
    "prompt": 0.6,
    "user_profile": 0.3,
    "rating": 0.1,
    "genre": 0.1,
    "year": 0.1
}
personalized = dp.get_personalized_weights(user_id, default_weights)

# --- Initialization (before expander) ---
for k in default_weights:
    if f"weight_{k}" not in st.session_state:
        st.session_state[f"weight_{k}"] = default_weights[k]

if "last_user_id" not in st.session_state:
    st.session_state["last_user_id"] = user_id

if st.session_state["last_user_id"] != user_id:
    personalized = dp.get_personalized_weights(user_id, default_weights)
    for k in default_weights:
        st.session_state[f"weight_{k}"] = personalized[k]
    st.session_state["last_user_id"] = user_id
            
with st.expander("### ⚙️ Advanced search"):
    
    # Personalized weights button
    if st.button("Get weight based on my user profile"):
        personalized = dp.get_personalized_weights(user_id, default_weights)
        for k in personalized:
            st.session_state[f"weight_{k}"] = personalized[k]
                    
        print("Personalized weights:", personalized)
        st.rerun()
    
    
    top_n = st.slider("Number of movies to recommend", min_value=5, max_value=20, value=10)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.slider(
            "Weight: description", 0.0, 1.0, st.session_state["weight_prompt"], step=0.05, key="weight_prompt")
    with col2:
        st.slider(
            "Weight: user profile", 0.0, 1.0, st.session_state["weight_user_profile"], step=0.05, key="weight_user_profile")
    with col3:
        st.slider(
            "Weight: global rating", 0.0, 1.0, st.session_state["weight_rating"], step=0.05, key="weight_rating")
    with col4:
        st.slider(
            "Weight: genre", 0.0, 1.0, st.session_state["weight_genre"], step=0.05, key="weight_genre")
    with col5:
        st.slider(
            "Weight: year", 0.0, 1.0, st.session_state["weight_year"], step=0.05, key="weight_year")

        
    

    weights = {
        "prompt": st.session_state["weight_prompt"],
        "user_profile": st.session_state["weight_user_profile"],
        "rating": st.session_state["weight_rating"],
        "genre": st.session_state["weight_genre"],
        "year": st.session_state["weight_year"]
    }
        
    # Normalize
    total_weight = sum(weights.values())
    if total_weight > 0:
        for k in weights:
            weights[k] /= total_weight
    
    
    
    
    
    
    
    # Show bar chart if available
    if "genre_ratings" in st.session_state:
        if st.session_state["genre_ratings"] is not None:
            st.subheader("📊 Average rating by genre")

            data = st.session_state["genre_ratings"]
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(data["genre"], data["avg_rating"], color="skyblue")
            ax.set_xlabel("Average Rating")
            ax.set_title("User's average rating per genre")
            st.pyplot(fig)



# --- Button: generate recommendations ---
col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    if st.button("🎯 Search for Movies!"):
        st.session_state["recommendations"] = dp.movie_recommendation(
            user_id = user_id, 
            user_prompt = query, 
            data = dp.transform_data(),
            n_results = top_n, 
            weights = weights,
        )
        st.session_state["page"] = 0  # Reset to first page

# Define pagination parameters
items_per_page = 5
total_items = len(st.session_state.get("recommendations", []))
total_pages = (total_items - 1) // items_per_page + 1

# Current page
current_page = st.session_state["page"]
start = current_page * items_per_page
end = start + items_per_page
current_slice = st.session_state["recommendations"].iloc[start:end]

# Display recommendations if available
if "recommendations" in st.session_state:
    
    st.subheader("🎬 Recommended Movies:")
    
    # Page selector (dropdown or next/prev)
    if "page" not in st.session_state:
        st.session_state["page"] = 0

    col1, col2, col3 = st.columns([1, 6, 1])
    with col1:
        # Fill vertical space dynamically
        for _ in range(len(current_slice)):
            st.write("")
            
        st.markdown("<div style='text-align: left;'>", unsafe_allow_html=True)    
        if st.button("Prev ⬅️", key="prev_btn") and st.session_state["page"] > 0:
            st.session_state["page"] -= 1
    with col3:  
        # Fill vertical space dynamically
        for _ in range(len(current_slice)):
            st.write("")
            
        st.markdown("<div style='text-align: right;'>", unsafe_allow_html=True)    
        if st.button("Next ➡️", key="next_btn") and st.session_state["page"] < total_pages - 1:
            st.session_state["page"] += 1   
    
    # Refresh current page to avoid bugs between buttons
    current_page = st.session_state["page"]
    start = current_page * items_per_page
    end = start + items_per_page
    current_slice = st.session_state["recommendations"].iloc[start:end]
    
    
    with col2:
        for _, row in current_slice.iterrows():
            with st.expander(f"{row['title']} ({row['avg_rating']:.2f}/5)"):
                # Show Wikipedia intro only if it's not empty or a placeholder
                if row.get("wikipedia_intro") and row["wikipedia_intro"] not in ["-"]:
                    st.write(f"**Description:** {row['wikipedia_intro']}")
                else:
                    st.write(f"**Description:** ")
                
                # Show link if valid
                if row.get("wikipedia_link") and row["wikipedia_link"] not in ["-"]:
                    st.markdown(f"[🔗 Wikipedia page]({row['wikipedia_link']})", unsafe_allow_html=True)

                # Other metadata
                st.write(f"**Genres:** {row['genres']}")

                # ⭐️ Feedback system
                stars = st.slider(
                        "Rate this recommendation:",
                        min_value=0,
                        max_value=5,
                        value=0,
                        step=1,
                        format="%d ⭐",
                        key=f"feedback_slider_{row['movieId']}"
                    )

                # Save feedback when rated
                if stars > 0 and st.button(f"Submit feedback", key=f"submit_slider_{row['movieId']}"):
                    dp.save_recommendation_feedback(
                        user_id, 
                        row["movieId"], 
                        stars, 
                        row["score_breakdown"]
                    )
                    st.success(f"✅ Feedback for '{row['title']}' saved!")
                
# Show current page number
st.markdown(f"Page **{current_page + 1}** of **{total_pages}**")                
                
       

#########################
##### STATS SECTION #####
#########################

with st.expander("👍👎 Likes & Dislikes by Demographics and Genre", expanded=False):
    st.subheader("Likes & Dislikes by Age, Occupation, and Genre")

    # --- By Age Group ---
    st.write("#### By Age Group")
    age_stats = stats.get_likes_dislikes_by_age()
    fig, ax = plt.subplots()
    age_stats.plot(x="age", y=["like", "dislike"], kind="bar", ax=ax)
    ax.set_xlabel("Age Group")
    ax.set_ylabel("Count")
    ax.set_title("Likes & Dislikes by Age Group")
    st.pyplot(fig)

    # --- By Occupation ---
    st.write("#### By Occupation")
    occ_stats = stats.get_likes_dislikes_by_occupation()
    fig2, ax2 = plt.subplots()
    occ_stats.plot(x="occupation", y=["like", "dislike"], kind="bar", ax=ax2)
    ax2.set_xlabel("Occupation")
    ax2.set_ylabel("Count")
    ax2.set_title("Likes & Dislikes by Occupation")
    st.pyplot(fig2)

    # --- By Genre ---
    st.write("#### By Genre")
    genre_stats = stats.get_likes_dislikes_by_genre()
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    genre_stats.plot(x="genre", y=["like", "dislike"], kind="bar", ax=ax3)
    ax3.set_xlabel("Genre")
    ax3.set_ylabel("Count")
    ax3.set_title("Likes & Dislikes by Genre")
    st.pyplot(fig3)


