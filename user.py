import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

if not os.path.exists("feedback_data.csv"):
    pd.DataFrame(columns=["userId", "movieId", "title", "sim_query", "sim_user", "rating_scaled", "final_score", "feedback"]).to_csv("feedback.csv", index=False)

feedback_df = pd.read_csv("feedback_data.csv")  # o como lo estés almacenando

def save_feedback(feedback_row, filename="feedback_data.csv"):
    """
    Save a feedback row into a CSV file without duplicating entries for the same user and movie.
    """
    # Convert row to DataFrame
    new_entry = pd.DataFrame([feedback_row])
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        # Remove any existing feedback from same user for same movie
        df = df[~((df["userId"] == feedback_row["userId"]) & (df["movieId"] == feedback_row["movieId"]))]
        # Append the new entry
        df = pd.concat([df, new_entry], ignore_index=True)
    else:
        df = new_entry
    # Save back to file
    df.to_csv(filename, index=False)
    return df


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

#TODO: CHECK THIS
def estimate_weights_from_feedback(user_id, feedback_df):
    return 0.6, 0.3, 0.1

    #TODO:
    user_feedback = feedback_df[feedback_df["userId"] == user_id]
    
    # feedback_df.shape[0] < 20
    if user_feedback.empty or feedback_df.shape[0] < 20:
        # Not enough feedback: return default weights
        return 0.6, 0.3, 0.1

    # Normalize each feature (optional but recommended)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    features = user_feedback[["sim_query", "sim_user", "rating_scaled"]].values
    features_scaled = scaler.fit_transform(features)

    # Feedback as target
    target = user_feedback["feedback"].values

    # Ridge regression to find importance
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=0.1)
    model.fit(features_scaled, target)

    # Extract weights
    weights = model.coef_
    weights = weights / np.sum(weights)  # Normalize to sum to 1
    return tuple(weights)


# Page title
st.title("🎬 Movie Recommender")

# User input
user_id = st.number_input("Enter your User ID", min_value=1, max_value=1000, value=1)
query = st.text_area("Describe what you feel like watching (e.g., a sci-fi thriller with suspense):")

# Get weights from past feedbacks
weight_query, weight_user, weight_rating = estimate_weights_from_feedback(user_id, feedback_df)

# weight_query = 0.6
# weight_user = 0.3
# weight_rating = 0.1


# --- Button: generate recommendations ---
col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    if st.button("🎯 Search for Movies!"):
        st.session_state["recommendations"] = hybrid_recommendation(
            user_id, query, top_n=50,  # or whatever max you want to store
            w_query=weight_query,
            w_user=weight_user,
            w_rating=weight_rating
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
                    feedback_row = {
                        "userId": user_id,
                        "movieId": row["movieId"],
                        "title": row["title"],
                        "sim_query": row["sim_query"],
                        "sim_user": row["sim_user"],
                        "rating_scaled": row["rating_scaled"],
                        "final_score": row["final_score"],
                        "feedback": stars
                    }
                    save_feedback(feedback_row)
                    st.success(f"✅ Feedback for '{row['title']}' saved!")
                

# Show current page number
st.markdown(f"Page **{current_page + 1}** of **{total_pages}**")
