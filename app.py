import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

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


# Plot function for pie chart
def plot_score_pie(sim_query, sim_user, rating_scaled, w_query, w_user, w_rating):
    values = [
        w_query * sim_query,
        w_user * sim_user,
        w_rating * rating_scaled
    ]
    labels = ["Description", "User Profile", "Global Rating"]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]

    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax.axis('equal')
    return fig

# Page title
st.title("🎬 Movie Recommender")

# User input
user_id = st.number_input("Enter your User ID", min_value=1, max_value=1000, value=1)
query = st.text_area("Describe what you feel like watching (e.g., a sci-fi thriller with suspense):")

# Settings section
st.markdown("### ⚙️ Recommendation Settings")

# Number of results
top_n = st.slider("Number of movies to recommend", min_value=10, max_value=30, value=15)

# Relevance weights
col1, col2, col3 = st.columns(3)
with col1:
    weight_query = st.slider("Weight: description", 0.0, 1.0, 0.6, step=0.05)
with col2:
    weight_user = st.slider("Weight: user profile", 0.0, 1.0, 0.3, step=0.05)
with col3:
    weight_rating = st.slider("Weight: global rating", 0.0, 1.0, 0.1, step=0.05)

# Normalize weights
total_weight = weight_query + weight_user + weight_rating
weight_query /= total_weight
weight_user /= total_weight
weight_rating /= total_weight

if st.button("🎭 Show average rating per genre"):
    st.session_state["genre_ratings"] = get_average_rating_per_genre(user_id)

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
if st.button("🎯 Get recommendations"):
    st.session_state["recommendations"] = hybrid_recommendation(
        user_id, query, top_n=top_n,  # or whatever max you want to store
        w_query=weight_query,
        w_user=weight_user,
        w_rating=weight_rating
    )
    st.session_state["page"] = 0  # Reset to first page

# Define pagination parameters
items_per_page = 5
total_items = len(st.session_state.get("recommendations", []))
total_pages = (total_items - 1) // items_per_page + 1

# Page selector (dropdown or next/prev)
if "page" not in st.session_state:
    st.session_state["page"] = 0

col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    if st.button("⬅️ Previous") and st.session_state["page"] > 0:
        st.session_state["page"] -= 1
with col3:
    if st.button("Next ➡️") and st.session_state["page"] < total_pages - 1:
        st.session_state["page"] += 1



# Current page
current_page = st.session_state["page"]
start = current_page * items_per_page
end = start + items_per_page
current_slice = st.session_state["recommendations"].iloc[start:end]

# Display recommendations if available
if "recommendations" in st.session_state:
    st.subheader("🎬 Recommended Movies:")
    for _, row in current_slice.iterrows():
        with st.expander(f"{row['title']} ({row['avg_rating']:.2f}/5)"):
            st.write(f"**Genres:** {row['genres']}")
            st.write(f"**Final Score:** {row['final_score']:.3f}")
            st.write(f"**Explanation:** {row['explanation']}")
            
            fig = plot_score_pie(
                sim_query=row["sim_query"],
                sim_user=row["sim_user"],
                rating_scaled=row["rating_scaled"],
                w_query=weight_query,
                w_user=weight_user,
                w_rating=weight_rating
            )
            st.pyplot(fig)

# Show current page number
st.markdown(f"Page **{current_page + 1}** of **{total_pages}**")
