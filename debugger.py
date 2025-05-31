import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import os
# import styles

if not os.path.exists("feedback.csv"):
    pd.DataFrame(columns=["userId", "movieId", "title", "sim_query", "sim_user", "rating_scaled", "final_score", "feedback"]).to_csv("feedback.csv", index=False)


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
with st.expander("⚙️ Advanced search"):

    # Number of results
    top_n = st.slider("Number of movies to recommend", min_value=5, max_value=20, value=10)

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
col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    if st.button("🎯 Search for Movies!"): # use_container_width=True
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
                st.write(f"**Final Score:** {row['final_score']:.3f}")
                st.write(f"**Explanation:** {row['explanation']}")

                # Show pie chart with checkbox
                if st.checkbox(f"📊 Show score breakdown for '{row['title']}'", key=f"chart_{row['movieId']}"):
                    fig = plot_score_pie(
                        sim_query=row["sim_query"],
                        sim_user=row["sim_user"],
                        rating_scaled=row["rating_scaled"],
                        w_query=weight_query,
                        w_user=weight_user,
                        w_rating=weight_rating
                    )
                    st.pyplot(fig)
                

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
