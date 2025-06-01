
# 🎬 Movie Recommender Prototype with Streamlit + XAI

This is a hybrid movie recommendation system built with **Streamlit** and powered by **MovieLens** data. It allows users to receive personalized movie suggestions based on natural language input and past movie ratings. It also supports user feedback and explainable AI (XAI) techniques like SHAP.

---

## 📂 Project Structure

```
movie_recommender/
├── app.py                    # Main Streamlit application
├── data_processor.py         # File to load and manage the datasets
├── stats_data.py             # File to get the data to populate the graphs from the main application
├── extract_descriptions.py   # Script to enrich MovieLens data with Wikipedia descriptions
├── dataset/                  # Folder containing datasets (movies.csv, ratings.csv, tags.csv, etc.)
├── requirements.txt          # Python dependencies
└── 
```

---

## 🚀 Features

- 🎯 Hybrid recommendations (text query + user profile + rating score)
- 💬 Natural language input to describe what you want to watch
- ⭐ Feedback system (0 to 5 stars) to improve the model
- 📖 Description and intro for each movie (scraped from Wikipedia)
- 📊 Explainable AI (XAI) with SHAP
- 📄 Pagination to navigate large recommendation lists

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/movie-recommender.git
cd movie-recommender
```

### 2. Install dependencies

Use a virtual environment and install the requirements:

```bash
pip install -r requirements.txt
```

> Requirements include: `streamlit`, `pandas`, `scikit-learn`, `xgboost`, `shap`, `SPARQLWrapper`, `beautifulsoup4`, etc.

---

### 3. Prepare the data

Place your MovieLens dataset files inside the `archive/` folder. For example:

```
archive/
├── movies.csv
├── ratings.csv
├── tags.csv
└── links.csv
```

You can download them from: https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset/

---

### 4. Enrich movie data with Wikipedia descriptions (optional but recommended)

Run the enrichment script to attach descriptions and intro text from Wikipedia:

```bash
python extract_descriptions.py
```

This will create `movies_with_wikipedia_intro.csv` containing:
- movieId
- clean title
- release year
- short description
- Wikipedia URL
- intro paragraph

---

### 5. Launch the Streamlit app

Run the following command:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---


## 📄 License

MIT – feel free to use and adapt for educational or research purposes.
