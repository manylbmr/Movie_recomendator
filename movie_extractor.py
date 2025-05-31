# Updated Script: Write each result to CSV immediately (append mode)

import pandas as pd
import re
import time
import requests
from bs4 import BeautifulSoup
from SPARQLWrapper import SPARQLWrapper, JSON
import os

# Load MovieLens movies file
movies = pd.read_csv("archive/movie.csv")

# Extract year and clean title from the MovieLens title
def extract_year(title):
    match = re.search(r"\((\d{4})\)", title)
    return int(match.group(1)) if match else None

def clean_title(title):
    return re.sub(r"\(\d{4}\)", "", title).strip()

movies["clean_title"] = movies["title"].apply(clean_title)
movies["year"] = movies["title"].apply(extract_year)

# Function to query Wikidata for description and Wikipedia link
def get_wikidata_info(title):
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.setQuery(f"""
    SELECT ?movie ?movieLabel ?description ?article WHERE {{
      ?movie wdt:P31 wd:Q11424;
             rdfs:label "{title}"@en;
             schema:description ?description.
      FILTER(LANG(?description) = "en")
      OPTIONAL {{
        ?article schema:about ?movie ;
                 schema:isPartOf <https://en.wikipedia.org/> .
      }}
    }}
    LIMIT 1
    """)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        bindings = results["results"]["bindings"]
        if bindings:
            desc = bindings[0]["description"]["value"]
            link = bindings[0].get("article", {}).get("value", "No link found")
            return desc, link
        else:
            return "No description found", "No link found"
    except Exception as e:
        return f"Error: {str(e)}", "Error"

# Function to fetch Wikipedia intro text
def get_wikipedia_intro(wiki_url):
    headers = {
        "User-Agent": "MovieLensRecommenderBot/1.0 (mailto:youremail@example.com)"
    }
    try:
        response = requests.get(wiki_url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        p_tags = soup.select("p")
        for p in p_tags:
            text = p.get_text().strip()
            if text:
                return text
        return "No intro found"
    except Exception as e:
        return f"Error: {str(e)}"

# Prepare output CSV file
output_file = "movies_with_wikipedia_intro.csv"
write_header = not os.path.exists(output_file)

start_index = 15449  # 👈 Cambia este valor al índice desde donde quieres continuar

for i, row in movies.iloc[start_index:].iterrows():
    clean = row["clean_title"]
    print(f"[{i+1}/{len(movies)}] Fetching: {clean}")
    desc, link = get_wikidata_info(clean)
    intro = get_wikipedia_intro(link) if "wikipedia.org" in link else "No intro found"
    result_row = pd.DataFrame([{
        "movieId": row["movieId"],
        "title": row["title"],
        "clean_title": clean,
        "year": row["year"],
        "wikidata_description": desc,
        "wikipedia_link": link,
        "wikipedia_intro": intro
    }])
    result_row.to_csv(output_file, mode='a', header=write_header, index=False)
    write_header = False  # Only write header for the first row
    time.sleep(0.3)  # Be polite to endpoints

print(f"✅ Completed. Data saved incrementally to {output_file}")


#Merge files.

# Load the original movie file with genres
original_movies = pd.read_csv("archive/movie.csv")  # should contain 'movieId', 'title', 'genres'

# Load the enriched file with Wikipedia info
enriched_movies = pd.read_csv("movies_with_wikipedia_intro.csv", quotechar='"', escapechar='\\', on_bad_lines='skip')

# Merge on 'movieId' to bring the genres into the enriched file
merged = pd.merge(enriched_movies, original_movies[["movieId", "genres"]], on="movieId", how="left")

# Save to a new file
merged.to_csv("movies_with_genres_and_intro.csv", index=False)

print("✅ Genres successfully added. Output saved to 'movies_with_genres_and_intro.csv'")
