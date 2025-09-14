import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

# OMDb API key
API_KEY = '8eff6796'

# Function to fetch movie poster and description
def fetch_movie_data(title):
    try:
        url = f"http://www.omdbapi.com/?t={title}&apikey={API_KEY}"
        response = requests.get(url)
        data = response.json()
        poster = data.get('Poster', 'https://via.placeholder.com/150')
        plot = data.get('Plot', 'Description not available')
        return poster, plot
    except:
        return 'https://via.placeholder.com/150', 'Description not available'

# Load dataset
movies = pd.read_csv("imdb_top_1000.csv")

# Fill NaNs and combine relevant columns
movies.fillna('', inplace=True)
movies['combined_features'] = (
        movies['Genre'] + ' ' +
        movies['Director'] + ' ' +
        movies['Star1'] + ' ' +
        movies['Star2'] + ' ' +
        movies['Star3'] + ' ' +
        movies['Star4'] + ' ' +
        movies['Overview']
)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
vector_matrix = tfidf.fit_transform(movies['combined_features'])

# Cosine similarity
similarity = cosine_similarity(vector_matrix)

# Recommend function
def recommend(movie_title):
    movie_title = movie_title.lower()
    indices = movies[movies['Series_Title'].str.lower() == movie_title].index

    if len(indices) == 0:
        return []

    idx = indices[0]
    similarity_scores = list(enumerate(similarity[idx]))
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    top_movies = sorted_scores[1:7]  # ✅ now showing 6 recommendations

    recommendations = []
    for i in top_movies:
        movie_data = {}
        title = movies.iloc[i[0]]['Series_Title']
        poster, plot = fetch_movie_data(title)
        movie_data['title'] = title
        movie_data['poster'] = poster
        movie_data['plot'] = plot
        recommendations.append(movie_data)

    return recommendations

# Streamlit UI
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommendation System")

selected_movie = st.selectbox("Choose a movie you like:", movies['Series_Title'].values)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)

    if recommendations:
        # Show 3 recommendations per row
        for i in range(0, len(recommendations), 3):
            row_recs = recommendations[i:i+3]
            cols = st.columns(3)
            for col, rec in zip(cols, row_recs):
                with col:
                    st.image(rec['poster'], width=100)
                    st.markdown(f"**{rec['title']}**")
                    st.write(rec['plot'])
    else:
        st.warning("Movie not found in database. Please try another one.")
