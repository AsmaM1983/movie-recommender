import streamlit as st
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
import joblib
import requests
from io import BytesIO

# Fonction pour charger des données à partir de GitHub
def load_from_github(url):
    response = requests.get(url)
    response.raise_for_status()  # Vérifie si la requête a réussi
    return pd.read_csv(BytesIO(response.content))

# Charger les datasets à partir de GitHub
url_base = "https://github.com/AsmaM1983/movie-recommender/raw/main/"
movies_dfs = [load_from_github(f"{url_base}movies_df{i}.csv") for i in range(1, 7)]
movies_df = pd.concat(movies_dfs, ignore_index=True)
ratings_df = load_from_github(f"{url_base}ratings_df.csv")
weighted_df = load_from_github(f"{url_base}weighted_df.csv")

# Charger le modèle depuis GitHub
def load_model_from_github(url):
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Vérifie si la requête a réussi
    return joblib.load(BytesIO(response.content))

best_algo_model = load_model_from_github(f"{url_base}best_algo_model.pkl")

# Charger le modèle BERT
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Ajouter une colonne combinée pour les embeddings BERT
movies_df['combined_features'] = movies_df['title'] + ' ' + movies_df['bag_of_words']

# Calculer les embeddings pour les films
embeddings = model.encode(movies_df['combined_features'].tolist(), convert_to_tensor=True)

# Fonction pour récupérer les scores pondérés
def fetch_weighted_scores(movie_ids, weighted_df):
    scores = {}
    for movie_id in movie_ids:
        if movie_id in weighted_df['id'].values:
            score = weighted_df[weighted_df['id'] == movie_id]['weighted_score'].values[0]
            scores[movie_id] = score
    return scores

# Fonction pour les recommandations hybrides
def hybrid_recommendation_bert(user_id, algo_model, movies_df, embeddings, weighted_df, n=10):
    user_ratings = ratings_df[ratings_df['userId'] == user_id]
    predictions = []

    for index, row in user_ratings.iterrows():
        pred = algo_model.predict(row['userId'], row['movieId']).est
        predictions.append((row['movieId'], pred))

    top_collab_movies = [x[0] for x in sorted(predictions, key=lambda x: x[1], reverse=True)[:n]]

    last_watched_movieId = user_ratings.iloc[-1]['movieId']

    if last_watched_movieId in movies_df['id'].values:
        watched_movie_idx = movies_df[movies_df['id'] == last_watched_movieId].index[0]
        query_embedding = embeddings[watched_movie_idx]
        cos_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
        similar_movies = [(i, score.item()) for i, score in enumerate(cos_scores)]
        sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:n+1]
        top_content_movies = [movies_df.iloc[i[0]]['id'] for i in sorted_similar_movies]
    else:
        top_content_movies = []

    collab_weighted_scores = fetch_weighted_scores(top_collab_movies, weighted_df)
    content_weighted_scores = fetch_weighted_scores(top_content_movies, weighted_df)

    combined_scores = {}
    for movie_id, score in collab_weighted_scores.items():
        combined_scores[movie_id] = combined_scores.get(movie_id, 0) + 0.5 * score
    for movie_id, score in content_weighted_scores.items():
        combined_scores[movie_id] = combined_scores.get(movie_id, 0) + 0.5 * score

    sorted_movies = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)
    
    return sorted_movies[:n]

# Streamlit interface
st.title("Movie Recommendation System")

user_id = st.number_input("Enter User ID", min_value=1, max_value=ratings_df['userId'].max())

if st.button("Recommend Movies"):
    recommendations = hybrid_recommendation_bert(user_id, best_algo_model, movies_df, embeddings, weighted_df)
    st.write("Recommended Movies:")
    for movie_id in recommendations:
        movie = movies_df[movies_df['id'] == movie_id].iloc[0]
        st.write(f"Title: {movie['title']}, Genres: {', '.join(movie['genres'])}, Year: {movie['year']}")
