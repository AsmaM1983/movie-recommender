import streamlit as st
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
import joblib
import requests

# Charger les datasets à partir de GitHub
url_base = "https://raw.githubusercontent.com/AsmaM1983/movie-recommender/main/"
movies_dfs = [pd.read_csv(f"{url_base}movies_df{i}.csv", on_bad_lines='skip') for i in range(1, 7)]
movies_df = pd.concat(movies_dfs, ignore_index=True)
ratings_df = pd.read_csv(f"{url_base}ratings_df.csv", on_bad_lines='skip')

# Charger le modèle et le DataFrame pondéré
best_algo_model = joblib.load(requests.get(f"{url_base}best_algo_model.pkl", stream=True).raw)
weighted_df = pd.read_csv(f"{url_base}weighted_df.csv", on_bad_lines='skip')

# Charger le modèle BERT
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Ajouter une colonne combinée pour les embeddings BERT
movies_df['combined_features'] = movies_df['title'] + ' ' + movies_df['bag_of_words']

# Calculer les embeddings pour les films
embeddings = model.encode(movies_df['combined_features'].tolist(), convert_to_tensor=True)

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
