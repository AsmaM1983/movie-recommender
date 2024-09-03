import streamlit as st
import pandas as pd
import pickle
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
import torch

# Charger les fichiers nécessaires
@st.cache_data
def load_data():
    # Charger les parties du DataFrame movies_df et les combiner
    movies_df1 = pd.read_csv('https://github.com/AsmaM1983/movie-recommender/raw/main/movies_df1.csv')
    movies_df2 = pd.read_csv('https://github.com/AsmaM1983/movie-recommender/raw/main/movies_df2.csv')
    movies_df3 = pd.read_csv('https://github.com/AsmaM1983/movie-recommender/raw/main/movies_df3.csv')
    movies_df4 = pd.read_csv('https://github.com/AsmaM1983/movie-recommender/raw/main/movies_df4.csv')
    movies_df5 = pd.read_csv('https://github.com/AsmaM1983/movie-recommender/raw/main/movies_df5.csv')
    movies_df6 = pd.read_csv('https://github.com/AsmaM1983/movie-recommender/raw/main/movies_df6.csv')
    movies_df = pd.concat([movies_df1, movies_df2, movies_df3, movies_df4, movies_df5, movies_df6], ignore_index=True)
    
    ratings_df = pd.read_csv('https://github.com/AsmaM1983/movie-recommender/raw/main/ratings_small.csv')
    weighted_df = pd.read_csv('https://github.com/AsmaM1983/movie-recommender/raw/main/weighted_df.csv', index_col='id')
    best_algo_model = pickle.load(open('https://github.com/AsmaM1983/movie-recommender/raw/main/best_algo_model.pkl', 'rb'))
    
    return movies_df, ratings_df, weighted_df, best_algo_model

movies_df, ratings_df, weighted_df, best_algo_model = load_data()

# Charger le modèle BERT pré-entraîné
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Créer une nouvelle colonne combinant plusieurs caractéristiques textuelles pour les embeddings
movies_df['combined_features'] = movies_df['title'] + ' ' + movies_df['bag_of_words']

# Calculer les embeddings pour chaque film
embeddings = model.encode(movies_df['combined_features'].tolist(), convert_to_tensor=True)

# Fonction pour les recommandations hybrides pour les utilisateurs existants
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

# Fonction pour recommander des films pour les nouveaux utilisateurs
def recommend_for_new_user_top_rating_movies(weighted_df, movies_df, n=10, min_year=5):
    current_year = datetime.now().year
    sorted_df = pd.merge(weighted_df, movies_df[['id', 'year']], on='id', how='left')
    sorted_df['year'] = sorted_df['year'].fillna(0).astype(int)
    sorted_df = sorted_df[sorted_df['year'] >= (current_year - min_year)]
    sorted_df = sorted_df.drop_duplicates(subset='id', keep='first')
    sorted_df = sorted_df.sort_values(by='score', ascending=False)
    return sorted_df.head(n)

def show_movie_details(movie_ids, movies_df, weighted_scores):
    details_df = movies_df[movies_df['id'].isin(movie_ids)][['id', 'title', 'year', 'genres', 'director']]
    details_df['score'] = details_df['id'].map(weighted_scores)
    details_df = details_df.sort_values(by='score', ascending=False)
    details_df = details_df.drop_duplicates(subset='id', keep='first')
    
    st.write("Recommended Movies:")
    for _, row in details_df.iterrows():
        score = weighted_scores.get(row['id'], 0)
        st.write(f"Title: {row['title']} ({row['year']}), Genres: {', '.join(row['genres'])}, Director: {row['director']}, Weighted Score: {score:.2f}")

# Interface Streamlit
st.title("Movie Recommendation System")

user_id = st.number_input("Enter User ID:", min_value=1, step=1)

if st.button("Get Recommendations for Existing User"):
    recommended_movies = hybrid_recommendation_bert(user_id, best_algo_model, movies_df, embeddings, weighted_df, n=10)
    weighted_scores = fetch_weighted_scores(recommended_movies, weighted_df)
    show_movie_details(recommended_movies, movies_df, weighted_scores)

if st.button("Get Recommendations for New User"):
    top_movies = recommend_for_new_user_top_rating_movies(weighted_df, movies_df[['id', 'year']], n=10, min_year=8)
    weighted_scores = dict(zip(top_movies['id'], top_movies['score']))
    show_movie_details(top_movies['id'], movies_df, weighted_scores)
