import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import os

# Fonction pour télécharger un fichier depuis une URL
def download_file(url, dest_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(dest_path, 'wb') as f:
            f.write(response.content)
        st.success(f"Fichier téléchargé avec succès : {dest_path}")
    else:
        st.error(f"Erreur de téléchargement du fichier : {response.status_code}")

# URLs des fichiers nécessaires sur GitHub
movies_urls = [
    'https://github.com/AsmaM1983/movie-recommender/blob/main/movies_df1.csv',
    'https://github.com/AsmaM1983/movie-recommender/blob/main/movies_df2.csv',
    'https://github.com/AsmaM1983/movie-recommender/blob/main/movies_df3.csv',
    'https://github.com/AsmaM1983/movie-recommender/blob/main/movies_df4.csv',
    'https://github.com/AsmaM1983/movie-recommender/blob/main/movies_df5.csv',
    'https://github.com/AsmaM1983/movie-recommender/blob/main/movies_df6.csv'
]
ratings_url = 'https://github.com/AsmaM1983/movie-recommender/blob/main/ratings_small.csv'
model_url = 'https://github.com/AsmaM1983/movie-recommender/blob/main/best_algo_model.pkl'

# Chemins locaux des fichiers
movies_paths = [
    './movies_df1.csv',
    './movies_df2.csv',
    './movies_df3.csv',
    './movies_df4.csv',
    './movies_df5.csv',
    './movies_df6.csv'
]
merged_movies_df_path = './movies_df.csv'
ratings_path = './ratings_small.csv'
model_path = './best_algo_model.pkl'

# Télécharger les fichiers depuis GitHub
for url, path in zip(movies_urls, movies_paths):
    download_file(url, path)
download_file(ratings_url, ratings_path)
download_file(model_url, model_path)

# Fusionner les fichiers CSV
def merge_csv(files, output_file):
    try:
        dataframes = [pd.read_csv(file) for file in files]
        merged_df = pd.concat(dataframes)
        merged_df.to_csv(output_file, index=False)
        st.success(f"Fichiers fusionnés avec succès en : {output_file}")
    except Exception as e:
        st.error(f"Erreur lors de la fusion des fichiers CSV : {str(e)}")

# Fusionner les fichiers CSV
merge_csv(movies_paths, merged_movies_df_path)

# Charger les données et les modèles
if os.path.exists(merged_movies_df_path) and os.path.exists(ratings_path) and os.path.exists(model_path):
    movies_df = pd.read_csv(merged_movies_df_path)  # Charger le fichier CSV fusionné
    ratings_df = pd.read_csv(ratings_path)  # Charger un autre fichier CSV avec les évaluations des utilisateurs
    with open(model_path, 'rb') as f:
        algo_model = pickle.load(f)  # Charger le modèle depuis le fichier pickle
else:
    st.error("Les fichiers nécessaires n'ont pas été trouvés après fusion.")

# Calculer le weighted score et la similarité cosinus

# Filtrer les films ayant des valeurs manquantes pour vote_average ou vote_count
movies_df = movies_df[(movies_df['vote_average'].notnull()) & (movies_df['vote_count'].notnull())]

# Définir les variables nécessaires pour le calcul du score pondéré
R = movies_df['vote_average']
v = movies_df['vote_count']
m = movies_df['vote_count'].quantile(0.9)
c = movies_df['vote_average'].mean()

# Calculer la moyenne pondérée pour chaque film en utilisant la formule IMDB
movies_df['weighted_average'] = (R * v + c * m) / (v + m)

# Définir une time decay factor
current_year = 2020  # la dernière année de sortie de films dans la base de données
movies_df['time_decay_factor'] = 1 / (current_year - movies_df['year'] + 1)

# Initialiser le MinMaxScaler
scaler = MinMaxScaler()

# Fit et transformer les colonnes 'popularity', 'weighted_average', 'time_decay_factor', et 'revenue'
scaled = scaler.fit_transform(movies_df[['popularity', 'weighted_average', 'time_decay_factor', 'revenue']])

# Créer un DataFrame à partir des données mises à l'échelle
weighted_df = pd.DataFrame(scaled, columns=['popularity', 'weighted_average', 'time_decay_factor', 'revenue'])
weighted_df.index = movies_df['id']

# Calculer le score basé sur une combinaison pondérée de facteurs
weighted_df['score'] = (
    weighted_df['popularity'] * 0.4 +
    weighted_df['weighted_average'] * 0.4 +
    weighted_df['time_decay_factor'] * 0.05 +
    weighted_df['revenue'] * 0.15
)

# Vectorisation des 'bag_of_words' avec CountVectorizer
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(movies_df['bag_of_words'])

# Calcul de la similarité cosinus
cosine_sim = cosine_similarity(count_matrix, count_matrix)

# Création d'une série d'indices basée sur le titre du film
movies_df = movies_df.reset_index()
indices = pd.Series(movies_df.index, index=movies_df['title'])

# Fonction pour obtenir des recommandations de films basées sur la similarité cosinus
def get_recommendations(title, cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    similarity_scores = [i[1] for i in sim_scores]
    result_df = pd.DataFrame({
        'index': movie_indices,
        'title': movies_df['title'].iloc[movie_indices].values,
        'similarity_score': similarity_scores,
        'director': movies_df['director'].iloc[movie_indices].values,
        'genre': movies_df['genres'].iloc[movie_indices].values
    })
    return result_df

# Fonction pour prédire les notes des films
def hybrid_predicted_rating(userId, movieId, algo_model):
    collaborative_rating = algo_model.predict(userId, movieId).est
    sim_scores = list(enumerate(cosine_sim[movieId]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    similar_movies = movies_df.iloc[movie_indices]
    similar_movies['est'] = similar_movies['id'].apply(lambda x: algo_model.predict(userId, x).est)
    content_rating = similar_movies['est'].mean()
    weighted_score = weighted_df.loc[movies_df.loc[movieId, 'id'], 'score']
    final_rating = (0.5 * collaborative_rating) + (0.2 * content_rating) + (0.3 * weighted_score)
    return final_rating

# Fonction pour recommander des films pour les anciens utilisateurs
def fetch_weighted_scores(movie_ids, weighted_df):
    weighted_df = weighted_df.loc[~weighted_df.index.duplicated(keep='first')]
    weighted_scores = {}
    for movie_id in movie_ids:
        if movie_id in weighted_df.index:
            weighted_scores[movie_id] = weighted_df.loc[movie_id]['score']
        else:
            weighted_scores[movie_id] = 0
    return weighted_scores

def show_movie_details(movie_ids, movies_df, combined_scores):
    details_df = movies_df[movies_df['id'].isin(movie_ids)][['id', 'title', 'year', 'genres', 'director']]
    st.write("Recommended Movies:")
    for index, row in details_df.iterrows():
        score = combined_scores.get(row['id'], 0)
        st.write(f"Title: {row['title']} ({row['year']}), Genres: {', '.join(row['genres'])}, Director: {row['director']}, Combined Score: {score:.2f}")

def hybrid_recommendation(user_id, n=10):
    user_ratings = ratings_df[ratings_df['userId'] == user_id]
    predictions = []
    for index, row in user_ratings.iterrows():
        pred = algo_model.predict(row['userId'], row['movieId']).est
        predictions.append((row['movieId'], pred))
    top_collab_movies = [x[0] for x in sorted(predictions, key=lambda x: x[1], reverse=True)[:n]]
    last_watched_movieId = user_ratings.iloc[-1]['movieId']
    if last_watched_movieId in movies_df['id'].values:
        watched_movie_idx = movies_df[movies_df['id'] == last_watched_movieId].index[0]
        similar_movies = list(enumerate(cosine_sim[watched_movie_idx]))
        sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:n+1]
        top_content_movies = [movies_df.iloc[i[0]]['id'] for i in sorted_similar_movies]
    else:
        top_content_movies = []

    combined_scores = fetch_weighted_scores(top_collab_movies + top_content_movies, weighted_df)
    combined_scores = {k: v for k, v in sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)}
    top_movie_ids = list(combined_scores.keys())[:n]

    show_movie_details(top_movie_ids, movies_df, combined_scores)

# Interface utilisateur Streamlit
st.title("Recommandation de Films")
user_id = st.number_input("Entrez l'ID utilisateur", min_value=1, step=1)
if st.button("Recommander"):
    hybrid_recommendation(user_id)