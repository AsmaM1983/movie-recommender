import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import rarfile
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

# Fonction pour extraire un fichier RAR
def extract_rar(file_path, extract_path):
    try:
        with rarfile.RarFile(file_path) as rf:
            rf.extractall(extract_path)
        st.success(f"Fichier extrait avec succès : {extract_path}")
    except rarfile.BadRarFile:
        st.error("Le fichier n'est pas un fichier RAR valide.")
    except Exception as e:
        st.error(f"Erreur d'extraction du fichier RAR : {str(e)}")

# URLs des fichiers nécessaires sur GitHub
movies_rar_url = 'https://github.com/AsmaM1983/movie-recommender/raw/main/movies_df.rar'
ratings_url = 'https://github.com/AsmaM1983/movie-recommender/raw/main/ratings_small.csv'
model_url = 'https://github.com/AsmaM1983/movie-recommender/raw/main/best_algo_model.pkl'

# Chemins locaux des fichiers
movies_rar_path = './movies_df.rar'
ratings_path = './ratings_small.csv'
model_path = './best_algo_model.pkl'

# Télécharger les fichiers depuis GitHub
download_file(movies_rar_url, movies_rar_path)
download_file(ratings_url, ratings_path)
download_file(model_url, model_path)

# Chemin d'extraction pour le fichier RAR
extract_path = './extracted_movies/'

# Extraire le fichier RAR
if not os.path.exists(extract_path):
    os.makedirs(extract_path)
extract_rar(movies_rar_path, extract_path)

# Charger les données et les modèles
movies_csv_path = os.path.join(extract_path, 'movies_df.csv')
if os.path.exists(movies_csv_path) and os.path.exists(ratings_path) and os.path.exists(model_path):
    movies_df = pd.read_csv(movies_csv_path)  # Charger le fichier CSV extrait
    ratings_df = pd.read_csv(ratings_path)  # Charger un autre fichier CSV avec les évaluations des utilisateurs
    with open(model_path, 'rb') as f:
        algo_model = pickle.load(f)  # Charger le modèle depuis le fichier pickle
else:
    st.error("Les fichiers nécessaires n'ont pas été trouvés après extraction.")

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
        st.write(f"Movie ID {last_watched_movieId} not found in movies_df.")
        top_content_movies = []
    collab_weighted_scores = fetch_weighted_scores(top_collab_movies, weighted_df)
    content_weighted_scores = fetch_weighted_scores(top_content_movies, weighted_df)
    combined_scores = {}
    for movie_id, score in collab_weighted_scores.items():
        combined_scores[movie_id] = combined_scores.get(movie_id, 0) + 0.5 * score
    for movie_id, score in content_weighted_scores.items():
        combined_scores[movie_id] = combined_scores.get(movie_id, 0) + 0.5 * score
    sorted_movies = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)
    return sorted_movies[:n], combined_scores

# Fonction pour recommander des films pour les nouveaux utilisateurs
def recommend_for_new_user_top_rating_movies(df, movies_df, n=10, min_year=5):
    current_year = datetime.now().year
    sorted_df = pd.merge(df, movies_df[['id', 'year']], on='id', how='left')
    sorted_df['year'] = sorted_df['year'].fillna(0).astype(int)
    sorted_df = sorted_df[sorted_df['year'] >= (current_year - min_year)]
    sorted_df = sorted_df.drop_duplicates(subset='id', keep='first')
    sorted_df = sorted_df.sort_values(by='score', ascending=False)
    return sorted_df.head(n)

# Application Streamlit
st.title("Système de Recommandation de Films")

# Section pour prédire les notes des films
st.header("Prédire les notes des films")
user_id = st.number_input("Entrez l'ID de l'utilisateur", min_value=1, max_value=ratings_df['userId'].max(), value=1)
movie_id = st.number_input("Entrez l'ID du film", min_value=1, max_value=movies_df['id'].max(), value=1)

if st.button("Prédire la note"):
    predicted_rating = hybrid_predicted_rating(user_id, movie_id, algo_model)
    st.write(f"La note hybride prédite pour l'utilisateur {user_id} et le film {movie_id} est : {predicted_rating:.2f}")

# Section pour recommander des films aux anciens utilisateurs
st.header("Recommander des films aux anciens utilisateurs")
if st.button("Recommander des films (anciens utilisateurs)"):
    recommended_movies, combined_scores = hybrid_recommendation(user_id)
    st.write(f"Films recommandés pour l'utilisateur {user_id} :")
    show_movie_details(recommended_movies, movies_df, combined_scores)

# Section pour recommander des films aux nouveaux utilisateurs
st.header("Recommander des films aux nouveaux utilisateurs")
if st.button("Recommander des films (nouveaux utilisateurs)"):
    top_movies = recommend_for_new_user_top_rating_movies(weighted_df, movies_df[['id', 'year']], n=10, min_year=8)
    weighted_scores = dict(zip(top_movies['id'], top_movies['score']))
    st.write("Films recommandés pour les nouveaux utilisateurs :")
    show_movie_details(top_movies['id'], movies_df, weighted_scores)

# Section pour recommander des films basées sur la similarité cosinus
st.header("Recommander des films similaires")
movie_title = st.text_input("Entrez le titre du film")
if st.button("Recommander des films similaires"):
    if movie_title in indices:
        similar_movies_df = get_recommendations(movie_title, cosine_sim)
        st.write("Films similaires recommandés :")
        st.write(similar_movies_df)
    else:
        st.write("Le titre du film n'est pas trouvé dans la base de données.")

if __name__ == "__main__":
    st.set_page_config(page_title="Système de Recommandation de Films", layout="wide")
    st.run()
