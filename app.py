import streamlit as st
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

try:
    import joblib
except ModuleNotFoundError:
    joblib = None

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')

@st.cache_data
def load_data():
    movies_path = os.path.join(DATA_DIR, 'movies.csv')
    ratings_path = os.path.join(DATA_DIR, 'ratings.csv')

    if not os.path.exists(movies_path) or not os.path.exists(ratings_path):
        st.error('Dataset files not found in data/. Please upload movies.csv and ratings.csv or run movie_recommender.py first.')
        st.stop()

    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    return movies, ratings

@st.cache_resource
def build_content_model(movies):
    movies['genres'] = movies['genres'].fillna('')
    movies['metadata'] = movies['title'].fillna('') + ' ' + movies['genres']

    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(movies['metadata'])

    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()
    return cosine_sim, indices

@st.cache_resource
def build_collab_model(ratings):
    user_item = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
    matrix = csr_matrix(user_item.values)
    n_features = matrix.shape[1]
    n_components = min(50, max(2, n_features - 1))

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_factors = svd.fit_transform(matrix)
    item_factors = svd.components_.T
    return user_item, user_factors, item_factors

@st.cache_resource
def load_models():
    models = {}
    movies, ratings = load_data()

    if joblib is not None and os.path.exists(os.path.join(MODEL_DIR, 'content_cosine_sim.joblib')):
        try:
            models['content_cosine_sim'] = joblib.load(os.path.join(MODEL_DIR, 'content_cosine_sim.joblib'))
            models['content_indices'] = joblib.load(os.path.join(MODEL_DIR, 'content_indices.joblib'))
            models['collab_user_item'] = joblib.load(os.path.join(MODEL_DIR, 'collab_user_item.joblib'))
            models['collab_user_factors'] = joblib.load(os.path.join(MODEL_DIR, 'collab_user_factors.joblib'))
            models['collab_item_factors'] = joblib.load(os.path.join(MODEL_DIR, 'collab_item_factors.joblib'))
            return models
        except Exception as e:
            st.warning(f'Could not load saved model files: {e}. Building models on-the-fly.')

    # Fallback: build models in-app if joblib is unavailable or model files missing
    content_cosine_sim, content_indices = build_content_model(movies)
    collab_user_item, collab_user_factors, collab_item_factors = build_collab_model(ratings)

    models['content_cosine_sim'] = content_cosine_sim
    models['content_indices'] = content_indices
    models['collab_user_item'] = collab_user_item
    models['collab_user_factors'] = collab_user_factors
    models['collab_item_factors'] = collab_item_factors

    return models

def recommend_content(title, movies, cosine_sim, indices, top_n=5):
    if title not in indices:
        return pd.DataFrame([], columns=['movieId','title','genres'])
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices][['movieId','title','genres']]

def recommend_hybrid(movie, genre, user_id, movies, ratings, cosine_sim, indices, collab_user_item, collab_user_factors, collab_item_factors, top_n=10):
    # build pool
    pool = movies.copy()
    if genre:
        pool = pool[pool['genres'].str.contains(genre, case=False, na=False)]
    if movie and movie in indices:
        content_back = recommend_content(movie, movies, cosine_sim, indices, top_n=top_n)
    else:
        content_back = pool.sample(min(len(pool), top_n), random_state=42) if len(pool) > 0 else movies.sample(n=top_n, random_state=42)

    if user_id in collab_user_item.index:
        idx = collab_user_item.index.get_loc(user_id)
        scores = collab_item_factors.dot(collab_user_factors[idx])
        recs = pd.DataFrame({'movieId': collab_user_item.columns, 'score': scores})
        rated = ratings[ratings['userId']==user_id]['movieId'].tolist()
        recs = recs[~recs['movieId'].isin(rated)].sort_values('score', ascending=False).head(top_n)
        recs = recs.merge(movies[['movieId','title','genres']], on='movieId')
        return recs[['movieId','title','genres']]

    return content_back.head(top_n)


def main():
    st.title('Movie Recommendation System')
    st.markdown('### Provide preferences and see recommendations.')

    movies, ratings = load_data()
    models = load_models()

    cosine_sim = models['content_cosine_sim']
    indices = models['content_indices']
    collab_user_item = models['collab_user_item']
    collab_user_factors = models['collab_user_factors']
    collab_item_factors = models['collab_item_factors']

    movie_choice = st.selectbox('Pick a movie you liked', options=[''] + sorted(movies['title'].unique().tolist()))
    genre_choice = st.selectbox('Preferred genre', options=[''] + sorted({g for row in movies['genres'] for g in (row.split('|') if pd.notna(row) else [])}))
    user_id_choice = st.number_input('User ID (optional, for collaborative suggestions)', min_value=1, step=1, value=1)
    top_n = st.slider('Number of recommendations', min_value=5, max_value=20, value=10)

    if st.button('Recommend Movies'):
        recs = recommend_hybrid(movie_choice, genre_choice, user_id_choice, movies, ratings, cosine_sim, indices, collab_user_item, collab_user_factors, collab_item_factors, top_n=top_n)
        st.write(f'Recommendations (model = hybrid): {len(recs)}')
        for _, row in recs.iterrows():
            st.write(f"- {row['title']} ({row['genres']})")

if __name__ == '__main__':
    main()
