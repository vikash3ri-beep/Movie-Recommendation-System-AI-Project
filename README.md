# Movie Recommendation System (Python + ML + Streamlit)

## Project structure

- `data/`
  - `movies.csv` (from MovieLens)
  - `ratings.csv` (from MovieLens)
- `model/`
  - Saved models (.joblib) after training
- `movie_recommender.py` - Full pipeline script (load, preprocess, EDA, models, save/load)
- `app.py` - Streamlit app for user interaction
- `README.md` - Project documentation

## Dataset

Use MovieLens data from:
- https://www.kaggle.com/datasets/grouplens/movielens-20m

## Setup

1. `pip install pandas numpy scikit-learn scipy seaborn matplotlib joblib streamlit`
2. Put `movies.csv`, `ratings.csv` inside `data/`.
3. Run:
   - `python movie_recommender.py` (build and save models, run sample recs)
   - `streamlit run app.py` (launch web UI)

## Features implemented

1. Loading + preprocessing (missing values, duplicates, type normalization)
2. Exploratory Data Analysis (rating distribution, genres, top movies)
3. Content-based recommendations (TF-IDF, cosine sim)
4. Collaborative filtering (SVD factorization)
5. Hybrid recommendation function
6. Evaluation metrics (RMSE, MAE)
7. Save and load model objects with joblib
8. Streamlit app with dropdowns, sliders, top-N recommendations

## Notes

- Ensure large dataset processing may be heavy in single-run, adjust for MovieLens 1M or subset for quick testing.
- This code is beginner-friendly and includes production-level patterns (modular function, caching, saved models).
