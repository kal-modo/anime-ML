import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page config
st.set_page_config(page_title="Anime Score Predictor", page_icon="⭐", layout="centered")

st.title("⭐ Anime Score Predictor")
st.write("Enter the details of a hypothetical anime to see what my Machine Learning model predicts it would score!")

# Load the model and columns
try:
    model = joblib.load('anime_model.pkl')
    model_columns = joblib.load('model_columns.pkl')
except FileNotFoundError:
    st.error("Model files not found! Please run the save cell in your Jupyter Notebook first.")
    st.stop()

# User Inputs
col1, col2 = st.columns(2)
with col1:
    episodes = st.number_input("Episodes", min_value=1, max_value=1000, value=12)
    members = st.number_input("Community Members", min_value=100, max_value=5000000, value=50000)
    air_year = st.number_input("Release Year", min_value=1960, max_value=2025, value=2023)

with col2:
    anime_type = st.selectbox("Format Type", ["TV", "Movie", "OVA", "ONA", "Special"])
    source = st.selectbox("Source Material", ["Manga", "Original", "Light novel", "Game", "Visual novel"])
    is_sequel = st.checkbox("Is this a Sequel?")

# Multi-select for genres
available_genres = [
    "Action", "Adventure", "Comedy", "Drama", "Fantasy", "Horror", 
    "Mystery", "Romance", "Sci-Fi", "Slice of Life", "Sports", "Supernatural"
]
selected_genres = st.multiselect("Select Genres", available_genres, default=["Action", "Fantasy"])

# Predict Button
if st.button("Predict Score!", type="primary"):
    # 1. Create a base dictionary with 0s for all model columns
    input_data = {col: 0 for col in model_columns}
    
    # 2. Fill in the numerical inputs
    input_data['Episodes'] = episodes
    input_data['Members'] = members
    input_data['air_year'] = air_year
    if 'Is_Sequel' in input_data:
        input_data['Is_Sequel'] = 1 if is_sequel else 0
    
    # 3. Fill in the categorical dummy inputs
    type_col = f"Type_{anime_type}"
    if type_col in input_data: input_data[type_col] = 1
        
    source_col = f"Source_{source}"
    if source_col in input_data: input_data[source_col] = 1
        
    # 4. Fill in genres
    for genre in selected_genres:
        if genre in input_data:
            input_data[genre] = 1
            
    # Convert to DataFrame matching model columns
    input_df = pd.DataFrame([input_data])[model_columns]
    
    # Predict!
    prediction = model.predict(input_df)[0]
    prediction = np.clip(prediction, 0, 10)
    
    # Show result
    st.success(f"### Predicted MAL Score: {prediction:.2f} / 10")
    if prediction >= 8.0:
        st.balloons()
