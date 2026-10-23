# ⭐ Anime Score Predictor & Recommender

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Gradient%20Boosting-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)

An end-to-end Machine Learning web application that predicts MyAnimeList (MAL) user scores for hypothetical or upcoming anime, uncovering industry trends and biases from a 4.5 GB dataset covering 25,000+ titles and millions of user ratings.

---

## 🌟 Live Demo

👉 **[Try the Live Web App Here](https://share.streamlit.io)** *(Deploy with 1-click on Streamlit Cloud)*

---

## 📌 Project Overview

### The Question
*Can you predict how an anime will be rated by the community, and recommend shows people will actually like, using public MyAnimeList metadata?*

### What Was Built
1. **Data Pipeline & Cleaning:** Parsed, cleaned, and filtered a multi-gigabyte dataset covering 25,000+ anime titles and active user ratings.
2. **Exploratory Data Analysis:** Analyzed historical genre shifts, episode length dynamics ("survivor bias"), and source material premiums.
3. **Machine Learning Model:** Trained a **Gradient Boosting Regressor** (`GradientBoostingRegressor`) to predict scores based on format, release year, community size, source material, and genre composition.
4. **Interactive Streamlit App:** Deployed a web interface allowing anyone to input hypothetical anime specifications and get instant score predictions.

---

## 📊 Key Findings & Model Insights

* **Popularity Bias Dominates:** Community member count (`Members`) accounted for **>60% of feature importance**, showing that community anticipation and community size heavily anchor final user ratings.
* **Survivor & Sequel Bias:** Shows with 50+ episodes exhibit systematically higher median scores—primarily because only well-received franchises survive long enough to produce multiple seasons, and dedicated fans continue rating them.
* **Source Material Premium:** Manga and Light Novel adaptations maintain higher, more consistent rating baselines, while original anime exhibit substantially higher variance.
* **Evaluation Metrics:**
  * **RMSE:** ~0.55
  * **MAE:** ~0.48 (predictions are within half a rating point on a 1–10 scale)
  * **R² Score:** ~0.53

---

## 🚀 Running Locally

If you want to run this application on your local machine:

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/anime-score-predictor.git
cd anime-score-predictor
```

### 2. Create a Virtual Environment (Optional)
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the App
```bash
streamlit run app.py
```
The app will automatically open in your browser at `http://localhost:8501`.

---

## 📂 Repository Structure

```
├── app.py                # Streamlit web application
├── anime_model.pkl       # Trained Gradient Boosting model
├── model_columns.pkl     # Feature column alignment for inference
├── requirements.txt      # Python dependencies for deployment
├── .gitignore            # Excludes large CSV files and virtual environments
└── README.md             # Project documentation
```

---

## 🛠️ Built With

* **Python 3**
* **Streamlit** — Web Application UI
* **Scikit-Learn** — Gradient Boosting Regression
* **Pandas & NumPy** — Data manipulation and feature engineering
* **Joblib** — Model serialization
