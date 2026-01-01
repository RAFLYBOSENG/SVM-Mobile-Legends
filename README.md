# TA-13 Sentiment Analysis (Streamlit App)

This Streamlit app demonstrates the TA-13 pipeline: preprocessing (cleaning, stopword removal, stemming), TF-IDF feature extraction, SVM training (Linear & RBF), hyperparameter tuning, evaluation, and WordCloud visualizations.

## Files

- `streamlit_app.py` : Streamlit application (in `Program/`).
- `preprocessing_stages/03_full_preprocessing.csv` : Recommended preprocessing snapshot (auto-used if present).
- `dataset/sentiment-ML.csv` : Raw dataset (ulasan) used if preprocessing snapshot not found.
- `requirements.txt` : Python dependencies.

## Install

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r Program/requirements.txt
python -m nltk.downloader punkt
python -m nltk.downloader stopwords
```

Install Sastrawi and wordcloud if needed:

```bash
pip install Sastrawi wordcloud
```

## Run

Start the Streamlit app:

```bash
streamlit run Program/streamlit_app.py
```

## Notes

- If `preprocessing_stages/03_full_preprocessing.csv` exists, the app will load it (faster). Otherwise it will preprocess `dataset/sentiment-ML.csv` on the fly.
- Sastrawi is optional; if not installed the app falls back to basic tokenization.
- GridSearchCV option can be slow; use only when needed.

## Next steps (optional)

- Commit changes and push to GitHub.
- Add a small UI polish and caching for faster re-runs.
