import sys, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
APP  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC  = os.path.abspath(os.path.join(ROOT, "src"))
for p in [ROOT, APP, SRC]:
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st
from app.style import page_header

st.set_page_config(page_title="About", page_icon="ℹ️", layout="wide")
page_header("ℹ️ About", "What this project is and how it was built.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Project")
    st.markdown("""
**Week-02 Reddit Mood Shift NLP** is a two-week personal learning project
exploring how emotional language differs across two Reddit communities —
r/depression and r/happy.

Built to practise:
- Classical NLP with NLTK and VADER
- Text vectorization with TF-IDF
- ML classification with sklearn
- Deep learning with TensorFlow and LSTM
- Interactive apps with Streamlit
""")

    st.subheader("Dataset")
    st.markdown("""
- 500 posts per subreddit via Reddit API (PRAW)
- Top posts from the past year
- English only
- No usernames or identifying information stored
""")

with col2:
    st.subheader("Stack")
    stack = {
        "Data":        ["praw", "pandas", "numpy"],
        "NLP":         ["nltk", "vader", "scikit-learn"],
        "ML":          ["tensorflow", "keras", "sklearn"],
        "Viz":         ["matplotlib", "seaborn", "wordcloud"],
        "App":         ["streamlit"],
        "Notebooks":   ["jupyter"],
    }
    for category, libs in stack.items():
        st.markdown(f"**{category}:** " + "  ·  ".join([f"`{l}`" for l in libs]))

    st.subheader("Structure")
    st.code("""
project/
├── src/          — NLP and analysis modules
├── notebooks/    — 31 Jupyter notebooks
├── app/          — Streamlit multi-page app
├── emotion_tool/ — Emotion illustration package
├── data/         — raw and clean CSVs
├── outputs/      — saved charts
└── docs/         — project documentation
    """)

st.divider()

st.subheader("Notebooks")
notebooks = [
    ("00–02", "Setup, fetch, clean"),
    ("03–07", "Sentiment, vocab, themes, TF-IDF, similarity"),
    ("08–10", "Classifier prep, upvotes, comments"),
    ("11–16", "sklearn, TF, LSTM, comparison, temporal"),
    ("17–19", "Summary, misclassified, confidence"),
    ("20–24", "Word shift, length, ngrams, profile, extremes"),
    ("25–28", "Readability, keywords, text stats, insights"),
    ("29–31", "Emotion tool demo, batch, app check"),
]
for nb_range, desc in notebooks:
    st.markdown(f"- **{nb_range}** — {desc}")

st.divider()
st.caption("Built as Biweek 2 of a personal NLP learning series.")