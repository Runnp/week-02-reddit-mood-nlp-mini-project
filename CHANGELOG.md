# Changelog

All notable changes to this project are documented here.
Format: Push number — description — date.

---

## Week 1 — Data & classic NLP

| Push | Description | Day |
|---|---|---|
| 1 | venv setup, requirements, jupyter running | Day 1 |
| 2 | Reddit API connected, 00_setup notebook | Day 1 |
| 3 | config.py and utils.py added | Day 1 |
| 4 | main.py wired to config and utils | Day 1 |
| 5 | 00_setup.ipynb all cells verified | Day 1 |
| 6 | preprocess.py with text cleaning | Day 2 |
| 7 | 01_fetch.ipynb — 1000 posts pulled | Day 2 |
| 8 | 02_clean.ipynb — text cleaned | Day 2 |
| 9 | mock_data.py — 1000 realistic posts | Day 3 |
| 10 | analysis.py — VADER scoring | Day 3 |
| 11 | 03_sentiment.ipynb — mood charts | Day 3 |
| 12 | 04_vocab.ipynb — word clouds | Day 4 |
| 13 | 05_themes.ipynb — theme heatmap | Day 4 |
| 14 | compare.py — cross-subreddit logic | Day 4 |
| 15 | 06_tfidf.ipynb — TF-IDF terms | Day 5 |
| 16 | 07_similarity.ipynb — cosine similarity | Day 5 |
| 17 | 08_classify_prep.ipynb — train/test split | Day 5 |
| 18 | 09_upvotes.ipynb — sentiment vs upvotes | Day 6 |
| 19 | 10_comments.ipynb — engagement score | Day 6 |
| 20 | main.py summary runner, notebooks cleaned | Day 6 |

## Week 2 — ML & classifiers

| Push | Description | Day |
|---|---|---|
| 21 | 11_classifier_sklearn.ipynb — LR and RF | Day 7 |
| 22 | 12_tensorflow.ipynb — embedding model | Day 7 |
| 23 | 13_predict.ipynb — interactive predictor | Day 7 |
| 24 | 14_lstm.ipynb — bidirectional LSTM | Day 8 |
| 25 | 15_model_comparison.ipynb | Day 8 |
| 26 | 16_temporal_classifier.ipynb | Day 8 |
| 27 | report.py — project health check | Day 8 |
| 28 | 17_summary.ipynb — master dashboard | Day 9 |
| 29 | README updated | Day 9 |
| 30 | v1.0 release tag | Day 9 |

## Post v1.0

| Push | Description |
|---|---|
| 31 | 18_misclassified.ipynb — error analysis |
| 32 | 19_confidence.ipynb — confidence scores |
| 33 | visualize.py — matplotlib helpers |
| 34 | report.py updated with new notebooks |
| 35 | docs/FINDINGS.md |
| 36 | docs/NOTEBOOKS.md |
| 37 | 20_wordshift.ipynb |
| 38 | 21_post_length_over_time.ipynb |
| 39 | logger.py |
| 40 | app/streamlit_app.py skeleton |
| 41 | docs/SETUP.md |
| 42 | Streamlit overview page |
| 43 | Streamlit mood over time page |
| 44 | Streamlit live predictor page |
| 45 | 22_ngram_analysis.ipynb |
| 46 | app/README.md |
| 47 | Streamlit word explorer page |
| 48 | 23_subreddit_profile.ipynb — radar chart |
| 49 | sentiment_utils.py |
| 50 | 24_extreme_posts.ipynb |
| 51 | Streamlit themes page |
| 52 | Streamlit similarity page |
| 53 | 25_readability.ipynb |
| 54 | 26_monthly_keywords.ipynb |
| 55 | text_stats.py |
| 56 | 27_text_stats.ipynb |
| 57 | docs/CHANGELOG.md |
| 83 | pages/08_reddit_fetch.py — live fetch UI |
| 84 | pages/09_live_stats.py — real-time health |
| 85 | pages/10_settings.py — app settings |
| 86 | app/main.py updated, CHANGELOG current |
| 87 | pages/11_about.py — project info page |
| 88 | pages __init__.py, sidebar navigation |
| 89 | CHANGELOG and README final update |

files = [
    ("data/raw/posts_raw.csv",            "Raw posts CSV"),
    ("data/clean/posts_clean.csv",        "Clean posts CSV"),
    ("data/clean/posts_sentiment.csv",    "Sentiment CSV"),
    ("data/clean/train_test_split.pkl",   "Train/test split"),
    ("data/clean/best_sklearn_model.pkl", "sklearn model"),
    ("data/clean/tf_tokenizer.pkl",       "TF tokenizer"),
    ("data/clean/tf_mood_model",          "TF embedding model"),
    ("data/clean/lstm_mood_model",        "LSTM model"),
    ("src/config.py",                     "config.py"),
    ("src/utils.py",                      "utils.py"),
    ("src/preprocess.py",                 "preprocess.py"),
    ("src/analysis.py",                   "analysis.py"),
    ("src/compare.py",                    "compare.py"),
    ("src/mock_data.py",                  "mock_data.py"),
    ("src/report.py",                     "report.py"),
    ("src/visualize.py",                  "visualize.py"),
    ("src/logger.py",                     "logger.py"),
    ("src/sentiment_utils.py",            "sentiment_utils.py"),
    ("src/text_stats.py",                 "text_stats.py"),
    ("app/streamlit_app.py",              "streamlit app"),
]