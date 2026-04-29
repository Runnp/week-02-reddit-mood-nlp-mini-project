# week-02-reddit-mood-shift-nlp

A Python + Jupyter NLP project that tracks how emotional tone shifts over time in r/depression and r/happy — then builds a mood classifier using TF-IDF and TensorFlow.

Built as a 2-week learning project exploring classical NLP, sentiment analysis, and introductory deep learning.

---

## What it does

- Fetches real posts from r/depression and r/happy via the Reddit API
- Cleans and tokenizes text using NLTK
- Scores sentiment on every post using VADER
- Tracks mood trends over time with monthly aggregation
- Compares vocabulary, themes, and linguistic style between the two communities
- Measures how similar/different the communities are using cosine similarity
- Trains a mood classifier (positive / neutral / negative) using TF-IDF + sklearn and TensorFlow/Keras

---

## Project structure

```
week-02-reddit-mood-shift-nlp/
├── src/
│   ├── config.py          # settings and API keys loader
│   ├── utils.py           # reusable load/save/summary helpers
│   ├── preprocess.py      # text cleaning and tokenization
│   ├── analysis.py        # VADER sentiment scoring
│   ├── compare.py         # cross-subreddit comparison logic
│   └── mock_data.py       # generates mock posts for offline dev
├── notebooks/
│   ├── 00_setup.ipynb     # verify environment and Reddit connection
│   ├── 01_fetch.ipynb     # pull posts from Reddit API
│   ├── 02_clean.ipynb     # preprocess and clean text
│   ├── 03_sentiment.ipynb # VADER scoring and mood-over-time charts
│   ├── 04_vocab.ipynb     # word clouds and frequency analysis
│   ├── 05_themes.ipynb    # theme heatmap (support, venting, recovery...)
│   ├── 06_tfidf.ipynb     # TF-IDF signature words and bigrams
│   ├── 07_similarity.ipynb# cosine similarity between communities
│   └── 08_classify_prep.ipynb # label encoding and train/test split
├── data/
│   ├── raw/               # fetched CSV files (gitignored)
│   └── clean/             # processed CSVs and pickle files (gitignored)
├── outputs/               # saved charts as PNG (gitignored)
├── main.py                # project entry point
├── requirements.txt
├── .env                   # Reddit API keys (never committed)
└── .gitignore
```

---

## Setup

### 1. Clone and create virtual environment

```bash
git clone https://github.com/yourusername/week-02-reddit-mood-shift-nlp
cd week-02-reddit-mood-shift-nlp

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

### 2. Download NLTK data

```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon'); nltk.download('punkt_tab')"
```

### 3. Set up Reddit API keys

Go to [reddit.com/prefs/apps](https://reddit.com/prefs/apps), create a **script** app, then create a `.env` file in the project root:

```
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=mood_study_bot/0.1
```

### 4. Verify setup

```bash
python main.py
```

Then open Jupyter and run `notebooks/00_setup.ipynb` — all cells should pass.

---

## Running without Reddit API

If you want to explore the project without connecting to Reddit, generate mock data first:

```bash
python src/mock_data.py
```

This creates a realistic `data/raw/posts_raw.csv` with 1000 posts and lets you run every notebook from `02_clean.ipynb` onwards.

---

## Running the notebooks

Run notebooks in order from the `notebooks/` folder:

| Notebook | What it does | Key output |
|---|---|---|
| `00_setup` | Environment check | confirms all imports and Reddit connection |
| `01_fetch` | Pulls 500 posts per subreddit | `data/raw/posts_raw.csv` |
| `02_clean` | Cleans and tokenizes text | `data/clean/posts_clean.csv` |
| `03_sentiment` | VADER scoring + mood over time | `mood_over_time.png` |
| `04_vocab` | Word clouds + frequency | `wordclouds.png`, `top_words.png` |
| `05_themes` | Theme tagging + heatmap | `theme_heatmap.png` |
| `06_tfidf` | TF-IDF signature words | `tfidf_terms.png` |
| `07_similarity` | Cosine similarity matrix | `similarity_heatmap.png` |
| `08_classify_prep` | Label + split data | `train_test_split.pkl` |

---

## Libraries used

| Library | Purpose |
|---|---|
| `praw` | Reddit API access |
| `nltk` | Tokenization, stopwords, VADER sentiment |
| `pandas` / `numpy` | Data manipulation |
| `matplotlib` / `seaborn` | Visualizations |
| `wordcloud` | Word cloud generation |
| `scikit-learn` | TF-IDF, cosine similarity, classifiers |
| `tensorflow` / `keras` | Neural network mood classifier |
| `python-dotenv` | Loads API keys from `.env` |

---

## Sample findings

> These are from mock data — real Reddit results will vary.

- r/depression posts average **3x longer** than r/happy posts by token count
- The two communities score **0.31 cosine similarity** — meaningfully different vocabulary
- Top r/depression bigrams include phrases around exhaustion and isolation
- Top r/happy bigrams center around achievement and connection
- Mood classifier reaches ~78% accuracy on the test set

---

## Notes on data and ethics

- All data is collected from public Reddit posts via the official API
- No usernames, profile data, or identifying information is stored
- Raw data is gitignored and stays local on your machine
- This project is for educational NLP study only — not clinical research

---

## Author

Built as Biweek 2 of a personal NLP learning series.
