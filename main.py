import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from utils import ensure_dirs

NOTEBOOKS = [
    ("00_setup",                  "verify environment and Reddit connection"),
    ("01_fetch",                  "pull 500 posts per subreddit from Reddit"),
    ("02_clean",                  "clean and tokenize text"),
    ("03_sentiment",              "VADER scoring and mood-over-time chart"),
    ("04_vocab",                  "word clouds and frequency analysis"),
    ("05_themes",                 "theme heatmap across months"),
    ("06_tfidf",                  "TF-IDF signature words and bigrams"),
    ("07_similarity",             "cosine similarity between communities"),
    ("08_classify_prep",          "label encoding and train/test split"),
    ("09_upvotes",                "sentiment vs upvote correlation"),
    ("10_comments",               "comment analysis and engagement score"),
    ("11_classifier_sklearn",     "logistic regression and random forest"),
    ("12_tensorflow",             "embedding neural network"),
    ("13_predict",                "interactive mood predictor"),
    ("14_lstm",                   "bidirectional LSTM classifier"),
    ("15_model_comparison",       "all models compared side by side"),
    ("16_temporal_classifier",    "mood over time with LSTM predictions"),
    ("17_summary",                "master dashboard and key findings"),
    ("18_misclassified",          "error analysis and misclassification"),
    ("19_confidence",             "prediction confidence distribution"),
    ("20_wordshift",              "word shift between communities"),
    ("21_post_length_over_time",  "post length trends and correlation"),
    ("22_ngram_analysis",         "bigram and trigram comparison"),
    ("23_subreddit_profile",      "radar chart and profile comparison"),
    ("24_extreme_posts",          "most positive, negative, upvoted posts"),
    ("25_readability",            "word length, sentence length, TTR"),
    ("26_monthly_keywords",       "top TF-IDF word per month"),
    ("27_text_stats",             "text stats radar and correlation"),
    ("28_final_insights",         "three key insights with interpretations"),
    ("29_emotion_tool_demo",      "emotion tool — single text demo"),
    ("30_emotion_batch",          "emotion tool — full dataset batch"),
]

SRC_FILES = [
    "config.py",
    "utils.py",
    "preprocess.py",
    "analysis.py",
    "compare.py",
    "mock_data.py",
    "report.py",
    "visualize.py",
    "logger.py",
    "sentiment_utils.py",
    "text_stats.py",
]

EMOTION_FILES = [
    "emotion_tool/__init__.py",
    "emotion_tool/config.py",
    "emotion_tool/analyzer.py",
    "emotion_tool/charts.py",
    "emotion_tool/batch.py",
    "emotion_tool/runner.py",
]

def print_banner():
    print("=" * 52)
    print("  WEEK-02 REDDIT MOOD SHIFT NLP")
    print("  r/depression vs r/happy — NLP + ML study")
    print("=" * 52)

def print_data_summary():
    print("\n── Dataset ──────────────────────────────────")
    try:
        import pandas as pd
        df = pd.read_csv("data/clean/posts_sentiment.csv")
        print(f"  Total posts  : {len(df)}")
        for sub, count in df["subreddit"].value_counts().items():
            print(f"  r/{sub:14}: {count} posts")
        if "mood_label" in df.columns:
            print("\n  Mood breakdown:")
            for mood, count in df["mood_label"].value_counts().items():
                bar = "█" * int(count / len(df) * 20)
                print(f"    {mood:10}: {count:4d}  {bar}")
        if "month" in df.columns:
            months = sorted(df["month"].dropna().unique())
            print(f"\n  Date range   : {months[0]}  →  {months[-1]}")
            print(f"  Months       : {len(months)}")
    except FileNotFoundError:
        print("  No clean data yet.")
        print("  Run: python src/mock_data.py   to generate mock data")
        print("  Or:  run notebook 01_fetch     to pull real Reddit data")

def print_model_status():
    print("\n── Models ───────────────────────────────────")
    models = [
        ("data/clean/best_sklearn_model.pkl", "sklearn (LR / RF)"),
        ("data/clean/tf_mood_model",          "TF embedding model"),
        ("data/clean/lstm_mood_model",        "Bidirectional LSTM"),
        ("data/clean/train_test_split.pkl",   "train/test split"),
        ("data/clean/tf_tokenizer.pkl",       "TF tokenizer"),
    ]
    for path, label in models:
        status = "OK     " if os.path.exists(path) else "MISSING"
        print(f"  [{status}] {label}")

def print_src_status():
    print("\n── Source files ─────────────────────────────")
    for f in SRC_FILES:
        path   = os.path.join("src", f)
        status = "OK     " if os.path.exists(path) else "MISSING"
        print(f"  [{status}] src/{f}")
    for f in EMOTION_FILES:
        status = "OK     " if os.path.exists(f) else "MISSING"
        print(f"  [{status}] {f}")

def print_outputs():
    print("\n── Outputs ──────────────────────────────────")
    if os.path.exists("outputs"):
        charts = sorted(f for f in os.listdir("outputs") if f.endswith(".png"))
        csvs   = sorted(f for f in os.listdir("outputs") if f.endswith(".csv"))
        print(f"  Charts : {len(charts)}")
        print(f"  CSVs   : {len(csvs)}")
        if charts:
            print(f"  Latest : {charts[-1]}")
    else:
        print("  outputs/ folder not found.")

def print_notebooks():
    print("\n── Notebooks ────────────────────────────────")
    for name, desc in NOTEBOOKS:
        path   = os.path.join("notebooks", f"{name}.ipynb")
        status = "OK " if os.path.exists(path) else "   "
        print(f"  {status} {name:30} {desc}")

def print_quick_start():
    print("\n── Quick start ──────────────────────────────")
    print("  1. jupyter notebook")
    print("  2. python src/mock_data.py       (offline mock data)")
    print("  3. python src/report.py          (full health check)")
    print("  4. python emotion_tool/runner.py (emotion CLI tool)")
    print("  5. cd app && streamlit run streamlit_app.py")
    print("=" * 52)

def main():
    print_banner()
    ensure_dirs()
    print_data_summary()
    print_model_status()
    print_src_status()
    print_outputs()
    print_notebooks()
    print_quick_start()

if __name__ == "__main__":
    main()