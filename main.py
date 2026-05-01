import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from utils import ensure_dirs

NOTEBOOKS = [
    ("00_setup",          "verify environment and Reddit connection"),
    ("01_fetch",          "pull 500 posts per subreddit from Reddit"),
    ("02_clean",          "clean and tokenize text"),
    ("03_sentiment",      "VADER scoring and mood-over-time chart"),
    ("04_vocab",          "word clouds and frequency analysis"),
    ("05_themes",         "theme heatmap across months"),
    ("06_tfidf",          "TF-IDF signature words and bigrams"),
    ("07_similarity",     "cosine similarity between communities"),
    ("08_classify_prep",  "label encoding and train/test split"),
    ("09_upvotes",        "sentiment vs upvote correlation"),
    ("10_comments",       "comment analysis and engagement score"),
]

def print_notebooks():
    print("\nNotebook order:")
    for name, desc in NOTEBOOKS:
        print(f"  {name:25} {desc}")

def print_data_summary():
    try:
        import pandas as pd
        df = pd.read_csv("data/clean/posts_sentiment.csv")
        print("\nCurrent dataset:")
        print(f"  Total posts  : {len(df)}")
        for sub, count in df["subreddit"].value_counts().items():
            print(f"  r/{sub:14}: {count} posts")
        if "mood_label" in df.columns:
            print("\n  Mood breakdown:")
            for mood, count in df["mood_label"].value_counts().items():
                print(f"    {mood:10}: {count}")
        if "month" in df.columns:
            months = sorted(df["month"].dropna().unique())
            print(f"\n  Date range   : {months[0]} → {months[-1]}")
    except FileNotFoundError:
        print("\n  No clean data yet — run 01_fetch + 02_clean first")
        print("  Or run: python src/mock_data.py to generate mock data")

def main():
    print("=" * 48)
    print("  WEEK-02 REDDIT MOOD SHIFT NLP")
    print("=" * 48)

    print("\nChecking folders...")
    ensure_dirs()

    print_data_summary()
    print_notebooks()

    print("\nTo start:")
    print("  jupyter notebook")
    print("=" * 48)

if __name__ == "__main__":
    main()