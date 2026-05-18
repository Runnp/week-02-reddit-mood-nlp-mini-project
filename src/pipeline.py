import os
import pandas as pd
from utils import ensure_dirs, save_raw, save_clean
from preprocess import preprocess_df
from analysis import add_sentiment

def run_pipeline(df_raw, verbose=True):
    """
    Full pipeline: raw DataFrame → clean → sentiment scored.
    Works on both real Reddit data and mock data.
    """
    ensure_dirs()

    if verbose:
        print(f"[1/3] Raw data    : {len(df_raw)} posts")

    # clean
    df_clean = preprocess_df(df_raw)
    save_clean(df_clean, "posts_clean.csv")
    if verbose:
        print(f"[2/3] Clean data  : {len(df_clean)} posts")

    # sentiment
    df_sentiment = add_sentiment(df_clean)
    save_clean(df_sentiment, "posts_sentiment.csv")
    if verbose:
        print(f"[3/3] Sentiment   : done")
        print(f"\nMood breakdown:")
        for mood, count in df_sentiment["mood_label"].value_counts().items():
            bar = "█" * int(count / len(df_sentiment) * 30)
            print(f"  {mood:10} {count:4d}  {bar}")

    return df_sentiment

def run_from_csv(path="data/raw/posts_raw.csv", verbose=True):
    """Load raw CSV and run full pipeline."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw data not found at {path}")
    df_raw = pd.read_csv(path)
    return run_pipeline(df_raw, verbose=verbose)

def run_from_mock(verbose=True):
    """Generate mock data and run full pipeline."""
    from mock_data import generate_all
    df_raw = generate_all(save=True)
    return run_pipeline(df_raw, verbose=verbose)