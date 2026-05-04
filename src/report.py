import os
import sys
import pickle
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_clean

DIVIDER = "=" * 52
THIN    = "─" * 52

def section(title):
    print(f"\n{THIN}")
    print(f"  {title}")
    print(THIN)

def check_file(path, label):
    exists = os.path.exists(path)
    status = "OK" if exists else "MISSING"
    print(f"  [{status:7}] {label}")
    return exists

def run_report():
    print(f"\n{DIVIDER}")
    print("  WEEK-02 REDDIT MOOD SHIFT NLP")
    print("  Project Report")
    print(DIVIDER)

    # ── File health check ──────────────────────────
    section("File health check")
    files = [
        ("data/raw/posts_raw.csv",                "Raw posts CSV"),
        ("data/clean/posts_clean.csv",            "Clean posts CSV"),
        ("data/clean/posts_sentiment.csv",        "Sentiment CSV"),
        ("data/clean/train_test_split.pkl",       "Train/test split"),
        ("data/clean/best_sklearn_model.pkl",     "sklearn model"),
        ("data/clean/tf_tokenizer.pkl",           "TF tokenizer"),
        ("data/clean/tf_mood_model",              "TF embedding model"),
        ("data/clean/lstm_mood_model",            "LSTM model"),
    ]
    all_ok = all(check_file(path, label) for path, label in files)
    if all_ok:
        print(f"\n  All files present.")
    else:
        print(f"\n  Some files missing — run missing notebooks first.")

    # ── Dataset summary ────────────────────────────
    section("Dataset summary")
    try:
        df = load_clean("posts_sentiment.csv")

        print(f"  Total posts  : {len(df)}")
        for sub, count in df["subreddit"].value_counts().items():
            print(f"  r/{sub:14}: {count} posts")

        if "month" in df.columns:
            months = sorted(df["month"].dropna().unique())
            print(f"  Date range   : {months[0]}  →  {months[-1]}")
            print(f"  Months       : {len(months)}")

        if "token_count" in df.columns:
            print(f"\n  Avg token count:")
            for sub in ["depression", "happy"]:
                avg = df[df["subreddit"] == sub]["token_count"].mean()
                print(f"    r/{sub:14}: {avg:.1f} tokens/post")

    except FileNotFoundError:
        print("  posts_sentiment.csv not found — run notebooks 01-03 first.")

    # ── Sentiment summary ──────────────────────────
    section("Sentiment summary (VADER)")
    try:
        df = load_clean("posts_sentiment.csv")

        for sub in ["depression", "happy"]:
            data  = df[df["subreddit"] == sub]
            avg   = data["vader_compound"].mean()
            pos   = (data["mood_label"] == "positive").sum()
            neu   = (data["mood_label"] == "neutral").sum()
            neg   = (data["mood_label"] == "negative").sum()
            total = len(data)
            print(f"\n  r/{sub}")
            print(f"    avg compound : {avg:+.3f}")
            print(f"    positive     : {pos:4d}  ({pos/total:.0%})")
            print(f"    neutral      : {neu:4d}  ({neu/total:.0%})")
            print(f"    negative     : {neg:4d}  ({neg/total:.0%})")

    except FileNotFoundError:
        print("  Sentiment data not found.")

    # ── Model accuracy summary ─────────────────────
    section("Model accuracy summary")
    try:
        with open("data/clean/best_sklearn_model.pkl", "rb") as f:
            sk = pickle.load(f)
        print(f"  sklearn best model : {sk['accuracy']:.1%}")
    except FileNotFoundError:
        print("  sklearn model not found.")

    try:
        import tensorflow as tf
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from sklearn.metrics import accuracy_score
        import pickle as pkl

        with open("data/clean/train_test_split.pkl", "rb") as f:
            split = pkl.load(f)
        with open("data/clean/tf_tokenizer.pkl", "rb") as f:
            tf_data = pkl.load(f)

        X_test    = split["X_test"]
        y_test    = split["y_test"]
        MAX_LEN   = tf_data["max_len"]
        tokenizer = tf_data["tokenizer"]

        X_seq = pad_sequences(
            tokenizer.texts_to_sequences(X_test),
            maxlen=MAX_LEN, padding="post", truncating="post"
        )

        for label, path in [("TF embedding", "data/clean/tf_mood_model"),
                             ("LSTM         ", "data/clean/lstm_mood_model")]:
            if os.path.exists(path):
                m     = tf.keras.models.load_model(path)
                preds = m.predict(X_seq, verbose=0).argmax(axis=1)
                acc   = accuracy_score(y_test, preds)
                print(f"  {label}      : {acc:.1%}")

    except FileNotFoundError:
        print("  TF models not found.")

    # ── Output files ───────────────────────────────
    section("Generated outputs")
    if os.path.exists("outputs"):
        charts = [f for f in os.listdir("outputs") if f.endswith(".png")]
        if charts:
            for chart in sorted(charts):
                print(f"  {chart}")
            print(f"\n  Total charts : {len(charts)}")
        else:
            print("  No charts yet — run the notebooks.")
    else:
        print("  outputs/ folder not found.")

    # ── Notebook status ────────────────────────────
    section("Notebook status")
    notebooks = [
        "00_setup", "01_fetch", "02_clean", "03_sentiment",
        "04_vocab", "05_themes", "06_tfidf", "07_similarity",
        "08_classify_prep", "09_upvotes", "10_comments",
        "11_classifier_sklearn", "12_tensorflow", "13_predict",
        "14_lstm", "15_model_comparison", "16_temporal_classifier",
    ]
    for nb in notebooks:
        path = f"notebooks/{nb}.ipynb"
        check_file(path, nb)

    print(f"\n{DIVIDER}")
    print("  Run: jupyter notebook  to continue")
    print(DIVIDER + "\n")


if __name__ == "__main__":
    run_report()