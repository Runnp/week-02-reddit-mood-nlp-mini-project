import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

def score_sentiment(text):
    """Return VADER scores for a single text."""
    if not isinstance(text, str) or text.strip() == "":
        return {"neg": 0, "neu": 1, "pos": 0, "compound": 0}
    return sia.polarity_scores(text)

def add_sentiment(df, text_col="clean_text"):
    """Add VADER score columns to a DataFrame."""
    print(f"Scoring sentiment on {len(df)} posts...")
    scores = df[text_col].apply(score_sentiment)
    df = df.copy()
    df["vader_neg"]      = scores.apply(lambda s: s["neg"])
    df["vader_neu"]      = scores.apply(lambda s: s["neu"])
    df["vader_pos"]      = scores.apply(lambda s: s["pos"])
    df["vader_compound"] = scores.apply(lambda s: s["compound"])
    df["mood_label"]     = df["vader_compound"].apply(label_mood)
    print("Done. Mood distribution:")
    print(df["mood_label"].value_counts())
    return df

def label_mood(compound):
    """Convert compound score to a mood label."""
    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    else:
        return "neutral"

def monthly_sentiment(df):
    """Average compound score per subreddit per month."""
    return (
        df.groupby(["subreddit", "month"])["vader_compound"]
        .mean()
        .reset_index()
        .rename(columns={"vader_compound": "avg_compound"})
        .sort_values("month")
    )