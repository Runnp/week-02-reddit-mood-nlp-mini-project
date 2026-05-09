import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

def score_text(text):
    """Score a single string with VADER."""
    if not isinstance(text, str) or not text.strip():
        return {"neg": 0, "neu": 1, "pos": 0, "compound": 0}
    return sia.polarity_scores(text)

def label_from_compound(compound, pos_thresh=0.05, neg_thresh=-0.05):
    """Convert compound score to mood label."""
    if compound >= pos_thresh:
        return "positive"
    elif compound <= neg_thresh:
        return "negative"
    return "neutral"

def score_dataframe(df, text_col="clean_text"):
    """Add VADER columns and mood label to a DataFrame."""
    df = df.copy()
    scores = df[text_col].apply(score_text)
    df["vader_neg"]      = scores.apply(lambda s: s["neg"])
    df["vader_neu"]      = scores.apply(lambda s: s["neu"])
    df["vader_pos"]      = scores.apply(lambda s: s["pos"])
    df["vader_compound"] = scores.apply(lambda s: s["compound"])
    df["mood_label"]     = df["vader_compound"].apply(label_from_compound)
    return df

def monthly_avg(df, group_col="subreddit"):
    """Return monthly avg compound score per group."""
    return (
        df.groupby([group_col, "month"])["vader_compound"]
        .mean()
        .reset_index()
        .rename(columns={"vader_compound": "avg_compound"})
        .sort_values("month")
    )

def sentiment_summary(df):
    """Print a clean sentiment summary per subreddit."""
    for sub in df["subreddit"].unique():
        data = df[df["subreddit"] == sub]
        avg  = data["vader_compound"].mean()
        dist = data["mood_label"].value_counts(normalize=True)
        print(f"\nr/{sub}")
        print(f"  avg compound : {avg:+.3f}")
        for mood in ["positive","neutral","negative"]:
            pct = dist.get(mood, 0)
            bar = "█" * int(pct * 20)
            print(f"  {mood:10} {pct:.0%}  {bar}")

def top_sentiment_posts(df, sub, mood, n=5):
    """Return the n most extreme posts for a given subreddit and mood."""
    data = df[(df["subreddit"]==sub) & (df["mood_label"]==mood)]
    if mood == "positive":
        return data.nlargest(n, "vader_compound")[["title","vader_compound","month"]]
    elif mood == "negative":
        return data.nsmallest(n, "vader_compound")[["title","vader_compound","month"]]
    return data.sample(min(n, len(data)))[["title","vader_compound","month"]]