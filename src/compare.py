import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

COLORS = {"depression": "#E07070", "happy": "#7BC67E"}

def avg_sentiment_by_sub(df):
    """Return mean compound score per subreddit."""
    return df.groupby("subreddit")["vader_compound"].mean().round(3)

def mood_distribution_by_sub(df):
    """Return mood label counts per subreddit as a pivot table."""
    return (
        df.groupby(["subreddit", "mood_label"])
        .size()
        .unstack(fill_value=0)
    )

def top_words_by_sub(df, n=20):
    """Return top n words per subreddit as a dict of Counter."""
    result = {}
    for sub in df["subreddit"].unique():
        text = " ".join(df[df["subreddit"] == sub]["clean_text"].dropna())
        result[sub] = Counter(text.split()).most_common(n)
    return result

def vocabulary_overlap(df):
    """Return unique and shared word counts across subreddits."""
    subs = df["subreddit"].unique()
    word_sets = {}
    for sub in subs:
        text = " ".join(df[df["subreddit"] == sub]["clean_text"].dropna())
        word_sets[sub] = set(text.split())

    subs = list(subs)
    only_a  = word_sets[subs[0]] - word_sets[subs[1]]
    only_b  = word_sets[subs[1]] - word_sets[subs[0]]
    shared  = word_sets[subs[0]] & word_sets[subs[1]]

    return {
        f"only_{subs[0]}": len(only_a),
        f"only_{subs[1]}": len(only_b),
        "shared":          len(shared),
    }

def monthly_mood_trend(df):
    """Return avg compound score per subreddit per month."""
    return (
        df.groupby(["subreddit", "month"])["vader_compound"]
        .mean()
        .reset_index()
        .rename(columns={"vader_compound": "avg_compound"})
        .sort_values("month")
    )

def print_summary(df):
    """Print a clean comparison summary to console."""
    print("=" * 45)
    print("  REDDIT MOOD STUDY — SUMMARY")
    print("=" * 45)

    print("\nAvg sentiment score:")
    for sub, score in avg_sentiment_by_sub(df).items():
        bar = "█" * int(abs(score) * 40)
        sign = "+" if score > 0 else ""
        print(f"  r/{sub:14} {sign}{score:.3f}  {bar}")

    print("\nMood distribution:")
    dist = mood_distribution_by_sub(df)
    for sub in dist.index:
        print(f"  r/{sub}")
        for mood in dist.columns:
            print(f"    {mood:10}: {dist.loc[sub, mood]}")

    print("\nVocabulary overlap:")
    overlap = vocabulary_overlap(df)
    for k, v in overlap.items():
        print(f"  {k:25}: {v} words")

    print("=" * 45)