import pandas as pd
from collections import Counter

def one_liner(df):
    """Return a single sentence summarizing the dataset."""
    if df is None or len(df) == 0:
        return "No data loaded."

    avg_dep = df[df["subreddit"]=="depression"]["vader_compound"].mean()
    avg_hap = df[df["subreddit"]=="happy"]["vader_compound"].mean()
    gap     = abs(avg_dep - avg_hap)
    months  = df["month"].nunique()

    return (
        f"Across {len(df)} posts and {months} months, "
        f"r/depression averaged {avg_dep:+.3f} compound sentiment "
        f"vs r/happy at {avg_hap:+.3f} — "
        f"a gap of {gap:.3f} points."
    )

def key_numbers(df):
    """Return a dict of the most important project numbers."""
    if df is None:
        return {}
    return {
        "total_posts":     len(df),
        "subreddits":      df["subreddit"].nunique(),
        "months":          df["month"].nunique(),
        "avg_dep":         round(df[df["subreddit"]=="depression"]["vader_compound"].mean(), 3),
        "avg_hap":         round(df[df["subreddit"]=="happy"]["vader_compound"].mean(), 3),
        "pct_negative":    round((df["mood_label"]=="negative").mean(), 3),
        "pct_positive":    round((df["mood_label"]=="positive").mean(), 3),
        "most_common_mood": df["mood_label"].value_counts().idxmax(),
    }

def top_insight(df):
    """Return the single most striking finding."""
    if df is None:
        return "No data."
    nums  = key_numbers(df)
    gap   = abs(nums["avg_dep"] - nums["avg_hap"])
    if gap > 0.5:
        return f"The emotional gap between the two communities is large ({gap:.3f}) — their language reflects very different realities."
    elif gap > 0.3:
        return f"The two communities differ meaningfully in tone ({gap:.3f} gap) despite sharing common vocabulary."
    else:
        return f"The communities are closer in tone than expected ({gap:.3f} gap) — shared human experience shows through."

def monthly_direction(df):
    """Is mood trending up or down over time?"""
    if df is None or "month" not in df.columns:
        return {}
    result = {}
    for sub in df["subreddit"].unique():
        monthly = (
            df[df["subreddit"]==sub]
            .groupby("month")["vader_compound"]
            .mean().sort_index()
        )
        if len(monthly) < 2:
            result[sub] = "not enough data"
            continue
        first_half = monthly.iloc[:len(monthly)//2].mean()
        second_half = monthly.iloc[len(monthly)//2:].mean()
        diff = second_half - first_half
        if diff > 0.02:
            result[sub] = f"trending positive (+{diff:.3f})"
        elif diff < -0.02:
            result[sub] = f"trending negative ({diff:.3f})"
        else:
            result[sub] = f"stable ({diff:+.3f})"
    return result