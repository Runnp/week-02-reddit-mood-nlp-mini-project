import pandas as pd

def monthly_sentiment(df):
    return (
        df.groupby(["month", "subreddit"])["vader_compound"]
        .mean()
        .reset_index()
    )


def emotion_shift_index(df):
    """
    Measures how much emotional difference exists between subreddits over time.
    """
    pivot = df.pivot_table(
        index="month",
        columns="subreddit",
        values="vader_compound",
        aggfunc="mean"
    )

    if "depression" in pivot.columns and "happy" in pivot.columns:
        pivot["shift_index"] = pivot["happy"] - pivot["depression"]
    
    return pivot


def volatility(df):
    """
    How unstable mood is over time.
    """
    return df.groupby("subreddit")["vader_compound"].std()