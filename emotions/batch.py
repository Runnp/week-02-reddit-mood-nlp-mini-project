import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from emotion_tool.analyzer import analyze, score_emotions, score_sentence_sentiment
from emotion_tool.config import EMOTIONS, EMOTION_COLORS

def score_emotions_fast(text):
    """Faster version — no full analysis, just emotion scores."""
    import re
    from nltk.tokenize import word_tokenize
    if not isinstance(text, str):
        return {e: 0 for e in EMOTIONS}
    text  = text.lower()
    text  = re.sub(r"[^a-z\s]", "", text)
    words = set(text.split())
    scores = {}
    for emotion, keywords in EMOTIONS.items():
        hits = len(words & set(keywords))
        scores[emotion] = round(hits / max(len(words), 1), 4)
    return scores

def analyze_dataframe(df, text_col="clean_text", sample=None):
    """Add emotion score columns to a DataFrame."""
    if sample:
        df = df.sample(min(sample, len(df)), random_state=42).copy()
    else:
        df = df.copy()

    print(f"Scoring {len(df)} posts...")
    emotion_rows = df[text_col].apply(score_emotions_fast)
    emotion_df   = pd.DataFrame(emotion_rows.tolist(), index=df.index)

    df = pd.concat([df, emotion_df], axis=1)
    print("Done.")
    return df

def emotion_summary(df):
    """Print avg emotion scores per subreddit."""
    emotions = list(EMOTIONS.keys())
    summary  = df.groupby("subreddit")[emotions].mean().round(4)
    print("\nAvg emotion scores per subreddit:")
    print(summary.T.to_string())
    return summary

def plot_emotion_comparison(df, save_path=None):
    """Bar chart comparing all emotions across subreddits."""
    emotions = list(EMOTIONS.keys())
    summary  = df.groupby("subreddit")[emotions].mean()

    x     = np.arange(len(emotions))
    width = 0.35
    subs  = summary.index.tolist()

    sub_colors = {
        "depression": "#E07070",
        "happy":      "#7BC67E",
    }

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, sub in enumerate(subs):
        offset = (i - len(subs)/2 + 0.5) * width
        ax.bar(x + offset, summary.loc[sub],
               width, label=f"r/{sub}",
               color=sub_colors.get(sub,"#B0B0B0"),
               edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(emotions, fontsize=11)
    ax.set_ylabel("avg emotion score")
    ax.set_title("Emotion scores — r/depression vs r/happy",
                 fontsize=13, pad=12)
    ax.legend(frameon=False)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    return fig

def plot_emotion_over_time(df, emotion, save_path=None):
    """Line chart of one emotion over time per subreddit."""
    monthly = (
        df.groupby(["subreddit","month"])[emotion]
        .mean().reset_index().sort_values("month")
    )
    sub_colors = {"depression":"#E07070","happy":"#7BC67E"}
    fig, ax = plt.subplots(figsize=(13, 4))
    for sub, group in monthly.groupby("subreddit"):
        ax.plot(group["month"], group[emotion],
                label=f"r/{sub}",
                color=sub_colors.get(sub,"#555555"),
                linewidth=2, marker="o", markersize=4)
    ax.set_title(f"{emotion.capitalize()} intensity over time",
                 fontsize=13, pad=10,
                 color=EMOTION_COLORS.get(emotion,"#555555"))
    ax.set_xlabel("month")
    ax.set_ylabel(f"avg {emotion} score")
    ax.legend(frameon=False)
    ax.spines[["top","right"]].set_visible(False)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    return fig