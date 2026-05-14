"""
Enhanced visualization utilities for Streamlit dashboard
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {
    "depression": "#E07070",
    "happy": "#7BC67E",
    "positive": "#7BC67E",
    "neutral": "#B0B0B0",
    "negative": "#E07070",
}

def plot_sentiment_bars(df, subreddit_col="subreddit", sentiment_col="vader_compound"):
    """Create sentiment comparison bar chart."""
    sentiments = df.groupby(subreddit_col)[sentiment_col].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    colors_list = [COLORS.get(sub, "#B0B0B0") for sub in sentiments.index]
    bars = ax.bar(sentiments.index, sentiments.values, color=colors_list, alpha=0.8, edgecolor="white", linewidth=2)
    ax.axhline(0, color="#cccccc", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_ylabel("Avg VADER Compound", fontweight="bold")
    ax.set_title("Sentiment Comparison", fontweight="bold", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars, sentiments.values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.3f}', ha='center', va='bottom' if val > 0 else 'top', fontweight="bold", fontsize=10)
    plt.tight_layout()
    return fig

def plot_monthly_trend(df, subreddit_col="subreddit", month_col="month", sentiment_col="vader_compound"):
    """Create time-series trend plot."""
    fig, ax = plt.subplots(figsize=(14, 5))
    
    for sub in df[subreddit_col].unique():
        sub_data = df[df[subreddit_col] == sub].groupby(month_col)[sentiment_col].mean()
        color = COLORS.get(sub, "#B0B0B0")
        ax.plot(sub_data.index, sub_data.values, marker="o", label=f"r/{sub}", 
               linewidth=2.5, color=color, markersize=6, alpha=0.8)
    
    ax.axhline(0, color="#cccccc", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_xlabel("Month", fontweight="bold")
    ax.set_ylabel("Avg VADER Compound", fontweight="bold")
    ax.set_title("Emotional Tone Over Time", fontweight="bold", fontsize=12)
    ax.legend(loc="best", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig

def plot_distribution(df, col, colors_dict=None, title="Distribution"):
    """Create distribution pie chart."""
    counts = df[col].value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    
    if colors_dict:
        colors_list = [colors_dict.get(item, "#B0B0B0") for item in counts.index]
    else:
        colors_list = plt.cm.Set3(np.linspace(0, 1, len(counts)))
    
    wedges, texts, autotexts = ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
                                       colors=colors_list, startangle=90)
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")
        autotext.set_fontsize(10)
    
    ax.set_title(title, fontweight="bold", fontsize=12)
    plt.tight_layout()
    return fig

def plot_correlation_heatmap(df, cols, title="Correlation"):
    """Create correlation heatmap."""
    import seaborn as sns
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
               square=True, ax=ax, cbar_kws={"label": "Correlation"})
    ax.set_title(title, fontweight="bold", fontsize=12)
    plt.tight_layout()
    return fig

def plot_word_frequency(word_freq, top_n=15, title="Top Words"):
    """Create bar chart of word frequency."""
    top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
    words, counts = zip(*top_words)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(words, counts, color="#667eea", alpha=0.8, edgecolor="white", linewidth=1)
    ax.set_xlabel("Frequency", fontweight="bold")
    ax.set_title(title, fontweight="bold", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.invert_yaxis()
    
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(count + 0.5, bar.get_y() + bar.get_height()/2, 
               str(count), va="center", fontweight="bold", fontsize=9)
    
    plt.tight_layout()
    return fig

def plot_emotion_radar(emotions_dict, title="Emotional Profile"):
    """Create radar chart for emotions."""
    import matplotlib.patches as mpatches
    
    categories = list(emotions_dict.keys())
    values = list(emotions_dict.values())
    values += values[:1]  # Complete the circle
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    ax.plot(angles, values, 'o-', linewidth=2, color='#667eea')
    ax.fill(angles, values, alpha=0.25, color='#667eea')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, max(values) * 1.1)
    ax.set_title(title, fontweight="bold", fontsize=12, pad=20)
    ax.grid(True, linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    return fig
