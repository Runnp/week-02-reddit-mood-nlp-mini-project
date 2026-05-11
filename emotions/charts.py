import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from emotions.config import EMOTION_COLORS

def sadness_chart(analysis, ax=None):
    """
    Visualize sadness intensity — wave-style fill chart
    across sentences, colored by negative sentiment.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    sentences = analysis["sentences"]
    if not sentences:
        ax.text(0.5, 0.5, "No sentences found",
                ha="center", va="center", transform=ax.transAxes)
        return ax

    x    = np.arange(len(sentences))
    neg  = [-s["neg"] for s in sentences]   # negative so it fills downward

    ax.fill_between(x, neg, 0, alpha=0.4,
                    color=EMOTION_COLORS["sadness"], linewidth=0)
    ax.plot(x, neg, color=EMOTION_COLORS["sadness"],
            linewidth=2, marker="o", markersize=5)
    ax.axhline(0, color="#cccccc", linewidth=0.8, linestyle="--")

    ax.set_title("Sadness arc — sentence by sentence", fontsize=13, pad=10)
    ax.set_xlabel("sentence")
    ax.set_ylabel("negativity score")
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{i+1}" for i in x], fontsize=9)
    ax.set_ylim(-1.05, 0.1)
    ax.spines[["top","right"]].set_visible(False)

    # annotate most negative sentence
    if neg:
        worst_idx = int(np.argmin(neg))
        ax.annotate(
            f"most negative\n\"{sentences[worst_idx]['sentence'][:40]}...\"",
            xy=(worst_idx, neg[worst_idx]),
            xytext=(worst_idx + 0.3, neg[worst_idx] - 0.1),
            fontsize=8, color=EMOTION_COLORS["sadness"],
            arrowprops=dict(arrowstyle="->",
                            color=EMOTION_COLORS["sadness"], lw=1),
        )
    return ax

def happiness_chart(analysis, ax=None):
    """
    Visualize happiness intensity — upward wave fill chart
    across sentences, colored by positive sentiment.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    sentences = analysis["sentences"]
    if not sentences:
        ax.text(0.5, 0.5, "No sentences found",
                ha="center", va="center", transform=ax.transAxes)
        return ax

    x   = np.arange(len(sentences))
    pos = [s["pos"] for s in sentences]

    ax.fill_between(x, pos, 0, alpha=0.4,
                    color=EMOTION_COLORS["happiness"], linewidth=0)
    ax.plot(x, pos, color=EMOTION_COLORS["happiness"],
            linewidth=2, marker="o", markersize=5)
    ax.axhline(0, color="#cccccc", linewidth=0.8, linestyle="--")

    ax.set_title("Happiness arc — sentence by sentence", fontsize=13, pad=10)
    ax.set_xlabel("sentence")
    ax.set_ylabel("positivity score")
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{i+1}" for i in x], fontsize=9)
    ax.set_ylim(-0.1, 1.05)
    ax.spines[["top","right"]].set_visible(False)

    # annotate happiest sentence
    if pos and max(pos) > 0:
        best_idx = int(np.argmax(pos))
        ax.annotate(
            f"most positive\n\"{sentences[best_idx]['sentence'][:40]}...\"",
            xy=(best_idx, pos[best_idx]),
            xytext=(best_idx + 0.3, pos[best_idx] + 0.05),
            fontsize=8, color=EMOTION_COLORS["happiness"],
            arrowprops=dict(arrowstyle="->",
                            color=EMOTION_COLORS["happiness"], lw=1),
        )
    return ax

def emotion_radar(analysis, ax=None):
    """Radar chart of all five emotion scores."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6),
                                subplot_kw=dict(polar=True))

    emotions = analysis["emotions"]
    labels   = list(emotions.keys())
    values   = list(emotions.values())
    colors   = [EMOTION_COLORS[e] for e in labels]

    N      = len(labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    values += values[:1]

    ax.plot(angles, values, "o-", linewidth=2,
            color=EMOTION_COLORS.get(analysis["dominant"],"#555555"))
    ax.fill(angles, values, alpha=0.2,
            color=EMOTION_COLORS.get(analysis["dominant"],"#555555"))
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks([0.01, 0.02, 0.03])
    ax.set_yticklabels(["low","mid","high"], fontsize=7, color="#aaaaaa")
    ax.set_title("Emotion profile", fontsize=12, pad=15)
    return ax

def compound_arc(analysis, ax=None):
    """Compound sentiment arc across all sentences."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 3))

    sentences = analysis["sentences"]
    x         = np.arange(len(sentences))
    compound  = [s["compound"] for s in sentences]
    colors    = ["#7BC67E" if c >= 0.05
                 else "#E07070" if c <= -0.05
                 else "#B0B0B0" for c in compound]

    ax.bar(x, compound, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="#cccccc", linewidth=0.8, linestyle="--")
    ax.set_title("Overall sentiment arc — positive / neutral / negative",
                 fontsize=13, pad=10)
    ax.set_xlabel("sentence")
    ax.set_ylabel("compound score")
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{i+1}" for i in x], fontsize=9)
    ax.spines[["top","right"]].set_visible(False)
    return ax

def emotion_bars(analysis, ax=None):
    """Horizontal bar chart of emotion scores."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    emotions = analysis["emotions"]
    labels   = list(emotions.keys())
    values   = list(emotions.values())
    colors   = [EMOTION_COLORS[e] for e in labels]

    bars = ax.barh(labels[::-1], values[::-1],
                   color=colors[::-1], edgecolor="white", linewidth=0.5)
    ax.set_title("Emotion scores", fontsize=12)
    ax.set_xlabel("score")
    ax.spines[["top","right"]].set_visible(False)

    dominant = analysis["dominant"]
    for bar, label in zip(bars, labels[::-1]):
        if label == dominant:
            ax.text(bar.get_width() + 0.0005,
                    bar.get_y() + bar.get_height()/2,
                    "← dominant", va="center", fontsize=9,
                    color=EMOTION_COLORS[label])
    return ax

def full_dashboard(analysis, save_path=None):
    """Generate the full emotion dashboard — all charts in one figure."""
    fig = plt.figure(figsize=(16, 14))
    gs  = gridspec.GridSpec(3, 2, figure=fig,
                             hspace=0.5, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :])   # compound arc — full width
    ax2 = fig.add_subplot(gs[1, 0])   # sadness
    ax3 = fig.add_subplot(gs[1, 1])   # happiness
    ax4 = fig.add_subplot(gs[2, 0], polar=True)  # radar
    ax5 = fig.add_subplot(gs[2, 1])   # emotion bars

    compound_arc(analysis, ax=ax1)
    sadness_chart(analysis, ax=ax2)
    happiness_chart(analysis, ax=ax3)
    emotion_radar(analysis, ax=ax4)
    emotion_bars(analysis, ax=ax5)

    mood_color = {
        "positive": "#7BC67E",
        "neutral":  "#B0B0B0",
        "negative": "#E07070",
    }.get(analysis["overall_mood"], "#555555")

    fig.suptitle(
        f"Emotion Analysis Dashboard  —  "
        f"overall: {analysis['overall_mood'].upper()}  |  "
        f"dominant: {analysis['dominant'].upper()}  |  "
        f"{analysis['word_count']} words",
        fontsize=14, fontweight="500", color=mood_color, y=1.01
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    return fig