import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np

COLORS = {
    "depression": "#E07070",
    "happy":      "#7BC67E",
    "positive":   "#7BC67E",
    "neutral":    "#B0B0B0",
    "negative":   "#E07070",
}

def style_ax(ax):
    """Remove top and right spines."""
    ax.spines[["top", "right"]].set_visible(False)
    return ax

def bar_chart(ax, labels, values, title, ylabel,
              colors=None, annotate=True):
    """Simple bar chart with optional value labels."""
    if colors is None:
        colors = [COLORS.get(l, "#B0B0B0") for l in labels]
    bars = ax.bar(labels, values, color=colors,
                  edgecolor="white", linewidth=0.5)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel(ylabel)
    style_ax(ax)
    if annotate:
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01 * max(values),
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    return ax

def line_chart(ax, x, y_dict, title, xlabel, ylabel,
               hline=None):
    """Multi-line chart. y_dict = {label: values}."""
    for label, values in y_dict.items():
        ax.plot(x, values, label=label,
                color=COLORS.get(label, "#555555"),
                linewidth=2, marker="o", markersize=4)
    if hline is not None:
        ax.axhline(hline, color="#cccccc",
                   linewidth=0.8, linestyle="--")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    style_ax(ax)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    return ax

def confusion_heatmap(ax, cm, classes, title):
    """Annotated confusion matrix heatmap."""
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes,
                linewidths=0.5, ax=ax, cbar=False)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    return ax

def save(filename, dpi=150):
    """Save current figure to outputs/."""
    import os
    os.makedirs("../outputs", exist_ok=True)
    path = f"../outputs/{filename}"
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved to outputs/{filename}")