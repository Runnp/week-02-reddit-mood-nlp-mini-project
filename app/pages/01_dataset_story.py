import sys, os

# fix paths for both direct run and streamlit multipage
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
APP  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC  = os.path.abspath(os.path.join(ROOT, "src"))

for p in [ROOT, APP, SRC]:
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from app.loader import load_data, data_missing

st.set_page_config(page_title="Dataset Story", page_icon="📊", layout="wide")
st.title("📊 Dataset Story")
st.markdown("Understanding the raw data before any analysis.")

df = load_data()
data_missing(df)

# ── metrics ───────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total posts",    len(df))
col2.metric("Subreddits",     df["subreddit"].nunique())
col3.metric("Months covered", df["month"].nunique())
col4.metric("Avg score",      f"{df['score'].mean():.0f}")

st.divider()

# ── subreddit balance ─────────────────────
st.subheader("Subreddit balance")
counts = df["subreddit"].value_counts()
colors = ["#E07070" if s == "depression" else "#7BC67E" for s in counts.index]

fig, ax = plt.subplots(figsize=(6, 3))
bars = ax.bar(counts.index, counts.values, color=colors,
              edgecolor="white", linewidth=0.5)
ax.set_ylabel("posts")
ax.spines[["top","right"]].set_visible(False)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 3,
            str(val), ha="center", fontsize=10)
plt.tight_layout()
st.pyplot(fig)

st.divider()

# ── sentiment comparison ──────────────────
st.subheader("Sentiment comparison")
col_a, col_b = st.columns(2)

with col_a:
    avg_sent = df.groupby("subreddit")["vader_compound"].mean()
    fig, ax  = plt.subplots(figsize=(5, 3))
    colors   = ["#E07070" if s == "depression" else "#7BC67E"
                for s in avg_sent.index]
    bars = ax.bar(avg_sent.index, avg_sent.values,
                  color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="#cccccc", linewidth=0.8, linestyle="--")
    ax.set_title("Avg VADER compound score", fontsize=11)
    ax.set_ylabel("compound")
    ax.spines[["top","right"]].set_visible(False)
    for bar, val in zip(bars, avg_sent.values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f"{val:+.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)

with col_b:
    dist = df.groupby(["subreddit","mood_label"]).size().unstack(fill_value=0)
    dist_pct = dist.div(dist.sum(axis=1), axis=0).round(3)
    st.markdown("**Mood label breakdown:**")
    st.dataframe(dist_pct, use_container_width=True)

st.divider()

# ── post length ───────────────────────────
st.subheader("Post length distribution")
if "token_count" in df.columns:
    fig, ax = plt.subplots(figsize=(10, 4))
    for sub, color in zip(["depression","happy"], ["#E07070","#7BC67E"]):
        data = df[df["subreddit"]==sub]["token_count"]
        data = data[data < data.quantile(0.95)]
        ax.hist(data, bins=30, alpha=0.6, color=color,
                label=f"r/{sub}", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("token count")
    ax.set_ylabel("posts")
    ax.legend(frameon=False)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
else:
    st.info("No token_count column — run 02_clean notebook first.")

st.divider()

# ── monthly post count ────────────────────
st.subheader("Posts per month")
monthly_counts = df.groupby(["month","subreddit"]).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(13, 3))
for sub, color in zip(["depression","happy"], ["#E07070","#7BC67E"]):
    if sub in monthly_counts.columns:
        ax.plot(monthly_counts.index, monthly_counts[sub],
                label=f"r/{sub}", color=color,
                linewidth=2, marker="o", markersize=4)
ax.set_ylabel("posts")
ax.legend(frameon=False)
ax.spines[["top","right"]].set_visible(False)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
st.pyplot(fig)

st.divider()

# ── raw snapshot ──────────────────────────
st.subheader("Data snapshot")
st.dataframe(
    df[["subreddit","title","vader_compound",
        "mood_label","month","score"]].head(15),
    use_container_width=True
)
st.caption("Raw dataset — 15 rows. Run notebooks 01 and 02 for full data.")