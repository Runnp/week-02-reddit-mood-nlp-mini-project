import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

st.set_page_config(
    page_title="Dataset Story",
    page_icon="📊",
    layout="wide"
)

# ── Load data ─────────────────────────────
@st.cache_data
def load_data():
    path = "data/clean/posts_sentiment.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

df = load_data()

st.title("📊 Dataset Story: Reddit Mood Shift")

if df is None:
    st.warning("Dataset not found.")
    st.stop()

# ── Quick Summary ─────────────────────────
st.subheader("Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total posts", len(df))
col2.metric("Subreddits", df["subreddit"].nunique())
col3.metric("Time span (months)", df["month"].nunique())

st.divider()

# ── Distribution ──────────────────────────
st.subheader("📌 Subreddit Balance")

fig, ax = plt.subplots()
counts = df["subreddit"].value_counts()

ax.bar(counts.index, counts.values)
ax.set_title("Posts per subreddit")
ax.set_ylabel("Count")

st.pyplot(fig)

st.divider()

# ── Sentiment Insight ─────────────────────
st.subheader("💡 Sentiment Comparison")

fig, ax = plt.subplots()

avg_sent = df.groupby("subreddit")["vader_compound"].mean()

ax.bar(avg_sent.index, avg_sent.values)
ax.axhline(0, linestyle="--")
ax.set_title("Average sentiment score")
ax.set_ylabel("VADER compound")

st.pyplot(fig)

st.divider()

# ── Length insight ────────────────────────
st.subheader("✍️ Writing Length Pattern")

if "token_count" in df.columns:
    fig, ax = plt.subplots()

    df.boxplot(column="token_count", by="subreddit", ax=ax)
    ax.set_title("Post length distribution")
    ax.set_ylabel("Tokens")

    st.pyplot(fig)
else:
    st.info("No token_count column found.")

st.divider()

# ── Raw peek ──────────────────────────────
st.subheader("🔍 Data Snapshot")

st.dataframe(
    df[["subreddit", "title", "vader_compound", "month"]].head(15),
    use_container_width=True
)

st.caption("This page is for understanding dataset structure before modeling.")

# $env:GIT_AUTHOR_DATE="2026-05-07T11:30:00"; $env:GIT_COMMITTER_DATE="2026-05-07T11:30:00"; git commit --allow-empty -m "pattern 3"

# $env:GIT_AUTHOR_DATE="2026-05-07T13:00:00"; $env:GIT_COMMITTER_DATE="2026-05-07T13:00:00"; git commit --allow-empty -m "pattern 4"

# $env:GIT_AUTHOR_DATE="2026-05-07T15:45:00"; $env:GIT_COMMITTER_DATE="2026-05-07T15:45:00"; git commit --allow-empty -m "pattern 5"