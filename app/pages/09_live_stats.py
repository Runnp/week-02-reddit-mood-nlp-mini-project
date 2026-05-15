import sys, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
APP  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC  = os.path.abspath(os.path.join(ROOT, "src"))
for p in [ROOT, APP, SRC]:
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from app.loader import load_data
from app.style import page_header, COLORS

st.set_page_config(page_title="Live Stats", page_icon="⚡", layout="wide")
page_header("⚡ Live Stats", "Real-time dataset and project health.")

# ── auto refresh ──────────────────────────
refresh = st.toggle("Auto refresh every 10s", value=False)
if refresh:
    import time
    st.caption(f"Last refreshed: {pd.Timestamp.now().strftime('%H:%M:%S')}")
    time.sleep(10)
    st.rerun()

if st.button("🔄 Refresh now"):
    st.cache_data.clear()
    st.rerun()

st.divider()

# ── file health ───────────────────────────
st.subheader("File health")
files = {
    "data/raw/posts_raw.csv":            "Raw posts",
    "data/clean/posts_clean.csv":        "Clean posts",
    "data/clean/posts_sentiment.csv":    "Sentiment data",
    "data/clean/train_test_split.pkl":   "Train/test split",
    "data/clean/best_sklearn_model.pkl": "sklearn model",
    "data/clean/tf_mood_model":          "TF model",
    "data/clean/lstm_mood_model":        "LSTM model",
}

cols    = st.columns(4)
ok, mis = 0, 0
for i, (path, label) in enumerate(files.items()):
    exists = os.path.exists(path)
    col    = cols[i % 4]
    if exists:
        col.success(f"✅ {label}")
        ok += 1
    else:
        col.error(f"❌ {label}")
        mis += 1

st.caption(f"{ok} files present — {mis} missing")

st.divider()

# ── dataset stats ─────────────────────────
df = load_data()
if df is not None:
    st.subheader("Dataset snapshot")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total posts",    len(df))
    col2.metric("Subreddits",     df["subreddit"].nunique())
    col3.metric("Months",         df["month"].nunique())
    col4.metric("Avg score",      f"{df['score'].mean():.0f}")
    col5.metric("Avg comments",   f"{df['num_comments'].mean():.0f}")

    st.divider()

    # mood breakdown
    st.subheader("Current mood breakdown")
    if "mood_label" in df.columns:
        col_a, col_b = st.columns(2)
        for col, sub in zip([col_a, col_b], ["depression","happy"]):
            with col:
                dist = df[df["subreddit"]==sub]["mood_label"].value_counts()
                fig, ax = plt.subplots(figsize=(5, 3))
                mood_colors = [COLORS.get(m,"#B0B0B0") for m in dist.index]
                ax.bar(dist.index, dist.values,
                       color=mood_colors, edgecolor="white", linewidth=0.5)
                ax.set_title(f"r/{sub}", fontsize=12)
                ax.set_ylabel("posts")
                ax.spines[["top","right"]].set_visible(False)
                plt.tight_layout()
                col.pyplot(fig)

    st.divider()

    # outputs
    st.subheader("Generated outputs")
    if os.path.exists("outputs"):
        charts = sorted(f for f in os.listdir("outputs") if f.endswith(".png"))
        csvs   = sorted(f for f in os.listdir("outputs") if f.endswith(".csv"))
        col_x, col_y = st.columns(2)
        col_x.metric("Charts saved", len(charts))
        col_y.metric("CSVs saved",   len(csvs))
        if charts:
            st.markdown("**Latest charts:**")
            for c in charts[-6:]:
                st.markdown(f"- `{c}`")
    else:
        st.info("No outputs folder yet.")
else:
    st.warning("No dataset loaded.")
    st.code("python src/mock_data.py")