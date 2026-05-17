import sys, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
APP  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC  = os.path.abspath(os.path.join(ROOT, "src"))
for p in [ROOT, APP, SRC]:
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st
import pandas as pd
import os
from app.loader import load_data
from app.style import page_header

st.set_page_config(page_title="Export", page_icon="📦", layout="wide")
page_header("📦 Export", "Download data and charts from the project.")

df = load_data()

# ── dataset exports ───────────────────────
st.subheader("Dataset")
if df is not None:
    col1, col2, col3 = st.columns(3)

    with col1:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Full dataset CSV",
                           csv, "posts_sentiment.csv",
                           "text/csv")

    with col2:
        dep = df[df["subreddit"]=="depression"].to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ r/depression only",
                           dep, "depression_posts.csv",
                           "text/csv")

    with col3:
        hap = df[df["subreddit"]=="happy"].to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ r/happy only",
                           hap, "happy_posts.csv",
                           "text/csv")

    st.divider()

    # filtered export
    st.subheader("Filtered export")
    col4, col5 = st.columns(2)
    with col4:
        mood_filter = st.multiselect("Filter by mood",
                                      ["positive","neutral","negative"],
                                      default=["negative"])
    with col5:
        sub_filter = st.multiselect("Filter by subreddit",
                                     ["depression","happy"],
                                     default=["depression"])

    filtered = df[
        (df["mood_label"].isin(mood_filter)) &
        (df["subreddit"].isin(sub_filter))
    ]
    st.metric("Matching posts", len(filtered))
    if len(filtered) > 0:
        st.download_button("⬇️ Download filtered CSV",
                           filtered.to_csv(index=False).encode("utf-8"),
                           "filtered_posts.csv", "text/csv")
        st.dataframe(filtered[["subreddit","title","mood_label",
                                "vader_compound","month"]].head(10),
                     use_container_width=True)
else:
    st.warning("No data found.")
    st.code("python src/mock_data.py")

st.divider()

# ── chart exports ─────────────────────────
st.subheader("Saved charts")
if os.path.exists("outputs"):
    charts = sorted(f for f in os.listdir("outputs") if f.endswith(".png"))
    if charts:
        st.markdown(f"{len(charts)} charts available")
        cols = st.columns(3)
        for i, chart in enumerate(charts):
            with cols[i % 3]:
                with open(os.path.join("outputs", chart), "rb") as f:
                    st.download_button(f"⬇️ {chart}",
                                       f.read(), chart, "image/png",
                                       key=chart)
    else:
        st.info("No charts yet — run the notebooks first.")
else:
    st.info("outputs/ folder not found.")