import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from app.loader import load_data, data_missing

st.set_page_config(page_title="Emotion Trends", page_icon="📈", layout="wide")
st.title("📈 Emotion Trends")
st.markdown("How emotional tone shifts across both communities over time.")

df = load_data()
data_missing(df)

# ── helper functions ──────────────────────
def monthly_sentiment(df):
    return (
        df.groupby(["month","subreddit"])["vader_compound"]
        .mean().reset_index()
    )

def emotion_shift_index(df):
    pivot = df.pivot_table(
        index="month", columns="subreddit",
        values="vader_compound", aggfunc="mean"
    )
    if "depression" in pivot.columns and "happy" in pivot.columns:
        pivot["shift_index"] = pivot["happy"] - pivot["depression"]
    return pivot

def volatility(df):
    return df.groupby("subreddit")["vader_compound"].std().round(4)

# ── filters ───────────────────────────────
months = sorted(df["month"].dropna().unique())
col1, col2 = st.columns(2)
with col1:
    selected_subs = st.multiselect(
        "Subreddits", ["depression","happy"],
        default=["depression","happy"]
    )
with col2:
    month_range = st.select_slider(
        "Month range", options=months,
        value=(months[0], months[-1])
    )

filtered = df[
    (df["subreddit"].isin(selected_subs)) &
    (df["month"] >= month_range[0]) &
    (df["month"] <= month_range[1])
]

st.divider()

# ── mood over time ────────────────────────
st.subheader("Mood over time")
monthly = monthly_sentiment(filtered)
colors  = {"depression":"#E07070","happy":"#7BC67E"}

fig, ax = plt.subplots(figsize=(13, 4))
for sub, group in monthly.groupby("subreddit"):
    ax.plot(group["month"], group["vader_compound"],
            label=f"r/{sub}", color=colors.get(sub,"#555555"),
            linewidth=2, marker="o", markersize=4)
ax.axhline(0, color="#cccccc", linewidth=0.8, linestyle="--")
ax.set_ylabel("avg compound score")
ax.legend(frameon=False)
ax.spines[["top","right"]].set_visible(False)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
st.pyplot(fig)

st.divider()

# ── shift index ───────────────────────────
st.subheader("Emotional shift index")
st.markdown("How far apart the two communities are emotionally each month — higher = bigger gap.")

pivot = emotion_shift_index(filtered)

if "shift_index" in pivot.columns:
    fig, ax = plt.subplots(figsize=(13, 3))
    colors_bar = ["#7BC67E" if v >= 0 else "#E07070"
                  for v in pivot["shift_index"]]
    ax.bar(pivot.index, pivot["shift_index"],
           color=colors_bar, edgecolor="white", linewidth=0.3)
    ax.axhline(0, color="#cccccc", linewidth=0.8, linestyle="--")
    ax.set_ylabel("happy − depression")
    ax.spines[["top","right"]].set_visible(False)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown(f"**Avg shift index:** `{pivot['shift_index'].mean():.3f}`")
else:
    st.info("Need both subreddits selected to show shift index.")

st.divider()

# ── volatility ────────────────────────────
st.subheader("Mood volatility")
st.markdown("Standard deviation of sentiment — higher = more unstable mood over time.")

vol = volatility(filtered)
col_a, col_b = st.columns(2)
for col, sub in zip([col_a, col_b], selected_subs):
    if sub in vol.index:
        col.metric(f"r/{sub} volatility", f"{vol[sub]:.4f}")

st.divider()

# ── monthly table ─────────────────────────
st.subheader("Monthly breakdown table")
if not pivot.empty:
    st.dataframe(pivot.round(3), use_container_width=True)

st.divider()

# ── best and worst months ─────────────────
st.subheader("Best and worst months")
for sub in selected_subs:
    sub_monthly = monthly[monthly["subreddit"]==sub]
    if len(sub_monthly) == 0:
        continue
    best  = sub_monthly.loc[sub_monthly["vader_compound"].idxmax()]
    worst = sub_monthly.loc[sub_monthly["vader_compound"].idxmin()]
    col_x, col_y = st.columns(2)
    col_x.metric(f"r/{sub} — best month",
                  best["month"], f"{best['vader_compound']:+.3f}")
    col_y.metric(f"r/{sub} — worst month",
                  worst["month"], f"{worst['vader_compound']:+.3f}")