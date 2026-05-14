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
import numpy as np
from emotions.analyzer import analyze
from emotions.config import EMOTION_COLORS

st.set_page_config(page_title="Compare Texts", page_icon="⚖️", layout="wide")
st.title("⚖️ Compare Two Texts")
st.markdown("Paste two texts side by side and compare their emotional profiles.")

# ── presets ───────────────────────────────
PRESETS = {
    "Depression vs Happy post": (
        "I can't get out of bed anymore. Everything feels pointless and I don't know how much longer I can do this. I am so tired of pretending to be okay.",
        "Got the job I have been working towards for two years. I actually cried. My family was so proud and I feel like I can finally breathe again.",
    ),
    "Before vs After recovery": (
        "I feel completely empty and alone. Nothing brings me joy. I have lost interest in everything I used to love.",
        "Six months into therapy and I finally feel like myself again. I still have hard days but I have hope now.",
    ),
    "Custom": ("", ""),
}

preset = st.selectbox("Load a preset", list(PRESETS.keys()))
default_a, default_b = PRESETS[preset]

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Text A**")
    text_a = st.text_area("Text A", value=default_a, height=150,
                           label_visibility="collapsed")
with col2:
    st.markdown("**Text B**")
    text_b = st.text_area("Text B", value=default_b, height=150,
                           label_visibility="collapsed")

if st.button("Compare", type="primary") and text_a.strip() and text_b.strip():
    result_a = analyze(text_a)
    result_b = analyze(text_b)

    st.divider()

    # ── summary metrics ───────────────────
    st.subheader("Summary")
    col_a, col_b = st.columns(2)
    mood_emoji = {"positive":"🟢","neutral":"⚪","negative":"🔴"}

    with col_a:
        st.markdown("**Text A**")
        st.metric("Overall mood",     f"{mood_emoji.get(result_a['overall_mood'],'')} {result_a['overall_mood'].upper()}")
        st.metric("Dominant emotion", result_a["dominant"].upper())
        st.metric("Compound score",   f"{result_a['vader']['compound']:+.3f}")
        st.metric("Word count",       result_a["word_count"])

    with col_b:
        st.markdown("**Text B**")
        st.metric("Overall mood",     f"{mood_emoji.get(result_b['overall_mood'],'')} {result_b['overall_mood'].upper()}")
        st.metric("Dominant emotion", result_b["dominant"].upper())
        st.metric("Compound score",   f"{result_b['vader']['compound']:+.3f}")
        st.metric("Word count",       result_b["word_count"])

    st.divider()

    # ── emotion score comparison ──────────
    st.subheader("Emotion scores side by side")
    emotions  = list(result_a["emotions"].keys())
    scores_a  = list(result_a["emotions"].values())
    scores_b  = list(result_b["emotions"].values())
    x         = np.arange(len(emotions))
    width     = 0.35

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(x - width/2, scores_a, width, label="Text A",
           color="#6B9FD4", edgecolor="white", linewidth=0.5)
    ax.bar(x + width/2, scores_b, width, label="Text B",
           color="#7BC67E", edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(emotions, fontsize=11)
    ax.set_ylabel("emotion score")
    ax.set_title("Emotion comparison", fontsize=13)
    ax.legend(frameon=False)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

    st.divider()

    # ── sentiment arc overlay ─────────────
    st.subheader("Sentiment arc overlay")
    sents_a = result_a["sentences"]
    sents_b = result_b["sentences"]
    max_len = max(len(sents_a), len(sents_b))

    comp_a = [s["compound"] for s in sents_a]
    comp_b = [s["compound"] for s in sents_b]

    # pad shorter one with zeros
    comp_a += [0] * (max_len - len(comp_a))
    comp_b += [0] * (max_len - len(comp_b))

    fig, ax = plt.subplots(figsize=(13, 4))
    x_line  = np.arange(max_len)
    ax.plot(x_line, comp_a, label="Text A", color="#6B9FD4",
            linewidth=2, marker="o", markersize=5)
    ax.plot(x_line, comp_b, label="Text B", color="#7BC67E",
            linewidth=2, marker="s", markersize=5)
    ax.axhline(0, color="#cccccc", linewidth=0.8, linestyle="--")
    ax.fill_between(x_line, comp_a, comp_b, alpha=0.1, color="#999999")
    ax.set_title("Compound sentiment arc — Text A vs Text B", fontsize=13)
    ax.set_xlabel("sentence")
    ax.set_ylabel("compound score")
    ax.set_xticks(x_line)
    ax.set_xticklabels([f"S{i+1}" for i in x_line], fontsize=9)
    ax.legend(frameon=False)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

    st.divider()

    # ── radar overlay ─────────────────────
    st.subheader("Emotion radar overlay")
    labels = emotions
    vals_a = list(result_a["emotions"].values()) + [list(result_a["emotions"].values())[0]]
    vals_b = list(result_b["emotions"].values()) + [list(result_b["emotions"].values())[0]]
    N      = len(labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, vals_a, "o-", linewidth=2,
            color="#6B9FD4", label="Text A")
    ax.fill(angles, vals_a, alpha=0.15, color="#6B9FD4")
    ax.plot(angles, vals_b, "s-", linewidth=2,
            color="#7BC67E", label="Text B")
    ax.fill(angles, vals_b, alpha=0.15, color="#7BC67E")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title("Emotion radar overlay", fontsize=12, pad=15)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), frameon=False)
    plt.tight_layout()
    st.pyplot(fig)

    st.divider()

    # ── sentence tables ───────────────────
    st.subheader("Sentence breakdown")
    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("**Text A sentences:**")
        df_a = pd.DataFrame(result_a["sentences"])
        df_a["sentence"] = df_a["sentence"].str[:70]
        st.dataframe(df_a.round(3), use_container_width=True)
    with col_d:
        st.markdown("**Text B sentences:**")
        df_b = pd.DataFrame(result_b["sentences"])
        df_b["sentence"] = df_b["sentence"].str[:70]
        st.dataframe(df_b.round(3), use_container_width=True)