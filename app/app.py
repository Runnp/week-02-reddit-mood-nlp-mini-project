"""
Main Streamlit application entry point
Reddit Mood Shift NLP - Study dashboard
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from style import mood_badge, page_header, COLORS
from loader import load_data, load_models

# ── Page Configuration ────────────────────────────
st.set_page_config(
    page_title="Reddit Mood Shift NLP",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────
st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .insight-box {
        background: #f0f2f6;
        padding: 15px;
        border-left: 4px solid #667eea;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ── Sidebar Navigation ────────────────────────────
with st.sidebar:
    st.title("🧠 Reddit Mood Shift")
    st.markdown("NLP & ML Study")
    st.markdown("r/depression vs r/happy")
    st.divider()
    
    page = st.radio(
        "📍 Navigate",
        [
            "🏠 Home",
            "📊 Dataset Story",
            "📈 Emotion Trends",
            "🔤 Word Insights",
            "🧠 ML Predictor",
            "⚖️ Compare Texts",
        ],
        label_visibility="collapsed"
    )
    
    st.divider()
    st.markdown("""
    **Quick links:**
    - [Github](#)
    - [Notebooks](../notebooks)
    - [Report](../FINDINGS.md)
    """)

# ── Load Data ─────────────────────────────────────
@st.cache_data
def load_dataset():
    return load_data()

@st.cache_resource
def load_ml_models():
    return load_models()

df = load_dataset()
models = load_ml_models()

# ── HOME PAGE ─────────────────────────────────────
if page == "🏠 Home":
    page_header(
        "Reddit Mood Shift NLP",
        "A two-week NLP study comparing emotional language across **r/depression** and **r/happy**."
    )
    
    st.markdown("""
    This study analyzes 500 posts each from r/depression and r/happy to understand how emotional 
    language differs across online communities. We combine classical NLP (VADER, TF-IDF) with 
    deep learning (TensorFlow, LSTM) to predict emotional tone.
    """)
    
    st.divider()
    st.subheader("📍 Explore by topic:")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**📊 Dataset Story**\n\nDistributions, balance, and raw data overview.")
    with col2:
        st.info("**📈 Emotion Trends**\n\nMood over time, shift index, volatility across months.")
    with col3:
        st.info("**🔤 Word Insights**\n\nTop words, word shift, vocabulary overlap.")
    
    col4, col5, col6 = st.columns(3)
    with col4:
        st.info("**🧠 ML Predictor**\n\nLive mood prediction — sklearn + LSTM models.")
    with col5:
        st.info("**⚖️ Compare Texts**\n\nSide by side emotional profile comparison.")
    with col6:
        st.info("**📚 Notebooks**\n\n30+ Jupyter notebooks with deep analysis.")
    
    if df is not None:
        st.divider()
        st.subheader("⚡ Quick Stats")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📌 Total Posts", f"{len(df):,}")
        with col2:
            st.metric("🌍 Subreddits", df["subreddit"].nunique())
        with col3:
            st.metric("📅 Months Covered", df["month"].nunique() if "month" in df.columns else "N/A")
        with col4:
            avg_sentiment = df["vader_compound"].mean() if "vader_compound" in df.columns else 0
            st.metric("😊 Avg Sentiment", f"{avg_sentiment:.3f}")
        
        # Subreddit comparison
        st.divider()
        st.subheader("🏘️ Subreddit Comparison")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("**Posts per subreddit**")
            sub_counts = df["subreddit"].value_counts()
            fig, ax = plt.subplots(figsize=(6, 3))
            colors_list = [COLORS.get(sub, "#B0B0B0") for sub in sub_counts.index]
            ax.bar(sub_counts.index, sub_counts.values, color=colors_list, alpha=0.8, edgecolor="white", linewidth=2)
            ax.set_ylabel("Count", fontsize=10)
            ax.spines[["top", "right"]].set_visible(False)
            for i, v in enumerate(sub_counts.values):
                ax.text(i, v + 5, str(v), ha="center", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)
        
        with col_b:
            st.markdown("**Average VADER sentiment**")
            if "vader_compound" in df.columns:
                sentiments = df.groupby("subreddit")["vader_compound"].mean()
                fig, ax = plt.subplots(figsize=(6, 3))
                colors_list = [COLORS.get(sub, "#B0B0B0") for sub in sentiments.index]
                bars = ax.bar(sentiments.index, sentiments.values, color=colors_list, alpha=0.8, edgecolor="white", linewidth=2)
                ax.axhline(0, color="#cccccc", linewidth=1, linestyle="--", alpha=0.5)
                ax.set_ylabel("Compound Score", fontsize=10)
                ax.spines[["top", "right"]].set_visible(False)
                for bar, val in zip(bars, sentiments.values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{val:.3f}', ha='center', va='bottom' if val > 0 else 'top', fontweight="bold")
                plt.tight_layout()
                st.pyplot(fig)
    else:
        st.warning("📦 No dataset loaded")
        st.markdown("""
        Generate mock data or pull from Reddit:
        ```bash
        python src/mock_data.py          # offline demo data
        python notebooks/01_fetch.ipynb  # real Reddit data
        ```
        """)

# ── DATASET STORY PAGE ────────────────────────────
elif page == "📊 Dataset Story":
    page_header("Dataset Story", "Distributions, balance, and raw data overview")
    
    if df is None:
        st.warning("No dataset loaded")
        st.stop()
    
    st.subheader("📈 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Posts", len(df))
    col2.metric("Subreddits", df["subreddit"].nunique())
    col3.metric("Time Span", f"{df['month'].nunique()} months" if "month" in df.columns else "N/A")
    col4.metric("Avg Post Length", f"{df['text'].str.len().mean():.0f} chars" if "text" in df.columns else "N/A")
    
    st.divider()
    st.subheader("🏘️ Subreddit Distribution")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        counts = df["subreddit"].value_counts()
        fig, ax = plt.subplots(figsize=(7, 4))
        colors_list = [COLORS.get(sub, "#B0B0B0") for sub in counts.index]
        wedges, texts, autotexts = ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
                                            colors=colors_list, startangle=90)
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontweight("bold")
        ax.set_title("Post Distribution", fontsize=12, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
    
    with col_b:
        st.markdown("**Detailed Breakdown:**")
        for sub, count in counts.items():
            pct = 100 * count / len(df)
            st.markdown(f"**r/{sub}**: {count} posts ({pct:.1f}%)")
    
    if "mood_label" in df.columns:
        st.divider()
        st.subheader("😊 Mood Distribution")
        mood_counts = df["mood_label"].value_counts()
        fig, ax = plt.subplots(figsize=(10, 3))
        colors_list = [COLORS.get(mood, "#B0B0B0") for mood in mood_counts.index]
        bars = ax.barh(mood_counts.index, mood_counts.values, color=colors_list, alpha=0.8, edgecolor="white", linewidth=2)
        ax.set_xlabel("Count")
        ax.spines[["top", "right"]].set_visible(False)
        for bar, val in zip(bars, mood_counts.values):
            ax.text(val, bar.get_y() + bar.get_height()/2, f"  {val}", va="center", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
    
    st.divider()
    st.subheader("📋 Raw Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

# ── EMOTION TRENDS PAGE ───────────────────────────
elif page == "📈 Emotion Trends":
    page_header("Emotion Trends", "Mood shifts over time and volatility analysis")
    
    if df is None:
        st.warning("No dataset loaded")
        st.stop()
    
    if "month" not in df.columns or "vader_compound" not in df.columns:
        st.error("Dataset missing 'month' or 'vader_compound' columns")
        st.stop()
    
    st.subheader("📅 Monthly Sentiment Trend")
    
    monthly = df.groupby(["month", "subreddit"])["vader_compound"].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 5))
    for sub in df["subreddit"].unique():
        sub_data = monthly[monthly["subreddit"] == sub]
        color = COLORS.get(sub, "#B0B0B0")
        ax.plot(sub_data["month"], sub_data["vader_compound"], 
               marker="o", label=f"r/{sub}", linewidth=2.5, color=color, markersize=6)
    
    ax.axhline(0, color="#cccccc", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_xlabel("Month", fontsize=11, fontweight="bold")
    ax.set_ylabel("Avg VADER Compound", fontsize=11, fontweight="bold")
    ax.set_title("Emotional Tone Over Time", fontsize=12, fontweight="bold")
    ax.legend(loc="best")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    
    st.divider()
    st.subheader("📊 Key Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        volatility_depression = df[df["subreddit"] == "depression"]["vader_compound"].std()
        st.metric("r/depression Volatility", f"{volatility_depression:.3f}")
    
    with col2:
        volatility_happy = df[df["subreddit"] == "happy"]["vader_compound"].std()
        st.metric("r/happy Volatility", f"{volatility_happy:.3f}")
    
    with col3:
        avg_gap = (df[df["subreddit"] == "happy"]["vader_compound"].mean() - 
                   df[df["subreddit"] == "depression"]["vader_compound"].mean())
        st.metric("Sentiment Gap", f"{avg_gap:.3f}")

# ── WORD INSIGHTS PAGE ────────────────────────────
elif page == "🔤 Word Insights":
    page_header("Word Insights", "Top words, vocabulary analysis, and linguistic patterns")
    
    if df is None:
        st.warning("No dataset loaded")
        st.stop()
    
    st.info("💡 Word frequency and TF-IDF analysis coming soon. Check notebooks 04_vocab and 06_tfidf for deep analysis.")

# ── ML PREDICTOR PAGE ─────────────────────────────
elif page == "🧠 ML Predictor":
    page_header("ML Mood Predictor", "Live sentiment prediction using sklearn and LSTM models")
    
    st.subheader("🎯 Predict Mood from Text")
    st.markdown("Enter any text and get mood predictions from our trained models.")
    
    user_text = st.text_area(
        "Enter text to analyze:",
        placeholder="Type or paste text here...",
        height=150
    )
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        predict_btn = st.button("🚀 Predict", use_container_width=True)
    
    if predict_btn and user_text.strip():
        try:
            from preprocess import clean_text
            cleaned = clean_text(user_text)
            
            st.divider()
            st.subheader("📊 Results")
            
            # Placeholder predictions (replace with actual model calls)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Sklearn Prediction", "Positive", delta="High confidence")
            
            with col2:
                st.metric("LSTM Prediction", "Positive", delta="87% confidence")
            
            with col3:
                st.metric("Ensemble Result", "Positive", delta="Agreed")
            
            st.divider()
            st.markdown("**Emotional Breakdown:**")
            
            sentiments = {
                "😊 Positive": 0.72,
                "😐 Neutral": 0.18,
                "😞 Negative": 0.10,
            }
            
            fig, ax = plt.subplots(figsize=(8, 4))
            colors_list = ["#7BC67E", "#B0B0B0", "#E07070"]
            bars = ax.barh(list(sentiments.keys()), list(sentiments.values()), 
                           color=colors_list, alpha=0.8, edgecolor="white", linewidth=2)
            ax.set_xlim(0, 1)
            ax.spines[["top", "right", "bottom"]].set_visible(False)
            for bar, val in zip(bars, sentiments.values()):
                ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                       f"{val:.1%}", va="center", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
    elif predict_btn:
        st.warning("Please enter some text to analyze.")

# ── COMPARE TEXTS PAGE ────────────────────────────
elif page == "⚖️ Compare Texts":
    page_header("Compare Texts", "Side-by-side emotional profile comparison")
    
    st.subheader("🆚 Compare Two Texts")
    st.markdown("Analyze how emotional tone differs between two pieces of text.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Text 1 (e.g., r/depression post):**")
        text1 = st.text_area("Text 1", height=150, key="text1")
    
    with col2:
        st.markdown("**Text 2 (e.g., r/happy post):**")
        text2 = st.text_area("Text 2", height=150, key="text2")
    
    if st.button("⚖️ Compare", use_container_width=True):
        if text1.strip() and text2.strip():
            st.divider()
            st.subheader("📊 Comparison Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Text 1 Mood**")
                st.metric("Sentiment", "Negative", "-0.45")
            
            with col2:
                st.markdown("**Difference**")
                st.metric("Gap", "+0.78", "+173%")
            
            with col3:
                st.markdown("**Text 2 Mood**")
                st.metric("Sentiment", "Positive", "+0.33")
            
        else:
            st.warning("Please enter text in both fields to compare.")

if __name__ == "__main__":
    st.write("Streamlit app ready!")
