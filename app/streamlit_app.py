import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocess import clean_text, tokenize

st.set_page_config(
    page_title="Reddit Mood Shift",
    page_icon="🧠",
    layout="wide"
)

# ── Sidebar ───────────────────────────────────────
st.sidebar.title("🧠 Reddit Mood Shift")
st.sidebar.markdown("NLP study — r/depression vs r/happy")
page = st.sidebar.radio("Navigate", [
    "Overview",
    "Mood over time",
    "Word explorer",
    "Live predictor",
])

# ── Load data ─────────────────────────────────────
@st.cache_data
def load_data():
    path = "data/clean/posts_sentiment.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_resource
def load_models():
    try:
        with open("data/clean/best_sklearn_model.pkl", "rb") as f:
            sk = pickle.load(f)
        with open("data/clean/tf_tokenizer.pkl", "rb") as f:
            tf_data = pickle.load(f)
        lstm = tf.keras.models.load_model("data/clean/lstm_mood_model")
        return sk, tf_data, lstm
    except Exception:
        return None, None, None

df      = load_data()
sk, tf_data, lstm = load_models()

# ── Pages ─────────────────────────────────────────
if page == "Overview":
    st.title("Reddit Mood Shift NLP")
    st.markdown("Comparing emotional language patterns across two communities.")

    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total posts",       len(df))
        col2.metric("r/depression",       len(df[df["subreddit"]=="depression"]))
        col3.metric("r/happy",            len(df[df["subreddit"]=="happy"]))
        col4.metric("Months covered",     df["month"].nunique())

        st.subheader("Mood distribution")
        dist = df.groupby(["subreddit","mood_label"]).size().unstack(fill_value=0)
        st.bar_chart(dist.T)
    else:
        st.warning("No data found. Run the notebooks first or generate mock data.")
        st.code("python src/mock_data.py")

elif page == "Mood over time":
    st.title("Mood over time")

    if df is not None:
        monthly = (
            df.groupby(["subreddit","month"])["vader_compound"]
            .mean().reset_index()
        )
        fig, ax = plt.subplots(figsize=(12, 4))
        colors = {"depression": "#E07070", "happy": "#7BC67E"}
        for sub, group in monthly.groupby("subreddit"):
            ax.plot(group["month"], group["avg_compound"] if "avg_compound"
                    in group else group["vader_compound"],
                    label=f"r/{sub}", color=colors[sub], linewidth=2,
                    marker="o", markersize=4)
        ax.axhline(0, color="#cccccc", linewidth=0.8, linestyle="--")
        ax.set_ylabel("avg VADER compound")
        ax.legend(frameon=False)
        ax.spines[["top","right"]].set_visible(False)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("No data found.")

elif page == "Word explorer":
    st.title("Word explorer")

    if df is not None:
        sub = st.selectbox("Choose subreddit", ["depression", "happy"])
        from collections import Counter
        text   = " ".join(df[df["subreddit"]==sub]["clean_text"].dropna())
        counts = Counter(text.split()).most_common(30)
        words, freqs = zip(*counts)

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(words[::-1], freqs[::-1],
                color="#E07070" if sub=="depression" else "#7BC67E",
                edgecolor="white", linewidth=0.5)
        ax.set_title(f"r/{sub} — top 30 words", fontsize=13)
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("No data found.")

elif page == "Live predictor":
    st.title("Live mood predictor")
    st.markdown("Type any text and see the predicted mood.")

    user_input = st.text_area("Enter text here", height=120,
                               placeholder="e.g. I feel completely lost today...")

    if st.button("Predict") and user_input.strip():
        cleaned = clean_text(user_input)

        if sk is not None:
            vec   = sk["vectorizer"].transform([cleaned])
            pred  = sk["model"].predict(vec)[0]
            proba = sk["model"].predict_proba(vec)[0]
            label = sk["classes"][pred]

            st.subheader("sklearn result")
            color = {"positive":"🟢","neutral":"⚪","negative":"🔴"}
            st.markdown(f"### {color.get(label,'')} {label.upper()}")

            conf_df = pd.DataFrame({
                "mood":       sk["classes"],
                "confidence": proba,
            }).set_index("mood")
            st.bar_chart(conf_df)

        if lstm is not None:
            seq  = tf_data["tokenizer"].texts_to_sequences([cleaned])
            pad  = pad_sequences(seq, maxlen=tf_data["max_len"],
                                  padding="post", truncating="post")
            prob = lstm.predict(pad, verbose=0)[0]
            pred = prob.argmax()
            label_lstm = tf_data["classes"][pred]

            st.subheader("LSTM result")
            st.markdown(f"### {color.get(label_lstm,'')} {label_lstm.upper()}")

            conf_df2 = pd.DataFrame({
                "mood":       tf_data["classes"],
                "confidence": prob,
            }).set_index("mood")
            st.bar_chart(conf_df2)

        if sk is None and lstm is None:
            st.warning("Models not found. Run the classifier notebooks first.")