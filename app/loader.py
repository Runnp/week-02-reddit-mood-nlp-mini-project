import os
import sys
import pickle
import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

@st.cache_resource
def load_sklearn():
    candidates = [
        "data/clean/best_sklearn_model.pkl",
        "../data/clean/best_sklearn_model.pkl",
        os.path.join(os.path.dirname(__file__), "../data/clean/best_sklearn_model.pkl"),
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
    return None

@st.cache_resource
def load_lstm():
    candidates = [
        "data/clean/lstm_mood_model",
        "../data/clean/lstm_mood_model",
        os.path.join(os.path.dirname(__file__), "../data/clean/lstm_mood_model"),
    ]
    for path in candidates:
        if os.path.exists(path):
            import tensorflow as tf
            return tf.keras.models.load_model(path)
    return None

@st.cache_resource
def load_tokenizer():
    candidates = [
        "data/clean/tf_tokenizer.pkl",
        "../data/clean/tf_tokenizer.pkl",
        os.path.join(os.path.dirname(__file__), "../data/clean/tf_tokenizer.pkl"),
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
    return None

@st.cache_data
def load_data():
    candidates = [
        "data/clean/posts_sentiment.csv",
        "../data/clean/posts_sentiment.csv",
        os.path.join(os.path.dirname(__file__), "../data/clean/posts_sentiment.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return pd.read_csv(path)
    return None

@st.cache_resource
def load_sklearn():
    path = "data/clean/best_sklearn_model.pkl"
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_lstm():
    path = "data/clean/lstm_mood_model"
    if not os.path.exists(path):
        return None
    import tensorflow as tf
    return tf.keras.models.load_model(path)

@st.cache_resource
def load_tokenizer():
    path = "data/clean/tf_tokenizer.pkl"
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

def load_all():
    return {
        "df":        load_data(),
        "sk":        load_sklearn(),
        "lstm":      load_lstm(),
        "tokenizer": load_tokenizer(),
    }

def data_missing(df):
    if df is None:
        st.warning("No dataset found.")
        col1, col2 = st.columns(2)
        with col1:
            st.code("python src/mock_data.py", language="bash")
            st.caption("Generate mock data instantly")
        with col2:
            st.code("jupyter notebook\n# then run 01_fetch + 02_clean",
                    language="bash")
            st.caption("Or fetch real Reddit data")
        st.stop()

@st.cache_data
def load_data():
    # try multiple paths so it works from any working directory
    candidates = [
        "data/clean/posts_sentiment.csv",
        "../data/clean/posts_sentiment.csv",
        os.path.join(os.path.dirname(__file__), "../data/clean/posts_sentiment.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return pd.read_csv(path)
    return None