import os
import sys
import pickle
import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

# ── Data Loading Functions ────────────────────────

@st.cache_data
def load_data():
    """Load main dataset from CSV."""
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
    """Load sklearn model (LR/RF)."""
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
    """Load LSTM model (requires TensorFlow)."""
    try:
        import tensorflow as tf
    except ImportError:
        st.warning("⚠️ TensorFlow not installed. LSTM predictions disabled.")
        return None
    
    candidates = [
        "data/clean/lstm_mood_model",
        "../data/clean/lstm_mood_model",
        os.path.join(os.path.dirname(__file__), "../data/clean/lstm_mood_model"),
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return tf.keras.models.load_model(path)
            except Exception as e:
                return None
    return None

@st.cache_resource
def load_tokenizer():
    """Load TensorFlow tokenizer."""
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

def load_models():
    """Load all ML models and tokenizer."""
    return {
        "sklearn": load_sklearn(),
        "lstm": load_lstm(),
        "tokenizer": load_tokenizer(),
    }

def load_models():
    """Load all ML models and tokenizer."""
    return {
        "sklearn": load_sklearn(),
        "lstm": load_lstm(),
        "tokenizer": load_tokenizer(),
    }

def data_missing_warning():
    """Display warning when data is not loaded."""
    if True:
        st.warning("📦 No dataset found. Generate or fetch data first.")
        col1, col2 = st.columns(2)
        with col1:
            st.code("python src/mock_data.py", language="bash")
            st.caption("Generate mock data instantly")
        with col2:
            st.code("jupyter notebook\n# then run 01_fetch + 02_clean",
                    language="bash")
            st.caption("Or fetch real Reddit data")