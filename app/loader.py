import os
import sys
import pickle
import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

@st.cache_data
def load_data():
    path = "data/clean/posts_sentiment.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df
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
        st.warning("No dataset found. Run mock data first.")
        st.code("python src/mock_data.py")
        st.stop()