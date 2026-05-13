import streamlit as st

# consistent color palette
COLORS = {
    "depression": "#E07070",
    "happy":      "#7BC67E",
    "positive":   "#7BC67E",
    "neutral":    "#B0B0B0",
    "negative":   "#E07070",
    "sadness":    "#6B9FD4",
    "happiness":  "#7BC67E",
    "anxiety":    "#F4A261",
    "anger":      "#E07070",
    "hope":       "#C3A6E8",
}

MOOD_EMOJI = {
    "positive": "🟢",
    "neutral":  "⚪",
    "negative": "🔴",
}

def mood_badge(mood):
    """Render a colored mood badge."""
    color = COLORS.get(mood, "#B0B0B0")
    emoji = MOOD_EMOJI.get(mood, "")
    st.markdown(
        f'<span style="background:{color}22;color:{color};'
        f'padding:4px 12px;border-radius:99px;font-weight:500;'
        f'border:1px solid {color}55;">{emoji} {mood.upper()}</span>',
        unsafe_allow_html=True
    )

def page_header(title, subtitle=None, icon=None):
    """Consistent page header."""
    if icon:
        st.title(f"{icon} {title}")
    else:
        st.title(title)
    if subtitle:
        st.markdown(subtitle)
    st.divider()

def no_data_warning():
    """Consistent no-data message."""
    st.warning("No dataset loaded.")
    st.markdown("Run one of these first:")
    col1, col2 = st.columns(2)
    with col1:
        st.code("python src/mock_data.py")
        st.caption("Instant mock data — no Reddit needed")
    with col2:
        st.code("jupyter notebook\n# run 01_fetch + 02_clean")
        st.caption("Real Reddit data")

def model_missing_warning():
    """Consistent model-missing message."""
    st.warning("Trained models not found.")
    st.code("""
# Run these notebooks in order:
jupyter notebook
# 11_classifier_sklearn
# 12_tensorflow
# 14_lstm
    """)

def bar_colors(labels):
    """Return color list matching labels."""
    return [COLORS.get(l, "#B0B0B0") for l in labels]

def subheader(text):
    """Consistent subheader with less top padding."""
    st.markdown(f"#### {text}")

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
            with st.spinner("Loading LSTM model..."):
                return tf.keras.models.load_model(path)
    return None