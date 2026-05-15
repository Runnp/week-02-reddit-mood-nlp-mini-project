import sys, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
APP  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC  = os.path.abspath(os.path.join(ROOT, "src"))
for p in [ROOT, APP, SRC]:
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st
import json
import os
from app.style import page_header

st.set_page_config(page_title="Settings", page_icon="⚙️", layout="wide")
page_header("⚙️ Settings", "Configure the app behaviour and analysis parameters.")

SETTINGS_PATH = "app/settings.json"

def load_settings():
    if os.path.exists(SETTINGS_PATH):
        with open(SETTINGS_PATH) as f:
            return json.load(f)
    return {}

def save_settings(settings):
    with open(SETTINGS_PATH, "w") as f:
        json.dump(settings, f, indent=2)

settings = load_settings()

# ── sentiment thresholds ──────────────────
st.subheader("Sentiment thresholds")
st.markdown("Adjust how VADER compound scores are labelled.")

col1, col2 = st.columns(2)
with col1:
    pos_thresh = st.slider("Positive threshold",
                            0.01, 0.2,
                            float(settings.get("pos_thresh", 0.05)),
                            step=0.01)
with col2:
    neg_thresh = st.slider("Negative threshold",
                            -0.2, -0.01,
                            float(settings.get("neg_thresh", -0.05)),
                            step=0.01)

st.caption(f"compound ≥ {pos_thresh} → positive  |  compound ≤ {neg_thresh} → negative  |  else → neutral")

st.divider()

# ── fetch settings ────────────────────────
st.subheader("Fetch settings")
col3, col4 = st.columns(2)
with col3:
    fetch_limit = st.slider("Default fetch limit",
                             50, 1000,
                             int(settings.get("fetch_limit", 500)),
                             step=50)
with col4:
    time_filter = st.selectbox("Default time filter",
                                ["year","month","week","all"],
                                index=["year","month","week","all"].index(
                                    settings.get("time_filter","year")))

st.divider()

# ── display settings ──────────────────────
st.subheader("Display settings")
col5, col6 = st.columns(2)
with col5:
    chart_dpi   = st.selectbox("Chart DPI", [100, 150, 200],
                                index=[100,150,200].index(
                                    int(settings.get("chart_dpi",150))))
    top_n_words = st.slider("Default top N words", 10, 40,
                             int(settings.get("top_n_words", 20)))
with col6:
    show_raw    = st.checkbox("Show raw data tables by default",
                               value=bool(settings.get("show_raw", True)))
    show_captions = st.checkbox("Show chart captions",
                                 value=bool(settings.get("show_captions", True)))

st.divider()

# ── save ──────────────────────────────────
if st.button("💾 Save settings", type="primary"):
    new_settings = {
        "pos_thresh":    pos_thresh,
        "neg_thresh":    neg_thresh,
        "fetch_limit":   fetch_limit,
        "time_filter":   time_filter,
        "chart_dpi":     chart_dpi,
        "top_n_words":   top_n_words,
        "show_raw":      show_raw,
        "show_captions": show_captions,
    }
    save_settings(new_settings)
    st.success("Settings saved to app/settings.json")
    st.json(new_settings)

st.divider()

# ── danger zone ───────────────────────────
st.subheader("Danger zone")
col7, col8 = st.columns(2)
with col7:
    if st.button("🗑️ Clear cached data", type="secondary"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Cache cleared — reload the page.")
with col8:
    if st.button("↩️ Reset to defaults", type="secondary"):
        if os.path.exists(SETTINGS_PATH):
            os.remove(SETTINGS_PATH)
        st.success("Settings reset to defaults.")
        st.rerun()