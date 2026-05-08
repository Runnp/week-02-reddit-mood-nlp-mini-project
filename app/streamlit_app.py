import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
#import tensorflow as tf
#from tensorflow.keras.preprocessing.sequence import pad_sequences
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
#        lstm = tf.keras.models.load_model("data/clean/lstm_mood_model")
        return sk, tf_data, lstm
    except Exception:
        return None, None, None

df      = load_data()
sk, tf_data, lstm = load_models()

# ── Pages ─────────────────────────────────────────
    # if page == "Overview":
    # st.title("Reddit Mood Shift NLP")
    # st.markdown("Comparing emotional language patterns across **r/depression** and **r/happy** — 500 posts each, past year.")

    # if df is not None:
    #     # metrics row
    #     col1, col2, col3, col4 = st.columns(4)
    #     col1.metric("Total posts",     len(df))
    #     col2.metric("r/depression",    len(df[df["subreddit"]=="depression"]))
    #     col3.metric("r/happy",         len(df[df["subreddit"]=="happy"]))
    #     col4.metric("Months covered",  df["month"].nunique())

    #     st.divider()

    #     # sentiment gap
    #     col_a, col_b = st.columns(2)
    #     with col_a:
    #         st.subheader("Avg sentiment score")
    #         avgs = df.groupby("subreddit")["vader_compound"].mean()
    #         fig, ax = plt.subplots(figsize=(5, 3))
    #         colors  = ["#E07070" if s=="depression" else "#7BC67E"
    #                    for s in avgs.index]
    #         bars = ax.bar(avgs.index, avgs.values,
    #                       color=colors, edgecolor="white", linewidth=0.5)
    #         ax.axhline(0, color="#cccccc", linewidth=0.8, linestyle="--")
    #         ax.set_ylabel("compound score")
    #         ax.spines[["top","right"]].set_visible(False)
    #         for bar, val in zip(bars, avgs.values):
    #             ax.text(bar.get_x() + bar.get_width()/2,
    #                     bar.get_height() + 0.005,
    #                     f"{val:+.3f}", ha="center", va="bottom", fontsize=10)
    #         plt.tight_layout()
    #         st.pyplot(fig)

    #     with col_b:
    #         st.subheader("Mood distribution")
    #         dist = (
    #             df.groupby(["subreddit","mood_label"])
    #             .size().unstack(fill_value=0)
    #         )
    #         dist_pct = dist.div(dist.sum(axis=1), axis=0).round(3)
    #         st.dataframe(dist_pct, use_container_width=True)
    #         st.caption("Proportion of posts per mood label per subreddit.")

    #     st.divider()

    #     # avg post length
    #     st.subheader("Avg post length")
    #     if "token_count" in df.columns:
    #         col_c, col_d = st.columns(2)
    #         for col, sub in zip([col_c, col_d], ["depression","happy"]):
    #             avg = df[df["subreddit"]==sub]["token_count"].mean()
    #             col.metric(f"r/{sub}", f"{avg:.0f} tokens/post")

    #     # raw data peek
    #     st.divider()
    #     st.subheader("Raw data sample")
    #     st.dataframe(
    #         df[["subreddit","title","mood_label",
    #             "vader_compound","month"]].head(10),
    #         use_container_width=True
    #     )
    # else:
    #     st.warning("No data found — run notebooks or generate mock data first.")
    #     st.code("python src/mock_data.py")


# elif page == "Mood over time":
#     st.title("Mood over time")
#     st.markdown("Average VADER compound score per month across both communities.")

#     if df is not None:
#         # filter controls
#         col1, col2 = st.columns(2)
#         with col1:
#             subs = st.multiselect("Subreddits", ["depression","happy"],
#                                    default=["depression","happy"])
#         with col2:
#             months = sorted(df["month"].dropna().unique())
#             month_range = st.select_slider("Month range",
#                                             options=months,
#                                             value=(months[0], months[-1]))

#         filtered = df[
#             (df["subreddit"].isin(subs)) &
#             (df["month"] >= month_range[0]) &
#             (df["month"] <= month_range[1])
#         ]

#         monthly = (
#             filtered.groupby(["subreddit","month"])["vader_compound"]
#             .mean().reset_index()
#         )

#         # line chart
#         fig, ax = plt.subplots(figsize=(12, 4))
#         colors  = {"depression":"#E07070","happy":"#7BC67E"}
#         for sub, group in monthly.groupby("subreddit"):
#             ax.plot(group["month"], group["vader_compound"],
#                     label=f"r/{sub}", color=colors.get(sub,"#555555"),
#                     linewidth=2, marker="o", markersize=4)
#         ax.axhline(0, color="#cccccc", linewidth=0.8, linestyle="--")
#         ax.set_ylabel("avg VADER compound score")
#         ax.legend(frameon=False)
#         ax.spines[["top","right"]].set_visible(False)
#         plt.xticks(rotation=45, ha="right")
#         plt.tight_layout()
#         st.pyplot(fig)

#         st.divider()

#         # monthly stats table
#         st.subheader("Monthly breakdown")
#         pivot = monthly.pivot(index="month", columns="subreddit",
#                                values="vader_compound").round(3)
#         st.dataframe(pivot, use_container_width=True)

#         # most positive and negative months
#         st.divider()
#         col_a, col_b = st.columns(2)
#         for col, sub in zip([col_a, col_b], ["depression","happy"]):
#             sub_monthly = monthly[monthly["subreddit"]==sub]
#             if len(sub_monthly):
#                 best  = sub_monthly.loc[sub_monthly["vader_compound"].idxmax()]
#                 worst = sub_monthly.loc[sub_monthly["vader_compound"].idxmin()]
#                 with col:
#                     st.subheader(f"r/{sub}")
#                     st.metric("Best month",  best["month"],
#                                f"{best['vader_compound']:+.3f}")
#                     st.metric("Worst month", worst["month"],
#                                f"{worst['vader_compound']:+.3f}")
#     else:
#         st.warning("No data found.")

# elif page == "Live predictor":
#     st.title("Live mood predictor")
#     st.markdown("Type any text and get mood predictions from both models.")

#     col1, col2 = st.columns([2, 1])

#     with col1:
#         user_input = st.text_area("Enter text", height=150,
#                                    placeholder="e.g. I feel completely lost and alone today...")

#     with col2:
#         st.markdown("**Examples to try:**")
#         examples = [
#             "I can't get out of bed anymore",
#             "Got the job I've been working for",
#             "Just existing today, not great not terrible",
#             "Six months sober, never thought I'd make it",
#             "Everything feels pointless and heavy",
#         ]
#         for ex in examples:
#             if st.button(ex, key=ex):
#                 user_input = ex

#     if st.button("🔍 Predict mood", type="primary") and user_input and user_input.strip():
#         cleaned = clean_text(user_input)
#         st.divider()

#         col_sk, col_lstm = st.columns(2)
#         mood_emoji = {"positive":"🟢","neutral":"⚪","negative":"🔴"}

#         # sklearn
#         with col_sk:
#             st.subheader("sklearn model")
#             if sk is not None:
#                 vec   = sk["vectorizer"].transform([cleaned])
#                 pred  = sk["model"].predict(vec)[0]
#                 proba = sk["model"].predict_proba(vec)[0]
#                 label = sk["classes"][pred]
#                 st.markdown(f"### {mood_emoji.get(label,'')} {label.upper()}")
#                 conf_df = pd.DataFrame({
#                     "confidence": proba
#                 }, index=sk["classes"])
#                 st.bar_chart(conf_df)
#             else:
#                 st.warning("sklearn model not found.")

#         # lstm
#         with col_lstm:
#             st.subheader("LSTM model")
#             if lstm is not None:
#                 seq   = tf_data["tokenizer"].texts_to_sequences([cleaned])
#                 pad   = pad_sequences(seq, maxlen=tf_data["max_len"],
#                                        padding="post", truncating="post")
#                 prob  = lstm.predict(pad, verbose=0)[0]
#                 pred  = prob.argmax()
#                 label_lstm = tf_data["classes"][pred]
#                 st.markdown(f"### {mood_emoji.get(label_lstm,'')} {label_lstm.upper()}")
#                 conf_df2 = pd.DataFrame({
#                     "confidence": prob
#                 }, index=tf_data["classes"])
#                 st.bar_chart(conf_df2)
#             else:
#                 st.warning("LSTM model not found.")

#         # agreement check
#         st.divider()
#         if sk is not None and lstm is not None:
#             if label == label_lstm:
#                 st.success(f"Both models agree — **{label.upper()}**")
#             else:
#                 st.info(f"Models disagree — sklearn: **{label}** | LSTM: **{label_lstm}**")

#         # cleaned text peek
#         with st.expander("See cleaned text"):
#             st.code(cleaned)