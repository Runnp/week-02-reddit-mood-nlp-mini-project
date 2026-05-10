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

elif page == "Word explorer":
    st.title("Word explorer")
    st.markdown("Explore vocabulary patterns across both communities.")

    if df is not None:
        tab1, tab2, tab3 = st.tabs(["Top words", "Word shift", "Unique vocab"])

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                sub = st.selectbox("Subreddit", ["depression","happy"])
            with col2:
                n_words = st.slider("Number of words", 10, 40, 20)

            from collections import Counter
            text   = " ".join(df[df["subreddit"]==sub]["clean_text"].dropna())
            counts = Counter(text.split()).most_common(n_words)
            words, freqs = zip(*counts)

            fig, ax = plt.subplots(figsize=(10, 7))
            ax.barh(words[::-1], freqs[::-1],
                    color="#E07070" if sub=="depression" else "#7BC67E",
                    edgecolor="white", linewidth=0.5)
            ax.set_title(f"r/{sub} — top {n_words} words", fontsize=13)
            ax.set_xlabel("frequency")
            ax.spines[["top","right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)

        with tab2:
            from collections import Counter
            dep_words = Counter(" ".join(
                df[df["subreddit"]=="depression"]["clean_text"].dropna()).split())
            hap_words = Counter(" ".join(
                df[df["subreddit"]=="happy"]["clean_text"].dropna()).split())

            total_dep = sum(dep_words.values())
            total_hap = sum(hap_words.values())
            all_words = set(dep_words.keys()) | set(hap_words.keys())

            shifts = []
            for word in all_words:
                fd = dep_words.get(word,0) / total_dep
                fh = hap_words.get(word,0) / total_hap
                if max(dep_words.get(word,0), hap_words.get(word,0)) >= 5:
                    shifts.append({"word": word, "shift": fd - fh})

            shift_df = pd.DataFrame(shifts).sort_values("shift", ascending=False)
            top_n    = st.slider("Words per side", 5, 20, 12)
            combined = pd.concat([shift_df.head(top_n),
                                   shift_df.tail(top_n).iloc[::-1]])
            colors   = ["#E07070" if s > 0 else "#7BC67E"
                        for s in combined["shift"]]

            fig, ax = plt.subplots(figsize=(10, 9))
            ax.barh(combined["word"], combined["shift"],
                    color=colors, edgecolor="white", linewidth=0.5)
            ax.axvline(0, color="#cccccc", linewidth=0.8)
            ax.set_title("Word shift — depression vs happy", fontsize=13)
            ax.set_xlabel("← more in r/happy       more in r/depression →")
            ax.spines[["top","right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)

        with tab3:
            dep_set  = set(dep_words.keys()) if "dep_words" in dir() else set(
                " ".join(df[df["subreddit"]=="depression"]["clean_text"].dropna()).split())
            hap_set  = set(hap_words.keys()) if "hap_words" in dir() else set(
                " ".join(df[df["subreddit"]=="happy"]["clean_text"].dropna()).split())

            from collections import Counter
            dep_counter = Counter(" ".join(
                df[df["subreddit"]=="depression"]["clean_text"].dropna()).split())
            hap_counter = Counter(" ".join(
                df[df["subreddit"]=="happy"]["clean_text"].dropna()).split())
            dep_set = set(dep_counter.keys())
            hap_set = set(hap_counter.keys())

            only_dep = dep_set - hap_set
            only_hap = hap_set - dep_set
            shared   = dep_set & hap_set

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Only in r/depression", len(only_dep))
            col_b.metric("Only in r/happy",      len(only_hap))
            col_c.metric("Shared",               len(shared))

            st.divider()
            col_d, col_e = st.columns(2)
            with col_d:
                st.markdown("**Top unique depression words:**")
                top_dep_unique = sorted(
                    {w: dep_counter[w] for w in only_dep}.items(),
                    key=lambda x: -x[1])[:15]
                for w, c in top_dep_unique:
                    st.markdown(f"- `{w}` ({c})")
            with col_e:
                st.markdown("**Top unique happy words:**")
                top_hap_unique = sorted(
                    {w: hap_counter[w] for w in only_hap}.items(),
                    key=lambda x: -x[1])[:15]
                for w, c in top_hap_unique:
                    st.markdown(f"- `{w}` ({c})")
    else:
        st.warning("No data found.")

#page = st.sidebar.radio("Navigate", [
    "Overview",
    "Mood over time",
    "Word explorer",
    "Themes",
    "Live predictor",
])

elif page == "Themes":
    st.title("Theme analysis")
    st.markdown("How often do key themes appear across both communities?")

    if df is not None:
        THEMES = {
            "support":  ["help","support","friend","together","community","care","listen"],
            "venting":  ["tired","hate","angry","frustrated","sick","done","vent"],
            "advice":   ["advice","suggest","recommend","try","tips","idea","should"],
            "recovery": ["better","improve","progress","recover","hope","healing","sober"],
            "crisis":   ["crisis","harm","hurt","die","end","helpline","emergency"],
        }

        def tag_themes(text, themes):
            if not isinstance(text, str):
                return {t: 0 for t in themes}
            words = set(text.lower().split())
            return {theme: int(bool(words & set(kws)))
                    for theme, kws in themes.items()}

        theme_cols = pd.json_normalize(
            df["clean_text"].apply(lambda t: tag_themes(t, THEMES))
        )
        theme_cols.index = df.index
        df_t = pd.concat([df, theme_cols], axis=1)

        # overall hit rates
        st.subheader("Theme hit rates")
        col1, col2 = st.columns(2)
        colors_sub = {"depression": "#E07070", "happy": "#7BC67E"}

        for col, sub in zip([col1, col2], ["depression","happy"]):
            with col:
                st.markdown(f"**r/{sub}**")
                means = df_t[df_t["subreddit"]==sub][list(THEMES.keys())].mean()
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.barh(means.index[::-1], means.values[::-1],
                        color=colors_sub[sub], edgecolor="white", linewidth=0.5)
                ax.set_xlabel("hit rate")
                ax.spines[["top","right"]].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)

        st.divider()

        # heatmap by month
        st.subheader("Theme intensity by month")
        sub_sel = st.selectbox("Subreddit", ["depression","happy"])

        import seaborn as sns
        heat = (
            df_t[df_t["subreddit"]==sub_sel]
            .groupby("month")[list(THEMES.keys())]
            .mean()
        )
        fig, ax = plt.subplots(figsize=(13, 4))
        sns.heatmap(heat.T, annot=True, fmt=".2f",
                    cmap="YlOrRd" if sub_sel=="depression" else "YlGn",
                    linewidths=0.4, ax=ax, cbar_kws={"shrink":0.6})
        ax.set_title(f"r/{sub_sel} — theme intensity by month", fontsize=13)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)

        st.divider()

        # theme comparison bar
        st.subheader("Side by side comparison")
        theme_means = df_t.groupby("subreddit")[list(THEMES.keys())].mean()
        import numpy as np
        x     = np.arange(len(THEMES))
        width = 0.35
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.bar(x - width/2, theme_means.loc["depression"],
               width, label="r/depression", color="#E07070",
               edgecolor="white", linewidth=0.5)
        ax.bar(x + width/2, theme_means.loc["happy"],
               width, label="r/happy", color="#7BC67E",
               edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(list(THEMES.keys()), fontsize=11)
        ax.set_ylabel("avg hit rate")
        ax.legend(frameon=False)
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("No data found.")