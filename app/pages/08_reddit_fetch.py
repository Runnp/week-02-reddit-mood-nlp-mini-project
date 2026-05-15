import sys, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
APP  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC  = os.path.abspath(os.path.join(ROOT, "src"))
for p in [ROOT, APP, SRC]:
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st
import pandas as pd
import time
from app.style import page_header

st.set_page_config(page_title="Reddit Fetch", page_icon="🔗", layout="wide")
page_header("🔗 Reddit Fetch", "Connect to Reddit API and pull live posts.")

# ── API key inputs ────────────────────────
st.subheader("Reddit API credentials")
st.markdown("Get your keys at [reddit.com/prefs/apps](https://reddit.com/prefs/apps) — create a **script** app.")

col1, col2 = st.columns(2)
with col1:
    client_id     = st.text_input("Client ID",     type="password")
    client_secret = st.text_input("Client Secret", type="password")
with col2:
    user_agent    = st.text_input("User Agent",    value="mood_study_bot/0.1")
    subreddits    = st.multiselect("Subreddits",
                                    ["depression","happy","anxiety",
                                     "mentalhealth","CasualConversation"],
                                    default=["depression","happy"])

col3, col4 = st.columns(2)
with col3:
    limit       = st.slider("Posts per subreddit", 50, 500, 200)
with col4:
    time_filter = st.selectbox("Time filter",
                                ["year","month","week","all"],
                                index=0)

st.divider()

if st.button("🔌 Test connection", type="secondary"):
    if not client_id or not client_secret:
        st.error("Enter your Client ID and Secret first.")
    else:
        try:
            import praw
            reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent
            )
            with st.spinner("Testing connection..."):
                for sub in subreddits:
                    s = reddit.subreddit(sub)
                    st.success(f"r/{s.display_name} — {s.subscribers:,} subscribers ✅")
        except Exception as e:
            st.error(f"Connection failed: {e}")

st.divider()

if st.button("🚀 Fetch posts", type="primary"):
    if not client_id or not client_secret:
        st.error("Enter your API credentials first.")
    elif not subreddits:
        st.error("Select at least one subreddit.")
    else:
        try:
            import praw
            reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent
            )

            all_frames = []
            progress   = st.progress(0)
            status     = st.empty()

            for i, sub_name in enumerate(subreddits):
                status.markdown(f"Fetching r/{sub_name}...")
                subreddit = reddit.subreddit(sub_name)
                posts     = []

                for post in subreddit.top(time_filter=time_filter,
                                           limit=limit):
                    posts.append({
                        "id":           post.id,
                        "subreddit":    sub_name,
                        "title":        post.title,
                        "text":         post.selftext,
                        "score":        post.score,
                        "num_comments": post.num_comments,
                        "created_utc":  post.created_utc,
                    })
                    time.sleep(0.05)

                df_sub = pd.DataFrame(posts)
                df_sub["created_at"] = pd.to_datetime(
                    df_sub["created_utc"], unit="s")
                df_sub["month"] = df_sub["created_at"].dt.to_period("M").astype(str)
                all_frames.append(df_sub)
                st.success(f"r/{sub_name} — {len(df_sub)} posts fetched ✅")
                progress.progress((i+1) / len(subreddits))

            df = pd.concat(all_frames, ignore_index=True)
            os.makedirs("data/raw", exist_ok=True)
            df.to_csv("data/raw/posts_raw.csv", index=False)

            progress.empty()
            status.empty()
            st.balloons()
            st.success(f"Done — {len(df)} posts saved to data/raw/posts_raw.csv")
            st.dataframe(df[["subreddit","title","month","score"]].head(10),
                         use_container_width=True)

        except Exception as e:
            st.error(f"Fetch failed: {e}")