import time
import pandas as pd

def fetch_posts(reddit, subreddit_name, limit=500, time_filter="year"):
    subreddit = reddit.subreddit(subreddit_name)
    posts = []

    for post in subreddit.top(time_filter=time_filter, limit=limit):
        posts.append({
            "id":           post.id,
            "subreddit":    subreddit_name,
            "title":        post.title,
            "text":         post.selftext,
            "score":        post.score,
            "num_comments": post.num_comments,
            "created_utc":  post.created_utc,
        })
        time.sleep(0.05)

    df = pd.DataFrame(posts)
    df["created_at"] = pd.to_datetime(df["created_utc"], unit="s")
    df["month"]      = df["created_at"].dt.to_period("M").astype(str)

    print(f"r/{subreddit_name}: {len(df)} posts fetched")
    return df

def fetch_all(reddit, subreddits, limit=500, time_filter="year"):
    frames = []
    for name in subreddits:
        df = fetch_posts(reddit, name, limit, time_filter)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    print(f"Total: {len(combined)} posts")
    return combined

def build_reddit(client_id, client_secret, user_agent="mood_bot/0.1"):
    import praw
    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent
    )