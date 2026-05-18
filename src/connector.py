import os
from dotenv import load_dotenv

load_dotenv()

def get_reddit_client():
    """
    Build a Reddit client from .env file.
    Returns praw.Reddit or None if keys missing.
    """
    client_id     = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent    = os.getenv("REDDIT_USER_AGENT", "mood_bot/0.1")

    if not client_id or not client_secret:
        print("Reddit API keys not found in .env")
        print("Add REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET to your .env file")
        return None

    import praw
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent
    )
    print("Reddit client ready")
    return reddit

def test_connection(reddit, subreddits=None):
    """Quick connection test — print subscriber counts."""
    if reddit is None:
        print("No client — check your .env file")
        return False

    subs = subreddits or ["depression", "happy"]
    try:
        for name in subs:
            sub = reddit.subreddit(name)
            print(f"  r/{sub.display_name:15} {sub.subscribers:,} subscribers")
        return True
    except Exception as e:
        print(f"Connection failed: {e}")
        return False

def fetch_and_run(subreddits=None, limit=500, time_filter="year"):
    """
    One-call function:
    connect → fetch → pipeline → return sentiment DataFrame.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    from fetcher  import fetch_all
    from pipeline import run_pipeline

    reddit = get_reddit_client()
    if reddit is None:
        print("Falling back to mock data...")
        from pipeline import run_from_mock
        return run_from_mock()

    subs   = subreddits or ["depression", "happy"]
    df_raw = fetch_all(reddit, subs, limit=limit,
                       time_filter=time_filter)
    return run_pipeline(df_raw)