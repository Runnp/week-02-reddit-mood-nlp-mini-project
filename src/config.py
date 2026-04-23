import os
from dotenv import load_dotenv

load_dotenv()

# Reddit API
REDDIT_CLIENT_ID     = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT    = os.getenv("REDDIT_USER_AGENT", "mood_study_bot/0.1")

# Subreddits to study
SUBREDDITS = ["depression", "happy"]

# Data paths
DATA_RAW   = "data/raw/"
DATA_CLEAN = "data/clean/"
OUTPUTS    = "outputs/"

# Fetch settings
FETCH_LIMIT       = 500
FETCH_TIME_FILTER = "year"   # "day","week","month","year","all"
