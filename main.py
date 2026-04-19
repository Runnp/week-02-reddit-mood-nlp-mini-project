import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from utils import ensure_dirs
from config import SUBREDDITS, FETCH_LIMIT, FETCH_TIME_FILTER

def main():
    print("=== Reddit Mood Tracker ===\n")

    print("Checking folders...")
    ensure_dirs()

    print(f"\nProject settings:")
    print(f"  Subreddits  : {SUBREDDITS}")
    print(f"  Fetch limit : {FETCH_LIMIT} posts each")
    print(f"  Time filter : {FETCH_TIME_FILTER}")
    print(f"\nRun notebooks/ in order:")
    print(f"  00_setup   → verify environment")
    print(f"  01_fetch   → pull Reddit data")
    print(f"  02_clean   → preprocess text")
    print(f"  03_sentiment → VADER analysis")
    print(f"  04_vocab   → word frequency & clouds")
    print(f"  05_themes  → theme heatmap")
    print(f"  06_tfidf   → TF-IDF signature words")
    print(f"  07_classify → mood classifier")

if __name__ == "__main__":
    main()