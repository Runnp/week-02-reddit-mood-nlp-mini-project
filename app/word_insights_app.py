import pandas as pd
from collections import Counter
import re

def clean_words(text):
    return re.findall(r"\b[a-zA-Z']+\b", str(text).lower())


def get_top_words(df, subreddit, top_n=20):
    texts = df[df["subreddit"] == subreddit]["title"].dropna()
    
    words = []
    for t in texts:
        words.extend(clean_words(t))
    
    return Counter(words).most_common(top_n)


def compare_vocab(df, sub1="depression", sub2="happy", top_n=15):
    words1 = Counter()
    words2 = Counter()

    for t in df[df["subreddit"] == sub1]["title"].dropna():
        words1.update(clean_words(t))

    for t in df[df["subreddit"] == sub2]["title"].dropna():
        words2.update(clean_words(t))

    all_words = set(words1.keys()) | set(words2.keys())

    score = []
    for w in all_words:
        score.append((w, words1[w], words2[w]))

    score.sort(key=lambda x: abs(x[1] - x[2]), reverse=True)

    return score[:top_n]