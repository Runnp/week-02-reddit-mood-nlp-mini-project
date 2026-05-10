import re
import numpy as np
import pandas as pd
from collections import Counter

def avg_word_length(text):
    if not isinstance(text, str):
        return 0
    words = [w for w in text.split() if w.isalpha()]
    return round(np.mean([len(w) for w in words]), 3) if words else 0

def avg_sentence_length(text):
    if not isinstance(text, str):
        return 0
    sents = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
    words = text.split()
    return round(len(words) / len(sents), 3) if sents else 0

def type_token_ratio(text):
    if not isinstance(text, str):
        return 0
    words = [w.lower() for w in text.split() if w.isalpha()]
    return round(len(set(words)) / len(words), 3) if words else 0

def repetition_score(text):
    """How much of the text is repeated words — higher = more repetitive."""
    if not isinstance(text, str):
        return 0
    words  = [w.lower() for w in text.split() if w.isalpha()]
    if not words:
        return 0
    counts = Counter(words)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return round(repeated / len(words), 3)

def exclamation_rate(text):
    """Proportion of sentences ending in exclamation mark."""
    if not isinstance(text, str):
        return 0
    sents = re.split(r'[.!?]', text)
    excl  = text.count("!")
    return round(excl / len(sents), 3) if sents else 0

def question_rate(text):
    """Proportion of sentences ending in question mark."""
    if not isinstance(text, str):
        return 0
    sents = re.split(r'[.!?]', text)
    qs    = text.count("?")
    return round(qs / len(sents), 3) if sents else 0

def add_text_stats(df, text_col="clean_text", raw_col=None):
    """Add all text stat columns to a DataFrame."""
    col = raw_col if raw_col and raw_col in df.columns else text_col
    df  = df.copy()
    df["avg_word_len"]    = df[col].apply(avg_word_length)
    df["avg_sent_len"]    = df[col].apply(avg_sentence_length)
    df["ttr"]             = df[col].apply(type_token_ratio)
    df["repetition"]      = df[col].apply(repetition_score)
    df["exclamation_rate"]= df[col].apply(exclamation_rate)
    df["question_rate"]   = df[col].apply(question_rate)
    print("Text stats added:")
    print(df[["subreddit","avg_word_len","avg_sent_len",
              "ttr","repetition"]].groupby("subreddit").mean().round(3))
    return df