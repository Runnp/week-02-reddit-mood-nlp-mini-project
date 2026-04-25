import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

STOPWORDS = set(stopwords.words("english"))

def clean_text(text):
    """Clean a single post's text."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)       # remove URLs
    text = re.sub(r"\[deleted\]|\[removed\]", "", text)  # remove deleted markers
    text = re.sub(r"[^a-z\s]", "", text)              # keep only letters
    text = re.sub(r"\s+", " ", text).strip()          # collapse whitespace
    return text

def tokenize(text):
    """Tokenize and remove stopwords."""
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha() and t not in STOPWORDS]
    return tokens

def preprocess_df(df):
    """Apply cleaning to a full DataFrame."""
    print("Cleaning text...")
    df = df.copy()

    # combine title + body into one field
    df["raw_text"] = df["title"].fillna("") + " " + df["text"].fillna("")
    df["clean_text"] = df["raw_text"].apply(clean_text)
    df["tokens"] = df["clean_text"].apply(tokenize)
    df["token_count"] = df["tokens"].apply(len)

    # drop empty posts
    before = len(df)
    df = df[df["token_count"] > 3].reset_index(drop=True)
    print(f"Dropped {before - len(df)} empty/short posts")
    print(f"Remaining: {len(df)} posts")
    return df