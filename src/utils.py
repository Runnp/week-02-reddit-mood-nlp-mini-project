import os
import pandas as pd

def ensure_dirs():
    """Create all necessary folders if they don't exist."""
    for folder in ["data/raw", "data/clean", "outputs", "notebooks"]:
        os.makedirs(folder, exist_ok=True)
        print(f"  ✓ {folder}/")

def load_raw(filename="posts_raw.csv"):
    """Load raw CSV from data/raw/."""
    path = os.path.join("data/raw", filename)
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path}")
    return df

def load_clean(filename="posts_clean.csv"):
    """Load cleaned CSV from data/clean/."""
    path = os.path.join("data/clean", filename)
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path}")
    return df

def save_raw(df, filename="posts_raw.csv"):
    """Save DataFrame to data/raw/."""
    path = os.path.join("data/raw", filename)
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} rows → {path}")

def save_clean(df, filename="posts_clean.csv"):
    """Save DataFrame to data/clean/."""
    path = os.path.join("data/clean", filename)
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} rows → {path}")

def summary(df):
    """Quick overview of a DataFrame."""
    print(f"Shape     : {df.shape}")
    print(f"Columns   : {list(df.columns)}")
    print(f"Subreddits: {df['subreddit'].value_counts().to_dict()}")
    print(f"Nulls     :\n{df.isnull().sum()}")