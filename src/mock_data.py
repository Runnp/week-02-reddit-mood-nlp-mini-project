import pandas as pd
import numpy as np
import random
import time

random.seed(42)
np.random.seed(42)

# Realistic sample posts per subreddit
DEPRESSION_POSTS = [
    "I can't get out of bed anymore. Everything feels pointless and heavy.",
    "Does anyone else feel completely alone even when surrounded by people?",
    "I've been crying for no reason again. I don't know how much longer I can do this.",
    "Lost my job last month and I feel like I'm disappearing. Nobody notices.",
    "Therapy isn't working. Meds aren't working. I'm so tired of trying.",
    "I used to love music. Now nothing makes me feel anything at all.",
    "Three years of this and I still don't know how to explain it to my family.",
    "Today was a small win — I showered and made food. That's all I had.",
    "The hardest part is pretending to be okay at work every single day.",
    "I don't want to die, I just want to stop feeling like this.",
    "Relapsed again after six months. Feel like such a failure.",
    "My brain keeps telling me nobody would miss me. I know it's not true but.",
    "Does it ever actually get better or do we just learn to cope better?",
    "Woke up at 2pm again. Another day gone. I hate myself for this.",
    "Finally told my doctor. She actually listened. Small step but it felt huge.",
]

HAPPY_POSTS = [
    "Got the job I've been working towards for two years. I actually cried.",
    "My dog learned a new trick today and I am unreasonably proud of him.",
    "Stranger paid for my coffee this morning and it made my whole week.",
    "First time hiking alone and I feel like a completely different person.",
    "Six months sober today. Never thought I'd make it this far.",
    "My kid said I was their best friend today. I am not okay in the best way.",
    "Finished my first marathon at 47. It's never too late.",
    "Reconnected with my childhood best friend after 12 years. Like no time passed.",
    "Cooked a proper meal from scratch for the first time. Small thing, huge feeling.",
    "Got a handwritten thank you letter from a student I taught three years ago.",
    "Moved to a new city knowing nobody. Six months later I have genuine friends.",
    "My anxiety has been so much better this month. Slowly getting my life back.",
    "Random act of kindness thread — drop yours below, I need the serotonin.",
    "Proposed to my partner at our local park. She said yes obviously.",
    "Just sat in the sun for an hour doing nothing. Forgot how good that feels.",
]

def generate_mock_posts(subreddit, posts_list, n=500):
    rows = []
    base_time = int(time.time()) - (365 * 24 * 3600)  # one year ago

    for i in range(n):
        template = random.choice(posts_list)
        # add slight variation so posts aren't identical
        filler = random.choice([
            "", " Really needed to share this.",
            " Thanks for reading.",
            " Anyone else feel this way?",
            " Not sure why I'm posting this.",
            " Just needed to get this out.",
        ])
        rows.append({
            "id":           f"{subreddit[:3]}_{i:04d}",
            "subreddit":    subreddit,
            "title":        template,
            "text":         template + filler,
            "score":        int(np.random.exponential(500)),
            "num_comments": int(np.random.exponential(80)),
            "created_utc":  base_time + int(np.random.uniform(0, 365 * 24 * 3600)),
        })

    return pd.DataFrame(rows)

def generate_all(save=True):
    print("Generating mock data...")
    df_dep = generate_mock_posts("depression", DEPRESSION_POSTS, n=500)
    df_hap = generate_mock_posts("happy",      HAPPY_POSTS,      n=500)

    df = pd.concat([df_dep, df_hap], ignore_index=True)
    df["created_at"] = pd.to_datetime(df["created_utc"], unit="s")
    df["month"]      = df["created_at"].dt.to_period("M").astype(str)

    print(f"Generated {len(df)} mock posts")
    print(df["subreddit"].value_counts())

    if save:
        import os
        os.makedirs("data/raw", exist_ok=True)
        df.to_csv("data/raw/posts_raw.csv", index=False)
        print("Saved to data/raw/posts_raw.csv")

    return df

if __name__ == "__main__":
    generate_all()