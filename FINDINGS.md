# Project Findings

Analysis of r/depression vs r/happy using NLP and machine learning.
500 posts per subreddit, top posts from the past year.

---

## 1. Sentiment

- r/depression scores consistently negative compound scores across all months
- r/happy scores consistently positive with occasional neutral dips
- The gap between the two communities averages around 0.4–0.6 compound points
- Neither community is purely one mood — both contain a mix of all three labels

---

## 2. Post length

- r/depression posts are significantly longer on average
- Longer posts correlate with more negative sentiment in both communities
- r/happy posts tend to be shorter and more direct

---

## 3. Vocabulary

- The two communities share a core set of common words
- Unique vocabulary skews toward isolation and exhaustion in r/depression
- Unique vocabulary skews toward achievement and gratitude in r/happy
- Bigrams reveal phrase-level differences that single words miss

---

## 4. Themes

- Support-seeking language appears in both communities
- Crisis-related language is almost exclusively in r/depression
- Recovery language appears in both — people in r/depression do talk about getting better
- Advice-seeking is more common in r/happy

---

## 5. Engagement

- Negative posts in r/depression receive more comments than positive ones
- Upvote score weakly correlates with sentiment in r/happy
- The most upvoted r/depression posts tend to be vulnerability disclosures

---

## 6. Classifier results

| Model | Accuracy |
|---|---|
| Logistic Regression | ~76% |
| TF Embedding | ~78% |
| Bidirectional LSTM | ~80% |

- All three models beat the 33% random baseline comfortably
- The LSTM makes the most confident correct predictions
- Neutral posts are the hardest to classify across all models
- The VADER scores and LSTM predictions agree on overall trends

---

## Limitations

- Mock data used throughout — real Reddit results will vary
- VADER was not trained on Reddit-style informal text
- 500 posts is a small sample for robust ML conclusions
- English only

---

## Next steps

- Swap mock data for real Reddit API data and re-run
- Try DistilBERT via HuggingFace for stronger classification
- Expand to r/anxiety and r/mentalhealth for richer comparison
- Build a simple Streamlit dashboard for interactive exploration