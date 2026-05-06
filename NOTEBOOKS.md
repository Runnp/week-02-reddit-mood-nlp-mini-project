# Notebook Guide

A walkthrough of every notebook in order — what it does,
what it produces, and what to look for in the output.

---

## Week 1 — Data & classic NLP

### 00_setup
Verifies the full environment — imports, NLTK data, Reddit API
connection, and folder structure. Run this first every time you
clone the repo on a new machine.

**Look for:** both subreddits printing with subscriber counts.

---

### 01_fetch
Connects to Reddit via PRAW and pulls 500 posts from each
subreddit. Adds timestamps and saves to `data/raw/posts_raw.csv`.

**Look for:** `depression 500 / happy 500` in the output.
**Note:** skip this and run `python src/mock_data.py` if you
don't have Reddit API keys set up yet.

---

### 02_clean
Removes URLs, deleted markers, punctuation, and stopwords.
Combines title and body into one clean text field. Saves to
`data/clean/posts_clean.csv`.

**Look for:** token distribution chart — r/depression posts
should skew longer.

---

### 03_sentiment
Runs VADER on every post and adds compound score and mood label
columns. Groups by month and plots average mood over time.

**Look for:** the two communities should clearly separate on
the line chart with r/depression below zero and r/happy above.

---

### 04_vocab
Word frequency analysis and word clouds per subreddit. Also
splits by mood label to show which words appear most in negative
vs positive posts.

**Look for:** the word clouds should look visually very different
between the two communities.

---

### 05_themes
Tags each post with five themes — support, venting, advice,
recovery, crisis — and plots a heatmap of theme intensity by
month.

**Look for:** crisis theme should appear almost exclusively in
r/depression. Recovery should appear in both.

---

### 06_tfidf
TF-IDF vectorizer reveals the statistically significant words
per community — not just frequent words but words that are
uniquely characteristic. Includes bigrams.

**Look for:** bigrams are more revealing than single words here.

---

### 07_similarity
Cosine similarity matrix comparing the two subreddits and the
three mood groups against each other.

**Look for:** r/depression and r/happy should score low
similarity — if they score high, the mock data may be too
similar.

---

### 08_classify_prep
Labels posts as positive / neutral / negative from VADER scores
and creates a stratified train/test split. Saves as a pickle
file for all classifier notebooks.

**Look for:** balanced class counts in the chart.

---

### 09_upvotes
Scatter plot of sentiment vs upvote score with a trendline.
Also shows average upvotes broken down by mood label.

**Look for:** does negative sentiment drive more engagement in
r/depression?

---

### 10_comments
Comment count distribution and average comments per mood label.
Calculates a combined engagement score.

**Look for:** which mood gets the most comments in each
community?

---

## Week 2 — ML & classifiers

### 11_classifier_sklearn
Trains Logistic Regression and Random Forest on TF-IDF vectors.
Prints classification reports and plots confusion matrices.

**Look for:** which classes does each model confuse most?

---

### 12_tensorflow
Builds and trains an Embedding → GlobalAveragePooling → Dense
neural network. Plots training and validation curves.

**Look for:** if val_loss starts rising while train_loss keeps
falling, the model is overfitting — try more dropout.

---

### 13_predict
Interactive notebook — type any text and get mood predictions
from both the sklearn and TF models with confidence scores.

**Try:** paste a real Reddit post and see if the model gets it
right.

---

### 14_lstm
Replaces the simple pooling model with a Bidirectional LSTM.
Generally more accurate on sequential text.

**Look for:** training takes longer — does it actually beat the
simpler model?

---

### 15_model_comparison
Side-by-side comparison of all three models — accuracy, F1
per class, confusion matrices.

**Look for:** which class is hardest across all models? Usually
neutral.

---

### 16_temporal_classifier
Runs the LSTM classifier on all posts grouped by month. Plots
predicted mood distribution as a stacked bar chart and overlays
it against the VADER line.

**Look for:** do VADER and the LSTM agree on which months were
most negative?

---

### 17_summary
Master dashboard — seven charts in one figure covering
sentiment, vocabulary, mood trends, model accuracy, and
engagement.

**Look for:** the full picture of what the project found.

---

### 18_misclassified
Inspects what the LSTM got wrong — most confidently wrong
predictions, error type heatmap, pattern analysis.

**Look for:** does the model confuse negative and neutral more
than negative and positive?

---

### 19_confidence
Confidence score distribution per mood class. Shows which
classes the model is most and least sure about.

**Look for:** neutral predictions should have the lowest
average confidence.