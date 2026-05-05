## Project notebooks

| # | Notebook | Output |
|---|---|---|
| 00 | setup | environment verified |
| 01 | fetch | posts_raw.csv |
| 02 | clean | posts_clean.csv |
| 03 | sentiment | mood_over_time.png |
| 04 | vocab | wordclouds.png, top_words.png |
| 05 | themes | theme_heatmap.png |
| 06 | tfidf | tfidf_terms.png |
| 07 | similarity | similarity_heatmap.png |
| 08 | classify_prep | train_test_split.pkl |
| 09 | upvotes | sentiment_vs_upvotes.png |
| 10 | comments | engagement.png |
| 11 | classifier_sklearn | confusion_matrices.png |
| 12 | tensorflow | tf_training_curves.png |
| 13 | predict | interactive mood predictor |
| 14 | lstm | lstm_training_curves.png |
| 15 | model_comparison | all_models_accuracy.png |
| 16 | temporal_classifier | vader_vs_classifier.png |
| 17 | summary | summary_dashboard.png |

## Quick project health check

```cmd
python src/report.py
```

## src/ modules

| File | Purpose |
|---|---|
| config.py | settings and API keys |
| utils.py | load/save/summary helpers |
| preprocess.py | text cleaning and tokenization |
| analysis.py | VADER sentiment scoring |
| compare.py | cross-subreddit comparison logic |
| mock_data.py | generates offline mock posts |
| report.py | full project health check |