# Streamlit App

Interactive dashboard for the Reddit Mood Shift NLP project.

## Pages

| Page | What it shows |
|---|---|
| Overview | Dataset metrics, sentiment scores, mood distribution |
| Mood over time | Monthly sentiment line chart with filters |
| Word explorer | Top words and word clouds per subreddit |
| Live predictor | Type any text, get mood predictions from sklearn and LSTM |

## How to run

From the project root:

```cmd
venv\Scripts\activate
cd app
streamlit run streamlit_app.py
```

Opens at http://localhost:8501

## Requirements

Make sure you have data first:

```cmd
python src/mock_data.py        # generate mock data
jupyter notebook               # run notebooks 02-14 to build models
```

Then launch the app.

## Notes

- The live predictor needs trained models from notebooks 11 and 14
- If models are missing the predictor shows a warning but the rest of the app still works
- All charts use the same color scheme as the notebooks