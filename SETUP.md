# Setup Guide

Step by step from zero to running the full project.

---

## Requirements

- Python 3.9 or higher
- Git
- A Reddit account (for real data — optional)

---

## 1. Clone the repo

```cmd
git clone https://github.com/Runnp/week-02-reddit-mood-shift-nlp
cd week-02-reddit-mood-shift-nlp
```

## 2. Create and activate venv

```cmd
python -m venv venv
venv\Scripts\activate
```

## 3. Install dependencies

```cmd
pip install -r requirements.txt
```

## 4. Download NLTK data

```cmd
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon'); nltk.download('punkt_tab')"
```

## 5. Set up environment file

Create a `.env` file in the project root:


## 6. Verify setup

```cmd
python main.py
```

## 7. Generate mock data (optional — skip if using real Reddit)

```cmd
python src/mock_data.py
```

## 8. Launch Jupyter

```cmd
jupyter notebook
```

Run notebooks in order starting from `00_setup.ipynb`.

## 9. Run health check anytime

```cmd
python src/report.py
```

## 10. Launch Streamlit app

```cmd
cd app
streamlit run streamlit_app.py
```

---

## Troubleshooting

**`ModuleNotFoundError`** — make sure your venv is activated.

**Reddit API 401 error** — check your `.env` file has the correct keys with no spaces.

**Empty charts** — run `python src/mock_data.py` first to generate data.

**TF model not found** — run notebooks 12 and 14 first to train and save models.