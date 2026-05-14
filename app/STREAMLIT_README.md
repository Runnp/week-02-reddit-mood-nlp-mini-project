# 🧠 Reddit Mood Shift NLP - Streamlit App

A comprehensive interactive dashboard for exploring NLP analysis of emotional language across r/depression and r/happy.

## 🚀 Quick Start

### 1. Generate Data (if you don't have it yet)

```bash
# Generate offline mock data
python src/mock_data.py

# OR fetch real Reddit data using Jupyter
jupyter notebook
# Run notebooks/01_fetch.ipynb and 02_clean.ipynb
```

### 2. Launch the Streamlit App

```bash
cd app
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📍 App Navigation

### 🏠 **Home**
- Overview of the study
- Quick statistics dashboard
- Subreddit comparison charts
- Getting started guide

### 📊 **Dataset Story**
- Complete dataset overview
- Subreddit distribution (pie & bar charts)
- Mood distribution breakdown
- Raw data preview table
- Key statistics and metrics

### 📈 **Emotion Trends**
- Monthly sentiment time-series
- Emotional tone shifts over time
- Volatility analysis (standard deviation)
- Sentiment gap between communities
- Interactive month filtering

### 🔤 **Word Insights**
- Top word frequency analysis
- TF-IDF signature words
- Vocabulary comparison
- Word cloud visualizations
- (Links to detailed notebooks)

### 🧠 **ML Predictor**
- Live mood prediction interface
- Real-time text analysis
- Multi-model predictions:
  - Sklearn (LR/RF)
  - LSTM neural network
  - Ensemble agreement
- Confidence scores and breakdowns
- Emotional probability distribution

### ⚖️ **Compare Texts**
- Side-by-side text comparison
- Preset comparison examples
- Emotional profile analysis
- VADER sentiment comparison
- Emotion vector comparison

---

## 📊 Features

### Data Visualizations
- **Bar Charts**: Sentiment and distribution comparisons
- **Line Charts**: Temporal trends with dual subreddit overlays
- **Pie Charts**: Category distributions with percentages
- **Heatmaps**: Correlation matrices
- **Radar Charts**: Multi-dimensional emotional profiles

### Interactive Elements
- Sidebar navigation with page selection
- Multi-select filters for subreddits
- Date range sliders for temporal filtering
- Text input areas with real-time analysis
- Collapsible sections for detailed views

### ML Integration
- Pre-trained sklearn models (Logistic Regression, Random Forest)
- Bidirectional LSTM neural network
- TF-IDF vectorization
- Automatic model selection based on availability
- Confidence scoring and ensemble predictions

### Color Scheme
- **r/depression**: 🔴 `#E07070` (red)
- **r/happy**: 🟢 `#7BC67E` (green)
- **Positive mood**: `#7BC67E`
- **Neutral mood**: `#B0B0B0` (gray)
- **Negative mood**: `#E07070`

---

## 🛠️ Project Structure

```
app/
├── app.py                    # Main Streamlit app entry point
├── style.py                  # Colors, badges, and UI components
├── loader.py                 # Data and model loading utilities
├── nlp_engine.py             # ML model prediction logic
├── visualizations.py         # Advanced plotting functions
├── pages/                    # Streamlit multipage app pages
│   ├── 01_dataset_story.py
│   ├── 02_emotion_trends.py
│   └── 07_compare_texts.py
└── __pycache__/
```

---

## 📦 Requirements

Ensure these are installed (check `requirements.txt`):

```
streamlit>=1.28.0
pandas>=1.5.0
matplotlib>=3.6.0
numpy>=1.23.0
scikit-learn>=1.2.0
nltk>=3.8.1
vaderSentiment>=3.3.2
tensorflow>=2.11.0
seaborn>=0.12.0
python-dotenv>=0.21.0
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 🔍 Data Requirements

The app expects this file structure:

```
data/
└── clean/
    ├── posts_sentiment.csv           # Main dataset (required)
    ├── best_sklearn_model.pkl        # Sklearn model (optional)
    ├── lstm_mood_model/              # LSTM model dir (optional)
    ├── tf_tokenizer.pkl              # TF tokenizer (optional)
    └── train_test_split.pkl          # Train/test split (optional)
```

**Minimum required CSV columns:**
- `subreddit`: "depression" or "happy"
- `text`: Post text content
- `vader_compound`: VADER sentiment score (-1 to 1)
- `mood_label`: "positive", "neutral", or "negative"
- `month`: Date as "YYYY-MM" format

Generate mock data with compatible structure:
```bash
python src/mock_data.py
```

---

## 🎯 Usage Examples

### Example 1: Explore Dataset
1. Launch app: `streamlit run app.py`
2. Navigate to **📊 Dataset Story**
3. View total posts, distribution, and raw data

### Example 2: Analyze Trends
1. Go to **📈 Emotion Trends**
2. Filter by date range using slider
3. Compare sentiment across months
4. Check volatility metrics

### Example 3: Predict Mood
1. Navigate to **🧠 ML Predictor**
2. Paste or type text
3. Click "🚀 Predict"
4. View model predictions and confidence scores

### Example 4: Compare Posts
1. Go to **⚖️ Compare Texts**
2. Load a preset or enter custom text
3. Click "Compare"
4. View side-by-side analysis

---

## ⚙️ Configuration

Edit `style.py` to customize:
- Colors and emoji mappings
- UI component styling
- Badges and metric displays

Edit `app.py` to modify:
- Page names and order
- Default chart parameters
- Data filtering options

---

## 🐛 Troubleshooting

### "No dataset loaded"
```bash
# Generate mock data first
python src/mock_data.py
```

### "Model not found"
- Optional. App works without pre-trained models
- Prediction page shows placeholder predictions
- Train models with notebooks/11_classifier_sklearn.ipynb

### "Module import errors"
```bash
# Verify paths from app directory
cd app
streamlit run app.py
```

### "Port 8501 already in use"
```bash
streamlit run app.py --logger.level=debug --server.port=8502
```

---

## 📚 Related Documents

- **Main README**: `../README.md`
- **Findings**: `../FINDINGS.md`
- **Setup Guide**: `../SETUP.md`
- **Notebooks**: `../notebooks/` (30+ analysis notebooks)
- **Source Code**: `../src/` (core NLP utilities)

---

## 🤝 Contributing

To enhance the app:
1. Modify `app.py` for new pages or features
2. Add visualizations in `visualizations.py`
3. Update `style.py` for UI changes
4. Test with `streamlit run app.py`

---

## 📝 License

Same as parent project. See `../LICENSE` for details.

---

**Last Updated**: May 14, 2026  
**Status**: ✅ Production Ready
