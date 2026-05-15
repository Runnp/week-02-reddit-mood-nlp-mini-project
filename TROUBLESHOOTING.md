# 🛠️ Troubleshooting Guide - Reddit Mood Shift NLP

## Common Issues & Solutions

---

## ❌ Issue 1: "pip install os" Error

**Problem**: You get an error when trying to install `os`

**Why it happens**: `os` is a built-in Python module - it comes with Python and doesn't need to be installed.

**Solution**: Do NOT install `os`. Instead, install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Or install packages individually:
```bash
pip install streamlit pandas numpy matplotlib sklearn tensorflow
```

---

## ❌ Issue 2: "tensorflow cannot be found" Error

**Problem**: ImportError or ModuleNotFoundError for tensorflow

**Why it happens**: TensorFlow is a large package and may not be installed yet

**Solution 1 - Quick Install**:
```bash
pip install tensorflow
```

**Solution 2 - If Solution 1 fails**:
TensorFlow might require CUDA/GPU support. Install the CPU-only version:
```bash
pip install tensorflow-cpu
```

**Solution 3 - Skip TensorFlow (App still works!)**:
The app works with or without TensorFlow. If installation fails:
- ML predictions will still work with scikit-learn
- LSTM predictions will be skipped gracefully
- No errors will occur

---

## ❌ Issue 3: "No dataset loaded" in Streamlit

**Problem**: App shows warning about missing dataset

**Solution**: Generate mock data:
```bash
python src/mock_data.py
```

Or use the startup script (Windows):
```bash
run_app.bat
```

Or use the startup script (Mac/Linux):
```bash
python run_app.py
```

---

## ❌ Issue 4: "ModuleNotFoundError" for app modules

**Problem**: App can't find `style`, `loader`, `nlp_engine` modules

**Why it happens**: Working directory isn't set correctly

**Solution**: 
1. Navigate to the `app/` folder:
   ```bash
   cd app
   ```

2. Then run:
   ```bash
   streamlit run app.py
   ```

Or use the startup script which handles this automatically.

---

## ❌ Issue 5: "Port 8501 already in use"

**Problem**: Streamlit tries to run on port 8501 but it's already taken

**Solution**: Specify a different port:
```bash
streamlit run app.py --server.port=8502
```

Or kill the existing process on port 8501.

---

## ❌ Issue 6: Package installation takes very long

**Problem**: `pip install tensorflow` or others seem stuck

**Why it happens**: Some packages are large and take time to download/compile

**Solution**: Be patient and let it run. If it truly hangs (>10 min), try:
```bash
pip install --upgrade pip
pip install --no-cache-dir tensorflow
```

---

## ❌ Issue 7: Unicode/Encoding errors in terminal

**Problem**: Special characters (emojis, accents) cause errors

**Why it happens**: Terminal encoding isn't set to UTF-8

**Solution (Windows PowerShell)**:
```powershell
$env:PYTHONIOENCODING = "utf-8"
python run_app.py
```

**Solution (Windows CMD)**:
```cmd
set PYTHONIOENCODING=utf-8
python run_app.py
```

**Solution (Mac/Linux)**:
```bash
export PYTHONIOENCODING=utf-8
python run_app.py
```

---

## ✅ Verification Checklist

Run this to verify your setup:

```bash
# 1. Check Python version
python --version

# 2. Check virtual environment is active
which python  # Mac/Linux
where python  # Windows

# 3. Test imports
python -c "import streamlit, pandas, numpy, matplotlib, sklearn; print('OK')"

# 4. Test TensorFlow (optional)
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"

# 5. Check dataset
ls data/clean/posts_sentiment.csv  # Mac/Linux
dir data\clean\posts_sentiment.csv  # Windows

# 6. Launch app
cd app
streamlit run app.py
```

---

## 🚀 Quick Start Commands

### Option 1: Automated (Recommended)
```bash
# Windows
run_app.bat

# Mac/Linux
python run_app.py
```

### Option 2: Manual
```bash
# Generate data if needed
python src/mock_data.py

# Navigate to app directory
cd app

# Launch Streamlit
streamlit run app.py
```

### Option 3: Virtual Environment Manual
```bash
# Activate venv
venv\Scripts\activate.bat  # Windows
source venv/bin/activate   # Mac/Linux

# Install requirements
pip install -r requirements.txt

# Generate data
python src/mock_data.py

# Run app
cd app
streamlit run app.py
```

---

## 📋 Environment Requirements

**Minimum**:
- Python 3.8+
- pip or conda

**Required Packages**:
- streamlit
- pandas
- numpy
- matplotlib
- scikit-learn
- nltk
- seaborn

**Optional Packages** (app still works without):
- tensorflow (for LSTM predictions)

---

## 💡 Pro Tips

1. **Always use a virtual environment**:
   ```bash
   python -m venv venv
   ```

2. **Keep pip updated**:
   ```bash
   pip install --upgrade pip
   ```

3. **Cache is your friend** - Streamlit caches data loads:
   - First run: slower as data loads
   - Subsequent runs: instant (cached)
   - To clear cache: `streamlit cache clear`

4. **Debug issues**:
   ```bash
   streamlit run app.py --logger.level=debug
   ```

5. **Monitor app performance**:
   - Check terminal for warnings/errors
   - Use browser DevTools (F12) to check for frontend issues

---

## 📞 Still Having Issues?

1. **Check logs**: Look at terminal output for detailed error messages
2. **Read error messages carefully**: Python errors usually tell you exactly what's wrong
3. **Try the steps in order**: Follow the checklist above
4. **Reinstall packages**: `pip install --force-reinstall --no-cache-dir <package>`
5. **Check paths**: Ensure `data/clean/posts_sentiment.csv` exists

---

**Last Updated**: May 14, 2026
