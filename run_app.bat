@echo off
REM Reddit Mood Shift NLP - Streamlit App Launcher
REM This script activates the virtual environment and runs the Streamlit app

echo.
echo =========================================
echo  Reddit Mood Shift NLP - Streamlit App
echo =========================================
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if data exists
if not exist "data\clean\posts_sentiment.csv" (
    echo.
    echo [WARNING] No dataset found!
    echo.
    echo Generating mock data...
    python src\mock_data.py
    echo.
)

REM Launch Streamlit app
echo.
echo Launching Streamlit app at http://localhost:8501
echo.
cd app
streamlit run app.py

pause
