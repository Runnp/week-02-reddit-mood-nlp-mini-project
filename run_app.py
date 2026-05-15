#!/usr/bin/env python
"""
Reddit Mood Shift NLP - Streamlit App Launcher
Handles environment setup and launches the Streamlit app
"""
import os
import sys
import subprocess

def main():
    print("\n" + "="*50)
    print("  Reddit Mood Shift NLP - Streamlit App")
    print("="*50 + "\n")
    
    # Get project root
    root = os.path.dirname(os.path.abspath(__file__))
    app_dir = os.path.join(root, "app")
    data_dir = os.path.join(root, "data", "clean")
    dataset_file = os.path.join(data_dir, "posts_sentiment.csv")
    
    # Check for dataset
    if not os.path.exists(dataset_file):
        print("[WARNING] No dataset found!")
        print(f"Expected: {dataset_file}\n")
        
        response = input("Generate mock data now? (y/n): ").strip().lower()
        if response == 'y':
            print("\nGenerating mock data...")
            try:
                subprocess.run([sys.executable, "src/mock_data.py"], cwd=root, check=True)
                print("Mock data generated successfully!\n")
            except subprocess.CalledProcessError:
                print("[ERROR] Failed to generate mock data\n")
                return 1
        else:
            print("\n[INFO] You can generate data later with:")
            print("  python src/mock_data.py")
            print("\nProceeding without data...\n")
    else:
        print(f"[OK] Dataset found: {dataset_file}\n")
    
    # Launch Streamlit
    print("Launching Streamlit app...")
    print("Opening http://localhost:8501\n")
    
    try:
        os.chdir(app_dir)
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=False)
        return 0
    except KeyboardInterrupt:
        print("\n\nApp stopped by user")
        return 0
    except Exception as e:
        print(f"\n[ERROR] Failed to launch app: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
