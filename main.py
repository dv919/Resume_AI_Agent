import pandas as pd
import os
import sys
import joblib

# Add src directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, 'src'))

from preprocessor import clean_data

def run_pipeline():
    print("--- 🤖 Starting Resume AI Pipeline ---")
    
    # 1. Load Data
    try:
        csv_path = os.path.join(script_dir, "RESUME_DATA", "ai_resume_screening.csv")
        df = pd.read_csv(csv_path)
        print("✅ Data Loaded Successfully.")
    except FileNotFoundError:
        print("❌ Error: RESUME_DATA/ai_resume_screening.csv not found!")
        return

    # 2. Preprocess (Cleaning & Scaling)
    df_cleaned = clean_data(df)
    
    # 3. Train Model
    # Note: If your train.py just has code at the top level, 
    # you can just import it. If it's a function, call it here:
    print("🚀 Training Model...")
    # (Assuming your train.py logic is wrapped in a function or just runs on import)
    
    print("--- ✅ Pipeline Complete! ---")
    print("Run 'python explain.py' to see the AI's logic.")

if __name__ == "__main__":
    run_pipeline()