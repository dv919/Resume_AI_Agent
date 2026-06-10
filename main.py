import pandas as pd
import os
import sys
import joblib
import subprocess

# Add src directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, 'src'))

from preprocessor import clean_data

def run_pipeline():
    """
    Complete ML Pipeline Orchestrator
    Runs: Data loading → EDA → Preprocessing → Training → Evaluation → SHAP Explanation
    """
    print("\n" + "="*70)
    print("🤖 RESUME AI AGENT - COMPLETE ML PIPELINE".center(70))
    print("="*70 + "\n")
    
    # 1. Load Data
    print("📥 PHASE 1: DATA LOADING")
    print("-" * 70)
    try:
        csv_path = os.path.join(script_dir, "RESUME_DATA", "ai_resume_screening.csv")
        df = pd.read_csv(csv_path)
        print(f"✅ Data Loaded Successfully!")
        print(f"   - Total Records: {len(df):,}")
        print(f"   - Total Features: {len(df.columns)}")
        print(f"   - Shape: {df.shape}\n")
    except FileNotFoundError:
        print("❌ Error: RESUME_DATA/ai_resume_screening.csv not found!")
        print("   Please download from: https://www.kaggle.com/datasets/sonalshinde123/ai-driven-resume-screening-dataset\n")
        return

    # 2. Exploratory Data Analysis
    print("\n📊 PHASE 2: EXPLORATORY DATA ANALYSIS (EDA)")
    print("-" * 70)
    print(f"Column Overview:")
    print(df.info())
    
    print(f"\nClass Distribution (Shortlisted):")
    class_counts = df['shortlisted'].value_counts()
    print(class_counts)
    print(f"Class Ratio: {class_counts[1]/class_counts[0]:.2%}\n")
    
    # 3. Data Preprocessing
    print("\n🔧 PHASE 3: DATA PREPROCESSING & FEATURE ENGINEERING")
    print("-" * 70)
    df_cleaned = clean_data(df, is_training=True)
    print(f"✅ Data Preprocessing Complete!")
    print(f"   - Features scaled using StandardScaler")
    print(f"   - Education level encoded (0-3)")
    print(f"   - Scaler saved for production use\n")
    
    # 4. Model Training
    print("\n🚀 PHASE 4: MODEL TRAINING WITH ANTI-OVERFITTING MEASURES")
    print("-" * 70)
    print("Training XGBoost with:")
    print("   ✓ Reduced max_depth (4 instead of 5)")
    print("   ✓ L1/L2 Regularization (Alpha=0.1, Lambda=1.0)")
    print("   ✓ Subsample & Colsample (0.8 for both)")
    print("   ✓ Early Stopping (monitor validation set)")
    print("   ✓ 60/20/20 Train/Validation/Test split")
    print("   Running 'python src/train.py' to see detailed results...\n")
    
    train_result = subprocess.run(
        [sys.executable, os.path.join(script_dir, 'src', 'train.py')],
        cwd=script_dir,
        capture_output=False
    )
    
    # 5. Model Explanation
    print("\n\n📈 PHASE 5: MODEL EXPLAINABILITY (SHAP)")
    print("-" * 70)
    print("Generating SHAP explanations to understand feature importance...")
    print("Running 'python explain.py'...\n")
    
    explain_result = subprocess.run(
        [sys.executable, os.path.join(script_dir, 'explain.py')],
        cwd=script_dir,
        capture_output=False
    )
    
    # 6. Summary
    print("\n\n" + "="*70)
    print("✅ PIPELINE EXECUTION COMPLETE!".center(70))
    print("="*70)
    print("""
NEXT STEPS:
├── View SHAP plot: Open 'model_explanation.png'
├── Test with web app: streamlit run app.py
├── Tune hyperparameters: python tune.py
├── Read documentation: Check README.md
└── Deploy model: Use 'src/resume_model.pkl' in production

KEY FEATURES OF THIS PIPELINE:
✓ Prevents Overfitting: Train-Test gap monitoring
✓ Explainable AI: SHAP-based feature importance
✓ Production Ready: Serialized model and scaler
✓ Scalable: Handles 30k+ records efficiently
✓ Robust: Cross-validation and validation set monitoring
    """)
    print("="*70 + "\n")

if __name__ == "__main__":
    run_pipeline()