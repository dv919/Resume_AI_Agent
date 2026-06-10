import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

def clean_data(df, is_training=True):
    """
    Clean and preprocess data.
    
    Args:
        df: Input dataframe
        is_training: If True, fit and save scaler. If False, load saved scaler.
    """
    df = df.copy()
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    scaler_path = os.path.join(script_dir, 'scaler.pkl')
    
    # Fix Education Mapping (Ensure PhD is always the same number)
    if 'education_level' in df.columns:
        edu_map = {"High School": 0, "Bachelors": 1, "Masters": 2, "PhD": 3}
        df['education_level'] = df['education_level'].map(edu_map)

    # Handle shortlisted column if it exists (training data)
    if 'shortlisted' in df.columns:
        le = LabelEncoder()
        df['shortlisted'] = le.fit_transform(df['shortlisted'])

    cols_to_scale = ['years_experience', 'skills_match_score', 'project_count', 'resume_length', 'github_activity']
    existing_cols = [col for col in cols_to_scale if col in df.columns]

    if is_training:
        scaler = StandardScaler()
        df[existing_cols] = scaler.fit_transform(df[existing_cols])
        joblib.dump(scaler, scaler_path) # SAVE THE SCALER
        print("✅ Scaler saved for future predictions.")
    else:
        # LOAD THE SAVED SCALER
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            df[existing_cols] = scaler.transform(df[existing_cols])
        else:
            print("⚠️ Scaler not found! Run training first.")
    
    return df