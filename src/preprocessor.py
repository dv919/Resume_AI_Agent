import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

def clean_data(df):
    # 1. Convert Text to Numbers
    le = LabelEncoder()
    
    # Convert 'Yes/No' to 1/0
    df['shortlisted'] = le.fit_transform(df['shortlisted']) 
    
    # Convert Education (Bachelors, Masters, etc.) to 0, 1, 2...
    df['education_level'] = le.fit_transform(df['education_level'])
    
    # 2. Scaling (Professional Touch)
    # Some numbers are small (years_experience), some are large (resume_length).
    # Scaling makes them all "talk the same language" for the AI.
    scaler = StandardScaler()
    cols_to_scale = ['years_experience', 'skills_match_score', 'project_count', 'resume_length', 'github_activity']
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    
    print("✅ Preprocessing Complete: Text converted and numbers scaled.")
    return df

# Let's test it right here
if __name__ == "__main__":
    # Get the path to the CSV file relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "..", "RESUME_DATA", "ai_resume_screening.csv")
    df = pd.read_csv(csv_path)
    df_cleaned = clean_data(df)
    print(df_cleaned.head())