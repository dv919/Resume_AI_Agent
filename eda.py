import pandas as pd

# Load the 30k dataset
df = pd.read_csv("RESUME_DATA/ai_resume_screening.csv") 

print("--- Column Overview ---")
print(df.info())
print("\n--- Sample Decisions ---")
print(df['shortlisted'].value_counts()) # Shows how many Yes vs No