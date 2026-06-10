import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os
import sys

# Add src directory to path so we can import preprocessor
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, 'src'))

from preprocessor import clean_data

# 1. Load the Model and Data
print("🔍 Loading model for explanation...")
model_path = os.path.join(script_dir, 'src', 'resume_model.pkl')
model = joblib.load(model_path)
csv_path = os.path.join(script_dir, "RESUME_DATA", "ai_resume_screening.csv")
df = pd.read_csv(csv_path)
df = clean_data(df)

# Prepare Features (X)
X = df.drop('shortlisted', axis=1)

# 2. Initialize SHAP
# TreeExplainer is specifically for XGBoost/Random Forest
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# 3. Create the Visualization
print("📊 Generating SHAP Summary Plot...")
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X, show=False)

# Save the plot so you can see it
plt.savefig('model_explanation.png', bbox_inches='tight')
print("✅ Explanation saved as 'model_explanation.png'!")
plt.show()