import pandas as pd
import joblib  # To save the model
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os
from preprocessor import clean_data 

# 1. Load and Clean
print("🔄 Loading and cleaning 30k rows...")
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "..", "RESUME_DATA", "ai_resume_screening.csv")
df = pd.read_csv(csv_path)
df = clean_data(df)

# 2. Features (X) and Target (y)
X = df.drop('shortlisted', axis=1)
y = df['shortlisted']

# 3. Split data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the XGBoost Model
print("🚀 Training the XGBoost Model (this may take a few seconds)...")
model = XGBClassifier(
    n_estimators=100, 
    max_depth=5, 
    learning_rate=0.1, 
    use_label_encoder=False, 
    eval_metric='logloss'
)
model.fit(X_train, y_train)

# 5. Evaluate Performance
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n--- 📊 MODEL PERFORMANCE ---")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred))

# 6. Save the Model (The "Production" way)
model_path = os.path.join(script_dir, 'resume_model.pkl')
joblib.dump(model, model_path)
print("\n✅ Model saved as 'resume_model.pkl'")