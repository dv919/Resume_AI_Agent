import pandas as pd
import joblib
import os
import sys
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split

# Add src directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, 'src'))

from preprocessor import clean_data

# 1. Load and Prep Data
print("🔄 Preparing data for tuning...")
csv_path = os.path.join(script_dir, "RESUME_DATA", "ai_resume_screening.csv")
df = pd.read_csv(csv_path)
df = clean_data(df, is_training=True)

X = df.drop('shortlisted', axis=1)
y = df['shortlisted']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Define the "Hyperparameter Grid"
# We are giving the AI a range of values to test
param_dist = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'scale_pos_weight': [0.43] # Keeping our class balance fix!
}

# 3. Initialize RandomizedSearch
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

print("🧪 Starting Hyperparameter Search (Running 20 variations)...")
random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=20,           # Try 20 random combinations
    scoring='accuracy',
    cv=3,                # 3-fold Cross Validation
    verbose=1,
    random_state=42,
    n_jobs=-1            # Use all your CPU cores to make it fast
)

# 4. Run the Search
random_search.fit(X_train, y_train)

# 5. Results
print("\n🏆 BEST PARAMETERS FOUND:")
print(random_search.best_params_)

print(f"\n📈 Best Cross-Validation Accuracy: {random_search.best_score_*100:.2f}%")

# 6. Evaluate on Test Set to detect overfitting
from sklearn.metrics import accuracy_score
best_model = random_search.best_estimator_
y_test_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"\n🧪 Test Set Accuracy: {test_accuracy*100:.2f}%")

# Check for overfitting
train_test_gap = random_search.best_score_ - test_accuracy
print(f"\n🔍 Overfitting Check:")
print(f"   Cross-Val Accuracy: {random_search.best_score_*100:.2f}%")
print(f"   Test Accuracy: {test_accuracy*100:.2f}%")
print(f"   Gap: {train_test_gap*100:.2f}%")

if train_test_gap > 0.05:
    print("   ⚠️  WARNING: Potential overfitting detected (gap > 5%)")
    print("   Consider: reducing max_depth, increasing regularization, or using more data")
else:
    print("   ✅ Good generalization!")

# 6. Save the PERFECTED model
model_path = os.path.join(script_dir, 'src', 'resume_model.pkl')
joblib.dump(best_model, model_path)
print("\n✅ Best model saved as 'resume_model.pkl'")