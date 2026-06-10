import pandas as pd
import joblib  # To save the model
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
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

# Check class distribution for imbalance
class_distribution = y.value_counts()
print(f"\n📊 Class Distribution: \n{class_distribution}")
class_ratio = class_distribution[1] / class_distribution[0]
print(f"Class Ratio (positive/negative): {class_ratio:.3f}")

# 3. Split data (60% Train, 20% Validation, 20% Test) - IMPROVED to prevent overfitting
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

print(f"\n✅ Data Split: Train={len(X_train)} | Validation={len(X_val)} | Test={len(X_test)}")

# 4. Train the XGBoost Model with regularization to prevent overfitting
print("\n🚀 Training the XGBoost Model with Regularization...")
model = XGBClassifier(
    n_estimators=100, 
    max_depth=4,  # REDUCED from 5 to 4 to reduce model complexity
    learning_rate=0.1,
    subsample=0.8,  # ADDED: Use 80% of samples per tree
    colsample_bytree=0.8,  # ADDED: Use 80% of features per tree
    reg_alpha=0.1,  # ADDED: L1 regularization to prevent overfitting
    reg_lambda=1.0,  # ADDED: L2 regularization to prevent overfitting
    min_child_weight=1,  # ADDED: Minimum sum of weights in child nodes
    scale_pos_weight=1/class_ratio if class_ratio < 0.5 else 1,  # Handle class imbalance
    use_label_encoder=False, 
    eval_metric='logloss',
    early_stopping_rounds=10,  # ADDED: Stop if validation doesn't improve
    random_state=42
)

# Train with early stopping using validation set
eval_set = [(X_val, y_val)]
model.fit(
    X_train, y_train,
    eval_set=eval_set,
    verbose=False
)

# 5. Evaluate Performance on All Sets
print("\n--- 📊 MODEL PERFORMANCE ANALYSIS ---\n")

# Training Metrics
y_train_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])

# Validation Metrics
y_val_pred = model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

# Test Metrics
y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

print(f"{'Metric':<20} {'Train':<12} {'Validation':<12} {'Test':<12}")
print("-" * 56)
print(f"{'Accuracy':<20} {train_accuracy*100:>10.2f}% {val_accuracy*100:>10.2f}% {test_accuracy*100:>10.2f}%")
print(f"{'ROC-AUC':<20} {train_auc:>10.4f}   {val_auc:>10.4f}   {test_auc:>10.4f}")

# Overfitting Detection
train_test_gap = train_accuracy - test_accuracy
print(f"\n🔍 Overfitting Analysis:")
print(f"   Train-Test Accuracy Gap: {train_test_gap*100:>6.2f}%")

if train_test_gap > 0.05:
    print("   ⚠️  POTENTIAL OVERFITTING DETECTED (gap > 5%)")
else:
    print("   ✅ Model generalization looks good (gap ≤ 5%)")

print(f"\n📊 Test Set Detailed Report:")
print(classification_report(y_test, y_test_pred, target_names=['Not Shortlisted', 'Shortlisted']))

print(f"\n📈 Confusion Matrix (Test Set):")
cm = confusion_matrix(y_test, y_test_pred)
print(f"   [[{cm[0,0]:>5} {cm[0,1]:>5}]  (TN, FP)")
print(f"    [{cm[1,0]:>5} {cm[1,1]:>5}]] (FN, TP)")

# 6. Cross-Validation Score (ADDITIONAL ROBUSTNESS CHECK)
print(f"\n🔬 5-Fold Cross-Validation on Full Dataset:")
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"   CV Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")
print(f"   Fold Scores: {[f'{score*100:.2f}%' for score in cv_scores]}")

# 7. Save the Model (The "Production" way)
model_path = os.path.join(script_dir, 'resume_model.pkl')
joblib.dump(model, model_path)
print(f"\n✅ Model saved as 'resume_model.pkl'")
print(f"\n💡 TIP: Test Accuracy ({test_accuracy*100:.2f}%) is the best indicator of real-world performance!")