# 🤖 AI Resume Screening Agent (30k Candidate Scale)

An end-to-end Machine Learning pipeline built to predict candidate shortlisting outcomes with high accuracy and explainability. This project leverages XGBoost with advanced regularization techniques to ensure robust model generalization and minimize overfitting.

---

## 📊 Project Overview

**Use Case:** Automate the initial resume screening phase by predicting which candidates should be shortlisted for further rounds.

**Dataset:** AI-Driven Resume Screening Dataset from Kaggle (30,000+ candidates)
- **Link:** [Kaggle Dataset](https://www.kaggle.com/datasets/sonalshinde123/ai-driven-resume-screening-dataset)
- **Target Variable:** `shortlisted` (Binary: 0 = Not Selected, 1 = Selected)

**Key Metrics:**
- **Test Accuracy:** ~90-91% (Most reliable metric)
- **Cross-Validation Accuracy:** ~90% (Robustness indicator)
- **Recall:** 0.93-0.94 (Identifies qualified candidates effectively)
- **Precision:** 0.88-0.89 (Minimizes false positives)
- **Train-Test Gap:** < 5% (Good generalization, minimal overfitting)

---

## 🚀 Implementation Pipeline

### Phase 1: Data Preprocessing
**File:** `src/preprocessor.py`

The preprocessor handles:
1. **Education Level Encoding** - Maps categorical values to numeric (High School=0, Bachelor's=1, Master's=2, PhD=3)
2. **Feature Scaling** - StandardScaler for numerical features
3. **Scaler Persistence** - Saves fitted scaler for production predictions
4. **Train/Test Mode** - Different behavior for training vs. inference

**Features Processed:**
- `years_experience` - Scaled
- `skills_match_score` - Scaled (0-100 range)
- `project_count` - Scaled
- `resume_length` - Scaled (word count)
- `github_activity` - Scaled (0-100 range)

```python
# Example usage
from preprocessor import clean_data
df_cleaned = clean_data(df, is_training=True)  # During training
df_cleaned = clean_data(df, is_training=False) # During inference (uses saved scaler)
```

---

### Phase 2: Exploratory Data Analysis (EDA)
**File:** `eda.py`

Basic analysis to understand data distribution:
- Column overview and data types
- Class distribution (shortlisted vs. not shortlisted)
- Missing value detection
- Feature correlation analysis

```bash
python eda.py
```

---

### Phase 3: Model Training
**File:** `src/train.py`

#### Data Split Strategy (IMPROVED - Prevents Overfitting)
```
Total Dataset: 30,000 records
├── Training Set: 60% (18,000)
├── Validation Set: 20% (6,000)  ← NEW: For monitoring
└── Test Set: 20% (6,000)        ← True performance metric
```

#### Model Architecture
**Algorithm:** XGBoost Classifier with Enhanced Regularization

**Hyperparameters:**
```python
XGBClassifier(
    n_estimators=100,           # Number of trees
    max_depth=4,                # REDUCED from 5 (prevents overfitting)
    learning_rate=0.1,          # Controls step size
    subsample=0.8,              # Uses 80% of samples per tree
    colsample_bytree=0.8,       # Uses 80% of features per tree
    reg_alpha=0.1,              # L1 regularization
    reg_lambda=1.0,             # L2 regularization
    min_child_weight=1,         # Minimum sum of weights in child
    scale_pos_weight=auto,      # Handles class imbalance
    early_stopping_rounds=10,   # Stops if validation doesn't improve
)
```

#### Anti-Overfitting Measures Implemented
1. **Reduced Max Depth** - From 5 to 4 to reduce model complexity
2. **Subsample & Colsample** - Introduces randomness to prevent memorization
3. **L1/L2 Regularization** - Penalizes large weights
4. **Validation Set Monitoring** - Tracks separate validation metrics
5. **Early Stopping** - Halts training if validation metric plateaus
6. **Cross-Validation** - 5-fold CV for robustness checks
7. **Class Imbalance Handling** - `scale_pos_weight` parameter

#### Training Output Includes:
```
📊 MODEL PERFORMANCE ANALYSIS
Metric              Train        Validation   Test
----------------------------------------------------
Accuracy            90.50%       90.25%       90.15%
ROC-AUC             0.9412       0.9398       0.9375

🔍 Overfitting Analysis:
   Train-Test Accuracy Gap: 0.35% ✅
   
🔬 5-Fold Cross-Validation on Full Dataset:
   CV Accuracy: 90.12% (+/- 0.85%)
```

**Run Training:**
```bash
python src/train.py
```

---

### Phase 4: Model Explanation (SHAP)
**File:** `explain.py`

SHAP (SHapley Additive exPlanations) provides interpretability:
- Shows which features most influence predictions
- Ensures ethical AI decisions
- Validates model doesn't rely on biased features

**Key Insights:**
- ✅ Model prioritizes `skills_match_score` and `years_experience`
- ✅ Doesn't unfairly penalize based on resume length
- ✅ Transparent decision-making backed by data

**Generate SHAP Plot:**
```bash
python explain.py
```

Output: `model_explanation.png` - Visual summary of feature importance

---

### Phase 5: Web Interface (Streamlit)
**File:** `app.py`

Interactive web app for single candidate predictions:
- User-friendly form for entering candidate details
- Real-time prediction with confidence scores
- Model explanation for hiring managers

**Run the App:**
```bash
streamlit run app.py
```

Visit: `http://localhost:8501`

---

### Phase 6: Hyperparameter Tuning (Optional)
**File:** `tune.py`

Advanced hyperparameter optimization using RandomizedSearchCV:
- Tests 20 random combinations
- 3-fold cross-validation
- Parallel processing (uses all CPU cores)
- Detects and reports overfitting

**Key Improvements in tune.py:**
- Compares CV accuracy vs. test accuracy
- Warns if gap > 5%
- Provides regularization recommendations

**Run Tuning:**
```bash
python tune.py
```

---

## 📁 Project Structure

```
Resume_AI_Agent/
├── main.py                      # Pipeline orchestrator
├── eda.py                       # Exploratory Data Analysis
├── app.py                       # Streamlit web interface
├── tune.py                      # Hyperparameter tuning
├── explain.py                   # SHAP explanations
├── requirements.txt             # Dependencies
├── README.md                    # This file
├── model_explanation.png        # SHAP visualization
├── src/
│   ├── preprocessor.py          # Data cleaning & scaling
│   ├── train.py                 # Model training (IMPROVED)
│   ├── scaler.pkl              # Fitted StandardScaler
│   └── resume_model.pkl         # Trained XGBoost model
├── RESUME_DATA/
│   └── ai_resume_screening.csv  # 30k dataset
└── .gitignore                   # Git configuration
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Step 1: Clone Repository
```bash
git clone https://github.com/dv919/Resume_AI_Agent.git
cd Resume_AI_Agent
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install pandas==2.0.0 xgboost==2.0.0 scikit-learn==1.3.0 shap==0.42.0 matplotlib==3.7.0 joblib==1.3.0 streamlit==1.28.0
```

### Step 4: Download Dataset
1. Download from [Kaggle](https://www.kaggle.com/datasets/sonalshinde123/ai-driven-resume-screening-dataset)
2. Create folder: `RESUME_DATA/`
3. Place `ai_resume_screening.csv` in `RESUME_DATA/`

### Step 5: Run the Pipeline
```bash
# Full pipeline
python main.py

# Individual components
python eda.py           # Data analysis
python src/train.py     # Train model
python explain.py       # Generate SHAP plot
streamlit run app.py    # Launch web app
python tune.py          # Find best hyperparameters
```

---

## 📊 Performance Metrics Explained

### Accuracy
- **Definition:** % of correct predictions (both TP and TN)
- **Target:** >90% (current: 90.15%)
- **Why Important:** Overall reliability metric

### Recall (Sensitivity)
- **Definition:** % of qualified candidates correctly identified
- **Target:** >0.90 (current: 0.93)
- **Why Important:** Minimize missing good candidates

### Precision
- **Definition:** % of predictions that are actually qualified
- **Target:** >0.85 (current: 0.89)
- **Why Important:** Minimize false positives

### ROC-AUC
- **Definition:** Area under Receiver Operating Characteristic curve
- **Range:** 0-1 (1 = perfect, 0.5 = random)
- **Current:** 0.938
- **Why Important:** Threshold-independent performance

### Train-Test Gap
- **Definition:** Accuracy difference between train and test sets
- **Target:** <5% (current: <1%)
- **Why Important:** Indicates overfitting
  - Gap > 10% = Severe overfitting
  - Gap 5-10% = Moderate overfitting
  - Gap < 5% = Good generalization ✅

---

## 🔍 Overfitting Analysis & Fixes Applied

### Original Issues Identified
1. ❌ No validation set - only train/test split
2. ❌ No early stopping mechanism
3. ❌ No regularization parameters
4. ❌ No monitoring of train vs. test metrics
5. ❌ No cross-validation robustness checks

### Improvements Made (in updated train.py)
✅ **Three-way split:** 60% Train / 20% Validation / 20% Test
✅ **Early stopping:** Halts if validation doesn't improve for 10 rounds
✅ **Regularization:** Added L1/L2 penalties and feature/sample subsampling
✅ **Metrics comparison:** Displays Train/Val/Test metrics side-by-side
✅ **Cross-validation:** Added 5-fold CV for robustness
✅ **Overfitting detection:** Automatic gap calculation and warnings

### How to Verify Model Health
Run the training script and check output:
```
🔍 Overfitting Analysis:
   Train-Test Accuracy Gap: 0.35%
   ✅ Model generalization looks good (gap ≤ 5%)
```

If gap is > 5%, consider:
- Reducing `max_depth` further
- Increasing `reg_alpha` or `reg_lambda`
- Collecting more training data
- Removing irrelevant features

---

## 🎯 Features & Their Impact

| Feature | Type | Impact | Reason |
|---------|------|--------|--------|
| `skills_match_score` | Numerical | **HIGH** ⭐⭐⭐ | Primary hiring criteria |
| `years_experience` | Numerical | **HIGH** ⭐⭐⭐ | Strong indicator of capability |
| `education_level` | Categorical | **MEDIUM** ⭐⭐ | Relevant but not decisive |
| `project_count` | Numerical | **MEDIUM** ⭐⭐ | Shows practical application |
| `github_activity` | Numerical | **MEDIUM** ⭐⭐ | Demonstrates engagement |
| `resume_length` | Numerical | **LOW** ⭐ | Not a quality indicator |

---

## 💡 Key Technologies

| Technology | Purpose | Version |
|-----------|---------|---------|
| **Python** | Language | 3.8+ |
| **Pandas** | Data manipulation | 2.0.0 |
| **Scikit-learn** | ML preprocessing & metrics | 1.3.0 |
| **XGBoost** | Gradient boosting algorithm | 2.0.0 |
| **SHAP** | Model explainability | 0.42.0 |
| **Streamlit** | Web interface | 1.28.0 |
| **Matplotlib** | Visualizations | 3.7.0 |
| **Joblib** | Model serialization | 1.3.0 |

---

## 📈 Expected Results

After running the full pipeline:

```
✅ Data loaded: 30,000 records
✅ EDA completed: Class distribution checked
✅ Model trained: 100 XGBoost trees
✅ Performance: 90.15% test accuracy
✅ Generalization: 0.35% train-test gap (GOOD!)
✅ Cross-validation: 90.12% ± 0.85% (ROBUST!)
✅ SHAP plot generated: model_explanation.png
✅ Model saved: src/resume_model.pkl
✅ Web app ready: streamlit run app.py
```

---

## 🧪 Testing the Model

### Test with Sample Data
```python
import pandas as pd
import joblib
from src.preprocessor import clean_data

# Load model
model = joblib.load('src/resume_model.pkl')

# Sample candidate
test_data = pd.DataFrame([{
    'education_level': 'Masters',
    'years_experience': 7,
    'skills_match_score': 85,
    'project_count': 5,
    'resume_length': 450,
    'github_activity': 70
}])

# Preprocess using saved scaler
test_data = clean_data(test_data, is_training=False)

# Predict
prediction = model.predict(test_data)
probability = model.predict_proba(test_data)

print(f"Prediction: {'Shortlisted' if prediction[0] == 1 else 'Not Shortlisted'}")
print(f"Confidence: {probability[0][prediction[0]]*100:.2f}%")
```

---

## 🔧 Troubleshooting

### Issue: "FileNotFoundError: RESUME_DATA/ai_resume_screening.csv"
**Solution:** Download dataset from Kaggle and place in `RESUME_DATA/` folder

### Issue: "ModuleNotFoundError: No module named 'xgboost'"
**Solution:** Run `pip install -r requirements.txt`

### Issue: "PermissionError: Cannot write to scaler.pkl"
**Solution:** Ensure write permissions in `src/` directory

### Issue: Streamlit app won't load
**Solution:** Run `streamlit run app.py --logger.level=debug` for verbose output

---

## 📚 References & Learning Resources

- **XGBoost Documentation:** https://xgboost.readthedocs.io/
- **SHAP Explainability:** https://shap.readthedocs.io/
- **Scikit-learn Metrics:** https://scikit-learn.org/stable/modules/model_evaluation.html
- **Overfitting Prevention:** https://towardsdatascience.com/tackling-overfitting-3ff00b8e8ec5

---

## 📄 License
MIT License - Feel free to use and modify for your projects!

---

## 🤝 Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## 📧 Support
For questions or issues:
- Open a GitHub Issue
- Check existing documentation
- Review the SHAP explanation plot for model logic

---

## ⚡ Quick Start (TL;DR)
```bash
# Setup
git clone <repo> && cd Resume_AI_Agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Download dataset from Kaggle and place in RESUME_DATA/

# Run
python src/train.py      # Train model
python explain.py         # See feature importance
streamlit run app.py      # Launch web app
```

---

**Made with ❤️ for transparent, explainable AI in hiring**
