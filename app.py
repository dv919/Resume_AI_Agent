import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import sys

# Add src directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, 'src'))

from preprocessor import clean_data # Reusing your pro cleaning logic

# 1. Page Config
st.set_page_config(page_title="AI Resume Screener", page_icon="🤖")

# 2. Load the Model
@st.cache_resource # This makes the app fast by loading the model once
def load_model():
    model_path = os.path.join(script_dir, 'src', 'resume_model.pkl')
    return joblib.load(model_path)

model = load_model()

# 3. UI Layout
st.title("🤖 AI Resume Screening Agent")
st.markdown("Enter candidate details below to predict shortlisting probability.")

with st.form("candidate_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        exp = st.number_input("Years of Experience", min_value=0, max_value=50, value=5)
        skills = st.slider("Skills Match Score", 0, 100, 75)
        projects = st.number_input("Number of Projects", 0, 50, 3)
        
    with col2:
        edu = st.selectbox("Education Level", ["High School", "Bachelors", "Masters", "PhD"])
        res_len = st.number_input("Resume Length (Words)", 100, 5000, 500)
        git = st.number_input("GitHub Activity Score", 0, 100, 50)

    submit = st.form_submit_button("Predict Result")

# 4. Prediction Logic
if submit:
    # Create a small dataframe for the input
    input_data = pd.DataFrame([{
        'years_experience': exp,
        'skills_match_score': skills,
        'education_level': edu,
        'project_count': projects,
        'resume_length': res_len,
        'github_activity': git
    }])
    
    # Clean the input using the saved scaler (is_training=False loads the saved scaler)
    cleaned_input = clean_data(input_data, is_training=False)

    # Double check that you are dropping the 'shortlisted' column before predicting
    if 'shortlisted' in cleaned_input.columns:
        cleaned_input = cleaned_input.drop('shortlisted', axis=1)

    # Make Prediction
    prediction = model.predict(cleaned_input)[0]
    probability = model.predict_proba(cleaned_input)[0]
    
    # Show Results
    if prediction == 1:
        st.success(f"✅ **Result: SHORTLISTED** (Confidence: {probability[1]*100:.1f}%)")
        st.balloons()
    else:
        st.error(f"❌ **Result: NOT SHORTLISTED** (Confidence: {probability[0]*100:.1f}%)")