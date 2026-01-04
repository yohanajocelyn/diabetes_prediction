import streamlit as st
import pickle
import numpy as np
import pandas as pd

with open('rf_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

st.set_page_config(page_title="Early Diabetes Risk Prediction", layout="centered")

st.title("🏥 Early Diabetes Risk Classification")
st.write("Please select the values for the symptoms below to get a prediction of your risk of prediabetes.")
st.markdown(
    """
    **Dataset source:**  
    Dataset acquired from the [Early Stage Diabetes Risk Prediction Dataset](https://www.kaggle.com/datasets/ishandutta/early-stage-diabetes-risk-prediction-dataset) on Kaggle.
    """
)

with st.form("prediction_form"):
    
    st.header("Patient Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gender Feature
        gender_display = st.radio("Gender", ["Male", "Female"])

        st.warning(
            "⚠️ Polyuria and polydipsia are strong indicators of diabetes. "
            "Please only select **Yes** if you are sure these symptoms are present."
        )
        
        polyuria = st.radio("Polyuria (Excessive urination)", ["Yes", "No"], horizontal=True)
        polydipsia = st.radio("Polydipsia (Excessive thirst)", ["Yes", "No"], horizontal=True)
        weight_loss = st.radio("Sudden weight loss", ["Yes", "No"], horizontal=True)
        weakness = st.radio("Weakness", ["Yes", "No"], horizontal=True)
        polyphagia = st.radio("Polyphagia (Excessive hunger)", ["Yes", "No"], horizontal=True)
        genital_thrush = st.radio("Genital Thrush", ["Yes", "No"], horizontal=True)
        
    with col2:
        visual_blurring = st.radio("Visual Blurring", ["Yes", "No"], horizontal=True)
        itching = st.radio("Itching", ["Yes", "No"], horizontal=True)
        irritability = st.radio("Irritability", ["Yes", "No"], horizontal=True)
        delayed_healing = st.radio("Delayed Healing", ["Yes", "No"], horizontal=True)
        partial_paresis = st.radio("Partial Paresis", ["Yes", "No"], horizontal=True)
        muscle_stiffness = st.radio("Muscle Stiffness", ["Yes", "No"], horizontal=True)
        alopecia = st.radio("Alopecia (Hair loss)", ["Yes", "No"], horizontal=True)
        obesity = st.radio("Obesity", ["Yes", "No"], horizontal=True)

    submit_btn = st.form_submit_button("Get Prediction")

if submit_btn:
    # Preprocess inputs
    def binary_map(value):
        return 1 if value == "Yes" else 0
    
    # Adjust Gender mapping according to how you trained the model
    gender_val = 1 if gender_display == "Male" else 0
    
    # Create the input dataframe with the EXACT column names used in training
    # Python is sensitive to order and naming
    input_data = pd.DataFrame({
        'Gender': [gender_val],
        'Polyuria': [binary_map(polyuria)],
        'Polydipsia': [binary_map(polydipsia)],
        'sudden weight loss': [binary_map(weight_loss)],
        'weakness': [binary_map(weakness)],
        'Polyphagia': [binary_map(polyphagia)],
        'Genital thrush': [binary_map(genital_thrush)],
        'visual blurring': [binary_map(visual_blurring)],
        'Itching': [binary_map(itching)],
        'Irritability': [binary_map(irritability)],
        'delayed healing': [binary_map(delayed_healing)],
        'partial paresis': [binary_map(partial_paresis)],
        'muscle stiffness': [binary_map(muscle_stiffness)],
        'Alopecia': [binary_map(alopecia)],
        'Obesity': [binary_map(obesity)]
    })