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
    st.divider()

    st.subheader("Basic Information")
    col_basic_1, col_basic_2 = st.columns(2)

    with col_basic_1:
        age = st.number_input("Age", min_value=1, max_value=120, value=30)

    with col_basic_2:
        gender_display = st.radio("Gender", ["Male", "Female"], horizontal=True)

    st.divider()

    st.subheader("Key Symptoms")

    st.warning(
        "⚠️ Polyuria and polydipsia are strong indicators of diabetes. "
        "Please only select **Yes** if you are sure these symptoms are present."
    )

    col_symptom_1, col_symptom_2 = st.columns(2)

    with col_symptom_1:
        polyuria = st.radio("Polyuria (Excessive urination)", ["Yes", "No"], horizontal=True)
        polydipsia = st.radio("Polydipsia (Excessive thirst)", ["Yes", "No"], horizontal=True)
        weight_loss = st.radio("Sudden Weight Loss", ["Yes", "No"], horizontal=True)

    with col_symptom_2:
        partial_paresis = st.radio("Partial Paresis (Weakness)", ["Yes", "No"], horizontal=True)
        irritability = st.radio("Irritability", ["Yes", "No"], horizontal=True)
        delayed_healing = st.radio("Delayed Healing", ["Yes", "No"], horizontal=True)
        alopecia = st.radio("Alopecia (Hair loss)", ["Yes", "No"], horizontal=True)
        itching = st.radio("Itching", ["Yes", "No"], horizontal=True)

    st.divider()

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