import streamlit as st
import pickle
import numpy as np
import pandas as pd

with open('rf_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

st.set_page_config(page_title="Health Prediction App", layout="centered")

st.title("🏥 Health Risk Classification App")
st.write("Please select the values for the symptoms below to get a prediction.")

with st.form("prediction_form"):
    
    st.header("Patient Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=30)
        gender_display = st.radio("Gender", ["Male", "Female"])
        polyuria = st.radio("Polyuria (Excessive urination)", ["Yes", "No"], horizontal=True)
        polydipsia = st.radio("Polydipsia (Excessive thirst)", ["Yes", "No"], horizontal=True)
        weight_loss = st.radio("Sudden Weight Loss", ["Yes", "No"], horizontal=True)
        
    with col2:
        partial_paresis = st.radio("Partial Paresis (Weakness)", ["Yes", "No"], horizontal=True)
        irritability = st.radio("Irritability", ["Yes", "No"], horizontal=True)
        delayed_healing = st.radio("Delayed Healing", ["Yes", "No"], horizontal=True)
        alopecia = st.radio("Alopecia (Hair loss)", ["Yes", "No"], horizontal=True)
        itching = st.radio("Itching", ["Yes", "No"], horizontal=True)

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
        'Polyuria': [binary_map(polyuria)],
        'Polydipsia': [binary_map(polydipsia)],
        'Age': [age],
        'Gender': [gender_val],
        'partial paresis': [binary_map(partial_paresis)],
        'sudden weight loss': [binary_map(weight_loss)],
        'Irritability': [binary_map(irritability)],
        'delayed healing': [binary_map(delayed_healing)],
        'Alopecia': [binary_map(alopecia)],
        'Itching': [binary_map(itching)]

    })

    try:
        prediction = model.predict(input_data)[0]
        
        # Display Result
        st.write("---")
        st.subheader("Prediction Result:")
        
        if prediction == 1:
            # change the high risk to based on confidence
            st.error(f"The model predicts: **Positive**")

            # give percentage confidence if available
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_data)[0][1]
                st.write(f"Confidence: **{proba * 100:.2f}%**")

            if proba < 0.2:
                st.info("Note: The confidence level is low. It is recommended to monitor symptoms and consult a healthcare professional if they persist.")
            elif proba < 0.5:
                st.info("Note: The confidence level is moderate. Consider seeking medical advice for further evaluation.")
            else:
                st.info("Note: The confidence level is high. It is strongly recommended to consult a healthcare professional for a comprehensive diagnosis.")
            st.warning("This prediction is not definitive. Please seek professional medical advice for accurate diagnosis.")

        else:
            st.success(f"The model predicts: **Negative**")
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_data)[0][0]
                st.write(f"Confidence: **{proba * 100:.2f}%**")
            
            st.info("The model indicates a low risk based on the provided symptoms. However, if symptoms persist or worsen, please consult a healthcare professional.")
            st.warning("This prediction is not definitive. Please seek professional medical advice for accurate diagnosis.")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")