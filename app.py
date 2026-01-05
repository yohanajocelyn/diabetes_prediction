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
        polyuria = st.radio("Polyuria (Excessive urination)", ["Yes", "No"], horizontal=True, index=1)
        polydipsia = st.radio("Polydipsia (Excessive thirst)", ["Yes", "No"], horizontal=True, index=1)
        weight_loss = st.radio("Sudden Weight Loss", ["Yes", "No"], horizontal=True, index=1)

    with col_symptom_2:
        partial_paresis = st.radio("Partial Paresis (Weakness)", ["Yes", "No"], horizontal=True, index=1)
        irritability = st.radio("Irritability", ["Yes", "No"], horizontal=True, index=1)
        delayed_healing = st.radio("Delayed Healing", ["Yes", "No"], horizontal=True, index=1)
        alopecia = st.radio("Alopecia (Hair loss)", ["Yes", "No"], horizontal=True, index=1)
        itching = st.radio("Itching", ["Yes", "No"], horizontal=True, index=1)

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
            st.error(f"The model predicts: **Positive**")

            # Calculate confidence
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_data)[0][1]
                st.write(f"Confidence: **{proba * 100:.2f}%**")

                # Confidence context logic
                if proba < 0.2:
                    st.info("Note: The confidence level is low. It is recommended to monitor symptoms.")
                elif proba < 0.5:
                    st.info("Note: The confidence level is moderate. Consider seeking medical advice.")
                else:
                    st.info("Note: The confidence level is high. It is strongly recommended to consult a healthcare professional.")
            
            st.warning("This prediction is not definitive. Please seek professional medical advice.")

            # --- 🏥 ADDED: MITRA KELUARGA ADVICE SECTION ---
            st.write("") # Spacer
            with st.expander("📋 Recommended Next Steps (Based on Mitra Keluarga Guidelines)", expanded=True):
                st.markdown("""
                **1. 🩺 Consult a Doctor (Priority)**
                * Do not self-diagnose based on this app. Visit an Internist or Endocrinologist for lab tests (HbA1c, Fasting Blood Sugar).
                
                **2. 🥗 Adjust Your Diet**
                * **Swap:** White rice for brown rice, oatmeal, or whole wheat.
                * **Cook:** Boil, steam, or grill instead of frying.
                * **Limit:** Sugar, salt, and saturated fats.
                
                **3. 🏃‍♂️ Stay Active**
                * Aim for **10-30 mins** of aerobic exercise daily (walking, cycling).
                * *Safety:* Do not exercise if blood sugar >200 mg/dl.
                
                **4. 📉 Monitor & Manage**
                * Check blood sugar regularly.
                * Manage stress levels to prevent hormonal sugar spikes.
                """)
            # -----------------------------------------------

        else:
            st.success(f"The model predicts: **Negative**")
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_data)[0][0]
                st.write(f"Confidence: **{proba * 100:.2f}%**")
            
            st.info("The model indicates a low risk based on the provided symptoms. However, if symptoms persist, please consult a professional.")
            st.warning("This prediction is not definitive. Please seek professional medical advice.")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")