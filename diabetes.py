import streamlit as st
import numpy as np
import pickle

st.title("Diabetes Prediction System")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.text_input("Pregnancies:")
    glucose = st.text_input("Glucose:")
    blood_pressure = st.text_input("Blood Pressure:")
    skin_thickness = st.text_input("Skin Thickness:")

with col2:
    insulin = st.text_input("Insulin:")
    bmi = st.text_input("BMI:")
    dpf = st.text_input("Diabetes Pedigree Function:")
    age = st.text_input("Age:")


def convert_input(input_value):
    try:
        return float(input_value)
    except ValueError:
        return None


input_data = (
    convert_input(pregnancies),
    convert_input(glucose),
    convert_input(blood_pressure),
    convert_input(skin_thickness),
    convert_input(insulin),
    convert_input(bmi),
    convert_input(dpf),
    convert_input(age),
)

input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

if st.button("Predict"):
    try:
        loaded_model = pickle.load(open("diabetes_model.sav", "rb"))
        prediction = loaded_model.predict(input_data_reshaped)  # Prediction result

        # Display prediction result
        if prediction[0] == 0:
            st.success("The person is not diabetic")
        else:
            st.error("The person is diabetic")
    except Exception as e:
        st.error(f"Error loading the model: {e}")
