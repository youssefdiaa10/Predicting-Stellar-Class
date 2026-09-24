import streamlit as st
import joblib
import numpy as np
import pandas as pd 

model = joblib.load("predicting_stellar_class.pkl")

classes = ["GALAXY", "QSO", "STAR"]
class_names = {
    0: "GALAXY",
    1: "QSO",
    2: "STAR", 
}

st.set_page_config(page_title="Stellar Classifier")
st.title("Stellar Classifier")
st.write("Enter astronomical figure measurements to predict the entity.")

alpha = st.number_input("Enter the value of the Right Ascension angle")
delta = st.number_input("Enter the value of the Declination angle")

ultraviolet = st.number_input("Enter the value of the Ultraviolet filter")
green = st.number_input("Enter the value of the Green filter")

red = st.number_input("Enter the value of the Red filter")
near_infrared = st.number_input("Enter the value of the near Infrared filter")

infrared = st.number_input("Enter the value of the Infrared filter")
redshift = st.number_input("Enter the value of the Red shift")

spectral_type = st.selectbox("Enter the value of the type of Spectral", options=("M", "O/B", "G/K", "A/F"))
galaxy_population = st.selectbox("Enter the value of the population of the Galaxy", options=("Red_Sequence", "Blue_Cloud"))

preprocessor = joblib.load("predicting_stellar_class_preprocessor.pkl")

if st.button("Predict"):
    raw_input = pd.DataFrame([
        {
            "alpha": alpha,
            "delta": delta,
            "u": ultraviolet,
            "g": green,
            "r": red,
            "i": near_infrared,
            "z": infrared,
            "redshift": redshift,
            "spectral_type": spectral_type,
            "galaxy_population": galaxy_population,
        }
    ])
    processed_input = preprocessor.transform(raw_input)
    prediction = model.predict(processed_input)[0]
    probabilities = model.predict_proba(processed_input)[0]

    st.success(f"Predicted stellar class: **{class_names[prediction]}**")
    st.bar_chart(dict(zip(model.classes_, probabilities)))
