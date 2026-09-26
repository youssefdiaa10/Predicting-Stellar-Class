import joblib
import pandas as pd 
import streamlit as st

model = joblib.load("predicting_stellar_class.pkl")

classes = ["GALAXY", "QSO", "STAR"]
class_names = {0: "GALAXY", 1: "QSO", 2: "STAR"}

st.set_page_config(page_title="Stellar Classifier", layout="wide")
st.title("Stellar Classifier")
st.markdown(
    """
    Classify an astronomical object as a **galaxy**, **quasar (QSO)**, or **star**.  
    Enter its sky coordinates, filter brightness measurements, redshift, spectral type,
    and galaxy population, then select **Predict** to see the predicted class and the model's
    confidence for each class.
    """
)

input_column, result_column = st.columns(2)

with input_column:
    st.header("Object measurements")
    row1 = st.columns(3)
    alpha = row1[0].number_input("Right Ascension (alpha)", help="The object's east-west position in the sky, measured at the J2000 epoch.")
    delta = row1[1].number_input("Declination (delta)", help="The object's north-south position in the sky, measured at the J2000 epoch.")
    ultraviolet = row1[2].number_input("Ultraviolet filter (u)", help="Brightness measured through the ultraviolet (u) filter.")

    row2 = st.columns(3)
    green = row2[0].number_input("Green filter (g)", help="Brightness measured through the green (g) filter.")
    red = row2[1].number_input("Red filter (r)", help="Brightness measured through the red (r) filter.")
    near_infrared = row2[2].number_input("Near-infrared filter (i)", help="Brightness measured through the near-infrared (i) filter.")

    row3 = st.columns(3)
    infrared = row3[0].number_input("Infrared filter (z)", help="Brightness measured through the infrared (z) filter.")
    redshift = row3[1].number_input("Redshift", help="How much the object's light has shifted toward longer wavelengths; this is related to its distance and motion.")
    spectral_type = row3[2].selectbox("Spectral type", options=("M", "O/B", "G/K", "A/F"), help="A broad grouping based on the object's spectrum, which reflects its temperature and properties.")

    galaxy_population = st.selectbox("Galaxy population", options=("Red_Sequence", "Blue_Cloud"), help="A broad galaxy grouping: Red Sequence galaxies are generally older, while Blue Cloud galaxies are generally younger and forming stars.")

with result_column:
    st.header("Classification")
    predict = st.button("Predict", type="primary", width="stretch")

preprocessor = joblib.load("predicting_stellar_class_preprocessor.pkl")

if predict:
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

    with result_column:
        st.success(f"Predicted stellar class: **{class_names[prediction]}**")
        #! Name each probability using the decoded class label, not its encoded index.
        chart_data = {
            class_names[int(class_index)]: probability
            for class_index, probability in zip(model.classes_, probabilities)
        }
        st.bar_chart(chart_data)
