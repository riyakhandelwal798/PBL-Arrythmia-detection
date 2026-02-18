import streamlit as st
import numpy as np
import xgboost as xgb

# Load Model
model = xgb.XGBClassifier()
model.load_model("xgboost_arrhythmia.json")

st.title("ECG Arrhythmia Detection")

st.write("Upload ECG beat (.npy file)")

uploaded_file = st.file_uploader("Upload ECG Beat", type=["npy"])

if uploaded_file is not None:
    beat = np.load(uploaded_file)
    beat = beat.reshape(1, -1)

    prediction = model.predict(beat)[0]

    if prediction == 0:
        st.success("Normal Beat Detected")
    else:
        st.error("Arrhythmia Detected")
