import streamlit as st
import pandas as pd
import pickle
import warnings

# Suppress Streamlit warnings when running in bare mode
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")

# -------------------------------
# Load models and scalers
# -------------------------------
rice_model = pickle.load(open("rice_best_model.sav", "rb"))
wheat_model = pickle.load(open("wheat_best_model.sav", "rb"))
maize_model = pickle.load(open("maize_best_model.sav", "rb"))

rice_scaler = pickle.load(open("rice_scaler.sav", "rb"))
wheat_scaler = pickle.load(open("wheat_scaler.sav", "rb"))
maize_scaler = pickle.load(open("maize_scaler.sav", "rb"))

# -------------------------------
# App Title
# -------------------------------
st.title("🌱 Agriculture Pest Infestation Risk Prediction")
st.write("Select crop first, then enter environmental conditions")

# -------------------------------
# Crop Selection at TOP
# -------------------------------
st.subheader("🌾 Select Crop Type")

if "selected_crop" not in st.session_state:
    st.session_state.selected_crop = None

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🌾 Rice"):
        st.session_state.selected_crop = "Rice"

with col2:
    if st.button("🌾 Wheat"):
        st.session_state.selected_crop = "Wheat"

with col3:
    if st.button("🌾 Maize"):
        st.session_state.selected_crop = "Maize"

# -------------------------------
# Show inputs only after crop selection
# -------------------------------
if st.session_state.selected_crop:

    st.subheader(f"Enter Conditions for {st.session_state.selected_crop}")

    temperature = st.number_input("Temperature (°C)", 10, 45)
    humidity = st.number_input("Humidity (%)", 30, 100)
    rainfall = st.number_input("Rainfall (mm)", 0, 300)
    soil_moisture = st.number_input("Soil Moisture", 10, 100)
    wind_speed = st.number_input("Wind Speed (km/h)", 0, 40)
    soil_ph = st.number_input("Soil pH", 4.5, 9.0)
    crop_age = st.number_input("Crop Age (days)", 1, 150)

    input_df = pd.DataFrame([{
        "Temperature": temperature,
        "Humidity": humidity,
        "Rainfall": rainfall,
        "Soil_Moisture": soil_moisture,
        "Wind_Speed": wind_speed,
        "Soil_pH": soil_ph,
        "Crop_Age": crop_age
    }])

    if st.button("Predict Pest Risk"):

        if st.session_state.selected_crop == "Rice":
            scaled = rice_scaler.transform(input_df)
            prediction = rice_model.predict(scaled)

        elif st.session_state.selected_crop == "Wheat":
            scaled = wheat_scaler.transform(input_df)
            prediction = wheat_model.predict(scaled)

        else:
            scaled = maize_scaler.transform(input_df)
            prediction = maize_model.predict(scaled)

        if prediction[0] == 1:
            st.error(f"⚠ High Pest Infestation Risk for {st.session_state.selected_crop}")
        else:
            st.success(f"✅ Low Pest Infestation Risk for {st.session_state.selected_crop}")
