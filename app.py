import streamlit as st
import pandas as pd
import joblib
import os
from visualiser import Visualiser

# Load base models
model_no2 = joblib.load("models/model_no2.pkl")
model_temp = joblib.load("models/model_temp.pkl")

# Load optional models
def load_optional_model(name):
    path = f"models/{name}"
    return joblib.load(path) if os.path.exists(path) else None

model_pm25 = load_optional_model("model_pm25.pkl")
model_pm10 = load_optional_model("model_pm10.pkl")
model_co = load_optional_model("model_co.pkl")

# Streamlit setup
st.set_page_config(page_title="EnviroCast", layout="wide")
st.title("🌤️ EnviroCast: Weather & Air Quality Dashboard - Jaipur")

# ---------------------------
# 📂 Upload for Trend Analysis
# ---------------------------
uploaded = st.file_uploader("📂 Upload your weather or pollution dataset (CSV)", type="csv", key="trend_upload")

if uploaded:
    df = pd.read_csv(uploaded)
    st.write("📊 Data Preview", df.head())
    st.write("✅ Columns in uploaded file:", df.columns.tolist())

    st.subheader("📈 Visualize Trends")
    col = st.selectbox("Choose a parameter to plot", df.columns)
    st.line_chart(df[col])
else:
    st.info("⬆️ Upload a CSV file to get started.")

# -----------------------
# 🔮 Prediction Section
# -----------------------
st.subheader("🧠 Make Predictions")

# User inputs
temp = st.slider("Temperature (°C)", 10, 45, 25)
precip = st.slider("Precipitation (mm)", 0.0, 5.0, 0.1)
tmax = st.slider("Max Temperature (°C)", 20, 50, 35)
tmin = st.slider("Min Temperature (°C)", 5, 30, 20)
day = st.slider("Day", 1, 31, 15)
month = st.slider("Month", 1, 12, 6)
hour = st.slider("Hour", 0, 23, 12)

input_features = {
    "temperature": temp,
    "precip": precip,
    "tmax": tmax,
    "tmin": tmin,
    "day": day,
    "month": month,
    "hour": hour
}

input_no2_df = pd.DataFrame([input_features])
input_temp_df = pd.DataFrame([{
    "temperature": temp,
    "precip": precip,
    "tmax": tmax,
    "tmin": tmin,
    "day": day,
    "month": month
}])

# Display predictions
pred_no2 = model_no2.predict(input_no2_df)
pred_temp = model_temp.predict(input_temp_df)

st.success(f"🔬 **Predicted Nitrogen Dioxide (NO₂)**: {pred_no2[0]:.2f} µg/m³")
st.success(f"🌡️ **Forecasted Temperature (next hour)**: {pred_temp[0]:.2f} °C")

if model_pm25:
    pred_pm25 = model_pm25.predict(input_no2_df)
    st.success(f"🧪 **Predicted PM2.5**: {pred_pm25[0]:.2f} µg/m³")
if model_pm10:
    pred_pm10 = model_pm10.predict(input_no2_df)
    st.success(f"🧪 **Predicted PM10**: {pred_pm10[0]:.2f} µg/m³")
if model_co:
    pred_co = model_co.predict(input_no2_df)
    st.success(f"🧪 **Predicted CO**: {pred_co[0]:.2f} ppm")

# -----------------------
# 🗺️ NO₂ Heatmap Section
# -----------------------
st.subheader("🗼 NO₂ Heatmap - City-wide Spatial View")

# Upload grid-level CSV (lat/lon + weather features)
grid_upload = st.file_uploader("📂 Upload Grid-Level CSV for Heatmap (optional)", type="csv", key="grid_upload")
custom_grid_df = None
if grid_upload:
    custom_grid_df = pd.read_csv(grid_upload)
    st.success("✅ Grid data uploaded successfully!")
    st.write(custom_grid_df.head())

map_type = st.radio("Select Map Type", ["Folium", "Plotly"])
year = st.selectbox("Select Grid Data Year (for fallback)", ["2019", "2022"])

st.markdown("✅ Choose driving features for model input:")
driving_factors = {
    "temperature": st.checkbox("Temperature", value=True),
    "precip": st.checkbox("Precipitation", value=True),
    "tmax": st.checkbox("Max Temperature", value=True),
    "tmin": st.checkbox("Min Temperature", value=True),
    "day": st.checkbox("Day", value=True),
    "month": st.checkbox("Month", value=True),
    "hour": st.checkbox("Hour", value=True)
}

# Generate the heatmap
if st.button("🌍 Generate NO₂ Heatmap"):
    vis = Visualiser(
        model=model_no2,
        driving_factors=driving_factors,
        city="Jaipur",
        year=year,
        custom_grid_df=custom_grid_df  #  This is the fixed name
    )

    if map_type == "Folium":
        folium_html = vis.foliumMap()
        st.components.v1.html(folium_html, height=600)
    else:
        fig = vis.plotlyMap(global_scale=True)
        st.plotly_chart(fig)
