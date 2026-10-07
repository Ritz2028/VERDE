import os

import joblib
import pandas as pd
import streamlit as st

from visualiser import Visualiser


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="VERDE - Air Quality Dashboard",
    page_icon="🌿",
    layout="wide"
)


# ============================================================
# MODEL LOADING
# ============================================================

MODEL_DIR = "models"


def load_model(filename):
    """
    Load a trained model from the models directory.
    Returns None if the model file does not exist.
    """
    path = os.path.join(MODEL_DIR, filename)

    if os.path.exists(path):
        return joblib.load(path)

    return None


model_no2 = load_model("model_no2.pkl")
model_temp = load_model("model_temp.pkl")
model_pm25 = load_model("model_pm25.pkl")
model_pm10 = load_model("model_pm10.pkl")
model_co = load_model("model_co.pkl")


# Make sure the main models exist
if model_no2 is None or model_temp is None:
    st.error(
        "Required trained models are missing. "
        "Please make sure the 'models/' folder contains "
        "model_no2.pkl and model_temp.pkl."
    )
    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("🌿 VERDE")
st.subheader("Air Quality Analysis & Prediction Dashboard")

st.markdown(
    """
    Analyze weather and air-quality data, generate pollutant predictions,
    and visualize predicted NO₂ concentrations across a Jaipur city grid.
    """
)


# ============================================================
# SECTION 1 — DATASET UPLOAD & TREND ANALYSIS
# ============================================================

st.header("📊 Air Quality & Weather Analysis")

uploaded = st.file_uploader(
    "Upload a weather or pollution dataset (CSV)",
    type=["csv"],
    key="trend_upload"
)

if uploaded is not None:

    try:
        df = pd.read_csv(uploaded)

        st.success("Dataset uploaded successfully!")

        st.write("### Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)

        st.write(
            f"**Rows:** {df.shape[0]}  |  "
            f"**Columns:** {df.shape[1]}"
        )

        st.write("### Available Columns")
        st.write(df.columns.tolist())

        # ----------------------------------------------------
        # Trend visualization
        # ----------------------------------------------------

        st.subheader("📈 Visualize Trends")

        numeric_columns = df.select_dtypes(
            include=["number"]
        ).columns.tolist()

        if numeric_columns:

            selected_column = st.selectbox(
                "Choose a parameter to visualize",
                numeric_columns,
                key="trend_parameter"
            )

            st.line_chart(
                df[selected_column],
                use_container_width=True
            )

        else:
            st.warning(
                "No numeric columns were found in the uploaded dataset."
            )

    except Exception as e:

        st.error(
            f"Unable to read the uploaded CSV file: {e}"
        )

else:

    st.info(
        "Upload a CSV file above to explore weather or pollution trends."
    )


# ============================================================
# SECTION 2 — POLLUTION & TEMPERATURE PREDICTION
# ============================================================

st.header("🧠 Environmental Predictions")

st.markdown(
    """
    Enter environmental conditions to generate model-based predictions.
    """
)

# ------------------------------------------------------------
# Environmental inputs
# ------------------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    temp = st.slider(
        "Temperature (°C)",
        min_value=10.0,
        max_value=45.0,
        value=25.0,
        step=0.5
    )

    precip = st.slider(
        "Precipitation (mm)",
        min_value=0.0,
        max_value=5.0,
        value=0.1,
        step=0.1
    )

    tmax = st.slider(
        "Maximum Temperature (°C)",
        min_value=20.0,
        max_value=50.0,
        value=35.0,
        step=0.5
    )

    tmin = st.slider(
        "Minimum Temperature (°C)",
        min_value=5.0,
        max_value=30.0,
        value=20.0,
        step=0.5
    )

with col2:

    day = st.slider(
        "Day",
        min_value=1,
        max_value=31,
        value=15
    )

    month = st.slider(
        "Month",
        min_value=1,
        max_value=12,
        value=6
    )


# ============================================================
# MODEL INPUTS
# ============================================================

# Pollutant models were trained using these 7 features:
#
# temperature
# precip
# tmax
# tmin
# day
# month
# hour
#
# The training data uses hour = 0 because the dataset is daily.
# Therefore we keep hour fixed at 0 during inference.

pollutant_features = {
    "temperature": temp,
    "precip": precip,
    "tmax": tmax,
    "tmin": tmin,
    "day": day,
    "month": month,
    "hour": 0
}


# Temperature model was trained using 6 features:
#
# temperature
# precip
# tmax
# tmin
# day
# month

temperature_features = {
    "temperature": temp,
    "precip": precip,
    "tmax": tmax,
    "tmin": tmin,
    "day": day,
    "month": month
}


input_pollutant_df = pd.DataFrame(
    [pollutant_features]
)

input_temperature_df = pd.DataFrame(
    [temperature_features]
)


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

if st.button(
    "🔮 Generate Predictions",
    use_container_width=True
):

    # --------------------------------------------------------
    # NO2
    # --------------------------------------------------------

    pred_no2 = model_no2.predict(
        input_pollutant_df
    )

    st.success(
        f"🔬 **Predicted NO₂:** "
        f"{pred_no2[0]:.2f} µg/m³"
    )


    # --------------------------------------------------------
    # Temperature
    # --------------------------------------------------------

    pred_temp = model_temp.predict(
        input_temperature_df
    )

    st.success(
        f"🌡️ **Predicted Temperature (next day):** "
        f"{pred_temp[0]:.2f} °C"
    )


    # --------------------------------------------------------
    # PM2.5
    # --------------------------------------------------------

    if model_pm25 is not None:

        pred_pm25 = model_pm25.predict(
            input_pollutant_df
        )

        st.success(
            f"🧪 **Predicted PM2.5:** "
            f"{pred_pm25[0]:.2f} µg/m³"
        )


    # --------------------------------------------------------
    # PM10
    # --------------------------------------------------------

    if model_pm10 is not None:

        pred_pm10 = model_pm10.predict(
            input_pollutant_df
        )

        st.success(
            f"🧪 **Predicted PM10:** "
            f"{pred_pm10[0]:.2f} µg/m³"
        )


    # --------------------------------------------------------
    # CO
    # --------------------------------------------------------

    if model_co is not None:

        pred_co = model_co.predict(
            input_pollutant_df
        )

        st.success(
            f"🧪 **Predicted CO:** "
            f"{pred_co[0]:.2f} ppm"
        )


# ============================================================
# SECTION 3 — NO2 SPATIAL HEATMAP
# ============================================================

st.header("🗺️ NO₂ Spatial Prediction")

st.markdown(
    """
    Generate an interactive NO₂ prediction map across a Jaipur
    latitude/longitude grid.
    """
)


# ============================================================
# CUSTOM GRID UPLOAD
# ============================================================

grid_upload = st.file_uploader(
    "Upload a custom grid CSV (optional)",
    type=["csv"],
    key="grid_upload"
)

custom_grid_df = None
custom_grid_valid = True


if grid_upload is not None:

    try:

        custom_grid_df = pd.read_csv(
            grid_upload
        )

        # ----------------------------------------------------
        # Required columns for the trained NO2 model
        # ----------------------------------------------------

        required_grid_columns = [
            "latitude",
            "longitude",
            "temperature",
            "precip",
            "tmax",
            "tmin",
            "day",
            "month",
            "hour"
        ]

        missing_columns = [
            column
            for column in required_grid_columns
            if column not in custom_grid_df.columns
        ]

        if missing_columns:

            custom_grid_valid = False

            st.error(
                "❌ Invalid grid file. Missing columns: "
                + ", ".join(missing_columns)
            )

            st.info(
                "Your grid CSV must contain: "
                + ", ".join(required_grid_columns)
            )

        else:

            st.success(
                "✅ Custom grid uploaded successfully!"
            )

            st.write("### Custom Grid Preview")

            st.dataframe(
                custom_grid_df.head(),
                use_container_width=True
            )

            st.write(
                f"**Grid points:** "
                f"{len(custom_grid_df)}"
            )


    except Exception as e:

        custom_grid_valid = False

        st.error(
            f"Unable to read the grid CSV: {e}"
        )


# ============================================================
# MAP SETTINGS
# ============================================================

map_col1, map_col2 = st.columns(2)

with map_col1:

    map_type = st.radio(
        "Select Map Type",
        ["Folium", "Plotly"],
        horizontal=True
    )

with map_col2:

    year = st.selectbox(
        "Select Grid Data Year",
        ["2019", "2022"]
    )


# ============================================================
# MODEL FEATURES USED FOR HEATMAP
# ============================================================

st.markdown("### 🌡️ Environmental Features")

st.info(
    """
    The trained NO₂ model requires all seven environmental features.
    These features are therefore used automatically for every grid point.
    """
)

st.code(
    """
temperature
precip
tmax
tmin
day
month
hour
""",
    language="text"
)


# All seven features are required because that is how model_no2.pkl
# was trained.

driving_factors = {
    "temperature": True,
    "precip": True,
    "tmax": True,
    "tmin": True,
    "day": True,
    "month": True,
    "hour": True
}


# ============================================================
# GENERATE HEATMAP
# ============================================================

if st.button(
    "🌍 Generate NO₂ Heatmap",
    use_container_width=True
):

    if not custom_grid_valid:

        st.error(
            "Please upload a valid grid CSV or remove the custom upload."
        )

    else:

        try:

            # ------------------------------------------------
            # Create visualiser
            # ------------------------------------------------

            vis = Visualiser(
                model=model_no2,
                driving_factors=driving_factors,
                city="Jaipur",
                year=year,
                custom_grid_df=custom_grid_df
            )


            # ------------------------------------------------
            # Folium map
            # ------------------------------------------------

            if map_type == "Folium":

                folium_html = vis.foliumMap()

                st.components.v1.html(
                    folium_html,
                    height=650,
                    scrolling=False
                )


            # ------------------------------------------------
            # Plotly map
            # ------------------------------------------------

            else:

                fig = vis.plotlyMap(
                    global_scale=True
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )


        except Exception as e:

            st.error(
                f"Unable to generate the NO₂ heatmap: {e}"
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "VERDE | Air Quality Analysis & Prediction Dashboard"
)
