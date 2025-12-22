VERDE: Weather & Air Quality Dashboard for Jaipur

VERDE is an intuitive Streamlit-based dashboard designed to visualize, analyze, 
and forecast weather conditions and air quality trends in Jaipur. 
It integrates pollution and weather datasets, applies machine learning models for prediction, 
and presents insights through interactive charts and forecasts.
The platform also features custom NO₂ heatmap generation using grid-level uploaded CSV data, 
enabling detailed spatial analysis.

Tech Stack
Python 3.10+
Pandas, Matplotlib
scikit-learn, XGBoost, Joblib
Streamlit for web dashboard

Features:

CSV Upload
Upload your cleaned or preprocessed dataset 
Upload a separate NO₂ heatmap grid-level dataset 
Trend Visualization
Select any column to generate a time-series chart.
Easily visualize trends in pollution and climate over time.
Predictive Modeling

Predict:
Nitrogen Dioxide (NO₂) in µg/m³
Temperature (next hour) in °C
PM2.5 – Particulate Matter < 2.5µm
PM10 – Particulate Matter < 10µm
Carbon Monoxide (CO) in ppm
NO₂ Heatmap
Upload a grid-level CSV file  containing lat/lon and weather columns.
Choose between Folium or Plotly map rendering.
