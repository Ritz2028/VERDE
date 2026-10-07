# VERDE — Weather & Air Quality Dashboard for Jaipur

VERDE is a Streamlit-based dashboard for analyzing weather and air-quality
data and generating machine-learning-based predictions.

## Features

- Upload CSV datasets for trend analysis
- Interactive pollution and weather visualizations
- NO₂ prediction using Random Forest regression
- PM2.5 prediction using Random Forest regression
- PM10 prediction using Random Forest regression
- CO prediction using Random Forest regression
- Next-day temperature prediction using XGBoost
- Interactive NO₂ spatial prediction maps
- Folium and Plotly map visualizations
- Custom latitude/longitude grid uploads

## Tech Stack

- Python
- Pandas
- Scikit-learn
- Random Forest
- XGBoost
- Streamlit
- Folium
- Plotly

## Spatial Visualization

The application generates NO₂ predictions across a configurable
latitude/longitude grid using environmental input features such as
temperature, precipitation, maximum temperature, minimum temperature,
day, month and hour.
