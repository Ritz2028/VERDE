import pandas as pd
import folium
from folium.plugins import HeatMap
import plotly.express as px
from branca.element import Element


class Visualiser:
    def __init__(self, model, driving_factors, city, year, custom_grid_df=None) -> None:
        self.model = model
        self.driving_factors = driving_factors
        self.city = city

        if custom_grid_df is not None:
            self.grid_df = custom_grid_df.copy()
            self.lat_min = self.grid_df['latitude'].min()
            self.lat_max = self.grid_df['latitude'].max()
            self.lon_min = self.grid_df['longitude'].min()
            self.lon_max = self.grid_df['longitude'].max()
        else:
            # Load grid based on city/year
            if city == "Jaipur":
                file = "data/Jaipur_2019_inference.csv" if year == "2019" else "data/Jaipur_2022_inference.csv"
                self.grid_df = pd.read_csv(file)
                self.lat_min, self.lat_max = 12.85, 13.20
                self.lon_min, self.lon_max = 77.45, 77.80
            else:
                self.grid_df = pd.read_csv("data/delhi_2019_inference.csv")
                self.lat_min, self.lat_max = 28.40, 28.90
                self.lon_min, self.lon_max = 76.80, 77.30

        # Normalize column names
        self.grid_df = self.grid_df.rename(columns={
            "Temperature": "temperature",
            "Rainfall": "precip"
        })

        # Ensure required columns exist
        for col in ["tmax", "tmin", "day", "month", "hour"]:
            if col not in self.grid_df.columns:
                self.grid_df[col] = 15

    def foliumMap(self):
        features = [key for key in self.driving_factors if self.driving_factors[key]]
        self.grid_df['NO2_prediction'] = self.model.predict(self.grid_df[features])

        min_val = self.grid_df['NO2_prediction'].min()
        max_val = self.grid_df['NO2_prediction'].max()
        if max_val - min_val < 10:
            max_val = min_val + 10

        # Normalize predictions to [0, 1]
        heat_data = [
            [row['latitude'], row['longitude'], (row['NO2_prediction'] - min_val) / (max_val - min_val)]
            for _, row in self.grid_df.iterrows()
        ]

        m = folium.Map(
            location=[(self.lat_min + self.lat_max) / 2, (self.lon_min + self.lon_max) / 2],
            zoom_start=12,
            min_zoom=10,
            max_zoom=14,
        )

        HeatMap(
            heat_data,
            radius=18,
            blur=35,
            max_zoom=10,
            min_opacity=0.5,
            gradient={
                0.0: '#0000FF',
                0.25: '#00FF00',
                0.5: '#FFA500',
                1.0: '#FF0000'
            }
        ).add_to(m)

        legend_html = f'''
        <div style="
            position: fixed;
            top: 60px;
            right: 20px;
            z-index: 9999;
            background-color: white;
            padding: 10px;
            border: 2px solid grey;
            font-size: 13px;
            width: 160px;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
        ">
            <b>NO₂ (µg/m³)</b><br>
            <div style="height: 12px; margin: 5px 0;
                background: linear-gradient(to right, #0000FF, #00FF00, #FFA500, #FF0000);">
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span>{min_val:.0f}</span><span>{(min_val + max_val)/2:.0f}</span><span>{max_val:.0f}</span>
            </div>
        </div>
        '''
        m.get_root().html.add_child(Element(legend_html))
        return m._repr_html_()

    def plotlyMap(self, global_scale=False):
        features = [key for key in self.driving_factors if self.driving_factors[key]]
        self.grid_df['NO2_prediction'] = self.model.predict(self.grid_df[features])

        color_min, color_max = (0, 60) if global_scale else (
            self.grid_df["NO2_prediction"].min(),
            self.grid_df["NO2_prediction"].max()
        )
        if color_max - color_min < 10:
            color_max = color_min + 10

        fig = px.scatter_mapbox(
            self.grid_df, lat='latitude', lon='longitude',
            color='NO2_prediction',
            color_continuous_scale=["blue", "green", "orange", "red"],
            range_color=[color_min, color_max],
            mapbox_style='open-street-map',
            zoom=11
        )

        fig.update_layout(
            mapbox=dict(
                center={
                    "lat": (self.lat_min + self.lat_max) / 2,
                    "lon": (self.lon_min + self.lon_max) / 2
                },
                zoom=9
            ),
            height=600,
            width=1000,
            coloraxis_colorbar=dict(
                title="NO₂ (µg/m³)",
                tickvals=[color_min, (color_min + color_max)/2, color_max],
                ticktext=["Low", "Moderate", "High"]
            )
        )

        return fig
