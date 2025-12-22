import pandas as pd

# Load datasets
air_df = pd.read_csv("air_quality.csv")
weather_df = pd.read_csv("weather.csv")

# Format air quality date to datetime
air_df['Date'] = pd.to_datetime(air_df['Date'], dayfirst=True)

# Format weather date to datetime
weather_df['DATE'] = pd.to_datetime(weather_df['DATE'])

# Select only matching date range (2018 onwards)
weather_df = weather_df[weather_df['DATE'] >= '2018-01-01']

#  Optional: Rename columns for clarity
weather_df = weather_df.rename(columns={
    'DATE': 'Date',
    'PRCP': 'Precipitation',
    'TAVG': 'AvgTemp',
    'TMAX': 'MaxTemp',
    'TMIN': 'MinTemp'
})

# Merge on 'Date'
merged_df = pd.merge(air_df, weather_df, on='Date', how='inner')

#  Save merged CSV
merged_df.to_csv("merged_data.csv", index=False)
print(" Merged successfully! Saved as merged_data.csv")
