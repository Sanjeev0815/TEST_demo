# Air Pollutant Forecasting Dashboard

A Streamlit-based dashboard application for forecasting air pollutants (O₃ and NO₂) using satellite and reanalysis data.

## Features

- **Data Upload & Preview**: Upload CSV or NetCDF files containing air quality data
- **Data Preprocessing**: Handle missing values, create lag features, and apply advanced domain-specific transformations
- **Model Training**: Train Random Forest and LSTM/GRU models for pollutant forecasting
- **Forecasting & Results**: Generate predictions with safety threshold monitoring and export capabilities

## Installation

1. Clone or download this repository
2. Install Python 3.11 or higher
3. Install the required packages:

```bash
pip install -r local_requirements.txt
```

Note: Rename `local_requirements.txt` to `requirements.txt` before running the pip install command.

## Usage

1. Start the application:

```bash
streamlit run app.py
```

2. Open your web browser and navigate to `http://localhost:8501`

3. Upload your air quality data (CSV format) with the following required columns:
   - `datetime` or `date`: Timestamp
   - `o3`: Ozone concentration (µg/m³)
   - `no2`: NO₂ concentration (µg/m³)
   - `temperature`: Temperature (°C)
   - `wind_speed`: Wind speed (m/s)
   - `humidity`: Relative humidity (%)

4. Follow the workflow through the pages:
   - **Data Upload & Preview**: Upload and inspect your data
   - **Data Preprocessing**: Configure preprocessing options
   - **Model Training**: Train forecasting models
   - **Forecasting & Results**: Generate and export predictions

## System Requirements

- Python 3.11+
- Minimum 4GB RAM (8GB recommended for large datasets)
- TensorFlow-compatible CPU (GPU optional but recommended for large models)

## Data Format

The application accepts CSV files with air quality and meteorological data. See the sample data format in the Data Upload page for detailed requirements.

## Export Options

- CSV reports with forecast data
- JSON format for API integration
- Text summary reports

## Safety Thresholds

The application uses WHO air quality guidelines:
- O₃: 100 µg/m³
- NO₂: 40 µg/m³