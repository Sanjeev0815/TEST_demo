import pandas as pd
import numpy as np
import xarray as xr
import streamlit as st
from io import StringIO, BytesIO

@st.cache_data
def load_and_validate_csv_data(file_content):
    """Cached CSV loading for better performance."""
    return pd.read_csv(StringIO(file_content))

class DataHandler:
    """Handles data loading and basic quality checks for air pollutant forecasting."""
    
    def __init__(self):
        self.required_columns = ['datetime', 'o3', 'no2', 'temperature', 'wind_speed', 'humidity']
        self.optional_columns = ['solar_radiation', 'pressure', 'latitude', 'longitude']
    
    def load_csv_data(self, uploaded_file):
        """Load CSV data from uploaded file."""
        try:
            # Read CSV file
            if hasattr(uploaded_file, 'read'):
                content = uploaded_file.read()
                if isinstance(content, bytes):
                    content = content.decode('utf-8')
                data = load_and_validate_csv_data(content)
            else:
                data = pd.read_csv(uploaded_file)
            
            # Basic data validation
            if data.empty:
                raise ValueError("Uploaded file is empty")
            
            # Check for datetime column (flexible naming)
            date_columns = ['datetime', 'date', 'timestamp', 'time']
            date_col = None
            for col in date_columns:
                if col in data.columns:
                    date_col = col
                    break
            
            if date_col is None:
                # Try to find datetime-like column
                for col in data.columns:
                    if 'date' in col.lower() or 'time' in col.lower():
                        date_col = col
                        break
            
            if date_col:
                data['datetime'] = pd.to_datetime(data[date_col])
                if date_col != 'datetime':
                    data = data.drop(columns=[date_col])
            else:
                st.warning("No datetime column found. Please ensure your data has a datetime column.")
            
            # Convert column names to lowercase for consistency
            data.columns = data.columns.str.lower()
            
            # Handle common column name variations
            column_mapping = {
                'o3_concentration': 'o3',
                'ozone': 'o3',
                'no2_concentration': 'no2',
                'nitrogen_dioxide': 'no2',
                'temp': 'temperature',
                'wind': 'wind_speed',
                'windspeed': 'wind_speed',
                'relative_humidity': 'humidity',
                'rh': 'humidity',
                'solar_rad': 'solar_radiation',
                'radiation': 'solar_radiation',
                'lat': 'latitude',
                'lon': 'longitude',
                'lng': 'longitude'
            }
            
            for old_name, new_name in column_mapping.items():
                if old_name in data.columns:
                    data = data.rename(columns={old_name: new_name})
            
            return data
            
        except Exception as e:
            raise Exception(f"Error loading CSV file: {str(e)}")
    
    def load_netcdf_data(self, uploaded_file):
        """Load NetCDF data from uploaded file."""
        try:
            # Save uploaded file temporarily
            with open('temp_netcdf.nc', 'wb') as f:
                f.write(uploaded_file.read())
            
            # Load with xarray
            ds = xr.open_dataset('temp_netcdf.nc')
            
            # Convert to pandas DataFrame
            data = ds.to_dataframe().reset_index()
            
            # Clean up temporary file
            import os
            os.remove('temp_netcdf.nc')
            
            # Basic processing similar to CSV
            data.columns = data.columns.str.lower()
            
            # Handle time dimension
            time_columns = ['time', 'datetime', 'date']
            for col in time_columns:
                if col in data.columns:
                    data['datetime'] = pd.to_datetime(data[col])
                    if col != 'datetime':
                        data = data.drop(columns=[col])
                    break
            
            return data
            
        except Exception as e:
            raise Exception(f"Error loading NetCDF file: {str(e)}")
    
    def check_data_quality(self, data):
        """Perform basic data quality checks."""
        quality_checks = {}
        
        # Check for required columns
        missing_required = [col for col in ['o3', 'no2'] if col not in data.columns]
        quality_checks['Has required pollutant columns'] = len(missing_required) == 0
        
        # Check for datetime column
        quality_checks['Has datetime column'] = 'datetime' in data.columns
        
        # Check for meteorological variables
        met_vars = ['temperature', 'wind_speed', 'humidity']
        has_met_vars = sum([1 for var in met_vars if var in data.columns])
        quality_checks[f'Has meteorological variables ({has_met_vars}/3)'] = has_met_vars >= 2
        
        # Check data size
        quality_checks['Sufficient data points (>100)'] = len(data) > 100
        
        # Check for excessive missing values
        if not data.empty:
            missing_percentage = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
            quality_checks['Missing data < 30%'] = missing_percentage < 30
        
        # Check for valid pollutant ranges
        if 'o3' in data.columns:
            o3_valid = ((data['o3'] >= 0) & (data['o3'] <= 1000)).all()
            quality_checks['O3 values in valid range (0-1000 µg/m³)'] = o3_valid
        
        if 'no2' in data.columns:
            no2_valid = ((data['no2'] >= 0) & (data['no2'] <= 500)).all()
            quality_checks['NO2 values in valid range (0-500 µg/m³)'] = no2_valid
        
        return quality_checks
    
    def get_data_summary(self, data):
        """Generate a summary of the dataset."""
        summary = {
            'total_records': len(data),
            'columns': list(data.columns),
            'date_range': None,
            'missing_data_percent': 0,
            'pollutant_stats': {}
        }
        
        if 'datetime' in data.columns:
            data['datetime'] = pd.to_datetime(data['datetime'])
            summary['date_range'] = {
                'start': data['datetime'].min(),
                'end': data['datetime'].max(),
                'days': (data['datetime'].max() - data['datetime'].min()).days
            }
        
        # Missing data percentage
        total_cells = len(data) * len(data.columns)
        missing_cells = data.isnull().sum().sum()
        summary['missing_data_percent'] = (missing_cells / total_cells) * 100
        
        # Pollutant statistics
        for pollutant in ['o3', 'no2']:
            if pollutant in data.columns:
                summary['pollutant_stats'][pollutant] = {
                    'mean': data[pollutant].mean(),
                    'median': data[pollutant].median(),
                    'std': data[pollutant].std(),
                    'min': data[pollutant].min(),
                    'max': data[pollutant].max(),
                    'missing_count': data[pollutant].isnull().sum()
                }
        
        return summary
    
    def validate_data_structure(self, data):
        """Validate that data has the minimum required structure for modeling."""
        errors = []
        warnings = []
        
        # Check for essential columns
        if 'o3' not in data.columns and 'no2' not in data.columns:
            errors.append("Data must contain at least one pollutant column (o3 or no2)")
        
        # Check datetime
        if 'datetime' not in data.columns:
            errors.append("Data must contain a datetime column")
        
        # Check minimum number of records
        if len(data) < 50:
            errors.append("Data must contain at least 50 records for meaningful modeling")
        
        # Check for meteorological variables
        met_vars = ['temperature', 'wind_speed', 'humidity']
        available_met_vars = [var for var in met_vars if var in data.columns]
        if len(available_met_vars) < 2:
            warnings.append("Consider including more meteorological variables for better predictions")
        
        # Check for temporal coverage
        if 'datetime' in data.columns:
            data['datetime'] = pd.to_datetime(data['datetime'])
            date_range_days = (data['datetime'].max() - data['datetime'].min()).days
            if date_range_days < 30:
                warnings.append("Consider using data covering a longer time period (>30 days)")
        
        return {'errors': errors, 'warnings': warnings}
