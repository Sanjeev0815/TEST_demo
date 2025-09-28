import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def load_css():
    """Load custom CSS for the application."""
    return """
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .alert-danger {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 0.75rem 1.25rem;
        border-radius: 0.375rem;
        margin: 1rem 0;
    }
    
    .alert-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.75rem 1.25rem;
        border-radius: 0.375rem;
        margin: 1rem 0;
    }
    
    .alert-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 0.75rem 1.25rem;
        border-radius: 0.375rem;
        margin: 1rem 0;
    }
    </style>
    """

def calculate_metrics(actual, predicted):
    """Calculate evaluation metrics for model predictions."""
    if len(actual) == 0 or len(predicted) == 0:
        return {'rmse': 0, 'mae': 0, 'r2': 0}
    
    # Ensure same length
    min_length = min(len(actual), len(predicted))
    actual = actual[-min_length:] if len(actual) > min_length else actual
    predicted = predicted[-min_length:] if len(predicted) > min_length else predicted
    
    try:
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        r2 = r2_score(actual, predicted)
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return {'rmse': 0, 'mae': 0, 'r2': 0}

def check_safety_thresholds(predictions, thresholds):
    """Check if predictions exceed safety thresholds."""
    alerts = {}
    
    for pollutant, threshold in thresholds.items():
        if pollutant in predictions:
            pred_values = predictions[pollutant].get('future_predictions', [])
            if pred_values:
                max_value = max(pred_values)
                alerts[pollutant] = {
                    'exceeds_threshold': max_value > threshold,
                    'max_value': max_value,
                    'threshold': threshold
                }
    
    return alerts

def generate_sample_data(start_date='2023-01-01', end_date='2023-12-31', freq='1H'):
    """
    Generate sample air quality data for demonstration purposes.
    This function should only be used when real data is not available.
    """
    
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    n_points = len(date_range)
    
    # Generate synthetic but realistic data
    np.random.seed(42)  # For reproducibility
    
    # Base patterns
    hour_pattern = np.sin(2 * np.pi * np.arange(n_points) / 24)  # Daily cycle
    seasonal_pattern = np.sin(2 * np.pi * np.arange(n_points) / (365 * 24))  # Seasonal cycle
    
    # Meteorological variables
    temperature = 15 + 10 * seasonal_pattern + 5 * hour_pattern + np.random.normal(0, 2, n_points)
    wind_speed = 3 + 2 * np.random.exponential(1, n_points)
    humidity = 60 + 20 * np.sin(2 * np.pi * np.arange(n_points) / (24 * 7)) + np.random.normal(0, 5, n_points)
    solar_radiation = np.maximum(0, 500 + 300 * hour_pattern + 200 * seasonal_pattern + np.random.normal(0, 50, n_points))
    pressure = 1013 + np.random.normal(0, 10, n_points)
    
    # Pollutant concentrations (correlated with meteorological conditions)
    o3_base = 50 + 20 * hour_pattern + 15 * (temperature - 15) / 10 - 5 * wind_speed / 3
    o3 = np.maximum(0, o3_base + np.random.normal(0, 10, n_points))
    
    no2_base = 30 - 10 * hour_pattern + 5 * (1 / (wind_speed + 1)) + 0.5 * (humidity - 60) / 10
    no2 = np.maximum(0, no2_base + np.random.normal(0, 5, n_points))
    
    # Create DataFrame
    data = pd.DataFrame({
        'datetime': date_range,
        'o3': o3,
        'no2': no2,
        'temperature': temperature,
        'wind_speed': wind_speed,
        'humidity': np.clip(humidity, 0, 100),  # Humidity should be 0-100%
        'solar_radiation': np.maximum(0, solar_radiation),
        'pressure': pressure,
        'latitude': 40.7128 + np.random.normal(0, 0.1, n_points),  # Around NYC
        'longitude': -74.0060 + np.random.normal(0, 0.1, n_points)
    })
    
    return data

def validate_model_inputs(data, target_column):
    """Validate inputs for model training."""
    errors = []
    warnings = []
    
    # Check if target column exists
    if target_column not in data.columns:
        errors.append(f"Target column '{target_column}' not found in data")
    
    # Check for sufficient data
    if len(data) < 100:
        warnings.append("Dataset has fewer than 100 samples. Consider using more data for better model performance.")
    
    # Check for missing values in target
    if target_column in data.columns:
        missing_target = data[target_column].isnull().sum()
        if missing_target > len(data) * 0.1:
            warnings.append(f"Target column '{target_column}' has {missing_target} missing values ({missing_target/len(data)*100:.1f}%)")
    
    # Check temporal coverage
    if 'datetime' in data.columns:
        date_range = (pd.to_datetime(data['datetime']).max() - pd.to_datetime(data['datetime']).min()).days
        if date_range < 30:
            warnings.append("Dataset covers less than 30 days. Longer time series may improve model performance.")
    
    return {'errors': errors, 'warnings': warnings}

def format_prediction_summary(predictions, target):
    """Format prediction results for display."""
    summary = {}
    
    if 'future_predictions' in predictions:
        future_preds = predictions['future_predictions']
        if future_preds:
            summary['current_forecast'] = future_preds[0]
            summary['max_forecast'] = max(future_preds)
            summary['min_forecast'] = min(future_preds)
            summary['avg_forecast'] = np.mean(future_preds)
            summary['trend'] = 'increasing' if future_preds[-1] > future_preds[0] else 'decreasing'
    
    if 'actual' in predictions and 'predicted' in predictions:
        metrics = calculate_metrics(predictions['actual'], predictions['predicted'])
        summary.update(metrics)
    
    return summary

def create_alerts_summary(predictions, safety_thresholds):
    """Create summary of safety alerts."""
    alerts = []
    
    for target, threshold in safety_thresholds.items():
        if target in predictions:
            future_preds = predictions[target].get('future_predictions', [])
            if future_preds:
                max_pred = max(future_preds)
                if max_pred > threshold:
                    alerts.append({
                        'pollutant': target.upper(),
                        'max_predicted': max_pred,
                        'threshold': threshold,
                        'severity': 'high' if max_pred > threshold * 1.5 else 'medium'
                    })
    
    return alerts

def export_predictions_to_csv(predictions, targets):
    """Export prediction results to CSV format."""
    export_data = []
    
    for target in targets:
        if target in predictions:
            pred_data = predictions[target]
            
            # Historical data
            if 'actual' in pred_data and 'predicted' in pred_data:
                actual_values = pred_data['actual'] 
                predicted_values = pred_data['predicted']
                min_len = min(len(actual_values), len(predicted_values))
                
                for i in range(min_len):
                    export_data.append({
                        'pollutant': target.upper(),
                        'data_type': 'historical',
                        'day': i + 1,
                        'actual_concentration': actual_values[i],
                        'predicted_concentration': predicted_values[i],
                        'timestamp': pd.Timestamp.now() - pd.Timedelta(days=min_len-i)
                    })
            
            # Future predictions
            if 'future_predictions' in pred_data:
                for i, pred in enumerate(pred_data['future_predictions']):
                    export_data.append({
                        'pollutant': target.upper(),
                        'data_type': 'forecast',
                        'day': i + 1,
                        'actual_concentration': None,
                        'predicted_concentration': pred,
                        'timestamp': pd.Timestamp.now() + pd.Timedelta(days=i+1)
                    })
    
    if export_data:
        df = pd.DataFrame(export_data)
        return df.to_csv(index=False)
    
    return ""

def export_predictions_to_json(predictions, targets):
    """Export prediction results to JSON format."""
    export_dict = {
        'export_timestamp': pd.Timestamp.now().isoformat(),
        'forecast_results': {}
    }
    
    for target in targets:
        if target in predictions:
            pred_data = predictions[target]
            export_dict['forecast_results'][target] = {
                'historical_actual': pred_data.get('actual', []),
                'historical_predicted': pred_data.get('predicted', []),
                'future_predictions': pred_data.get('future_predictions', []),
                'metrics': calculate_metrics(
                    pred_data.get('actual', []), 
                    pred_data.get('predicted', [])
                ) if 'actual' in pred_data and 'predicted' in pred_data else {}
            }
    
    import json
    return json.dumps(export_dict, indent=2, default=str)

def export_summary_report(predictions, targets, summary_df):
    """Export a text summary report."""
    report_lines = [
        "AIR POLLUTANT FORECAST SUMMARY REPORT",
        "=" * 50,
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "FORECAST OVERVIEW:",
        "-" * 20
    ]
    
    # Add summary table
    if not summary_df.empty:
        report_lines.append("Pollutant Summary:")
        report_lines.append(summary_df.to_string(index=False))
        report_lines.append("")
    
    # Add detailed predictions
    for target in targets:
        if target in predictions:
            pred_data = predictions[target]
            report_lines.extend([
                f"{target.upper()} DETAILED FORECAST:",
                "-" * 30
            ])
            
            if 'future_predictions' in pred_data:
                for i, pred in enumerate(pred_data['future_predictions']):
                    report_lines.append(f"Day {i+1}: {pred:.2f} µg/m³")
            
            if 'actual' in pred_data and 'predicted' in pred_data:
                metrics = calculate_metrics(pred_data['actual'], pred_data['predicted'])
                report_lines.extend([
                    "",
                    "Model Performance:",
                    f"  RMSE: {metrics['rmse']:.2f}",
                    f"  MAE: {metrics['mae']:.2f}",
                    f"  R²: {metrics['r2']:.3f}"
                ])
            
            report_lines.append("")
    
    report_lines.extend([
        "SAFETY THRESHOLDS:",
        "-" * 20,
        "O3: 100 µg/m³ (WHO guideline)",
        "NO2: 40 µg/m³ (WHO guideline)",
        "",
        "Note: This forecast is for research purposes.",
        "Consult official sources for public health decisions."
    ])
    
    return "\n".join(report_lines)
