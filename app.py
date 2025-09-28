import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from data_handler import DataHandler
from preprocessor import DataPreprocessor
from models import PollutantForecaster
from visualizer import Visualizer
from utils import load_css, calculate_metrics, check_safety_thresholds, export_predictions_to_csv, export_predictions_to_json, export_summary_report

# Page configuration
st.set_page_config(
    page_title="Air Pollutant Forecasting Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'predictions_made' not in st.session_state:
    st.session_state.predictions_made = False

def main():
    st.title("🌍 Air Pollutant Forecasting Dashboard")
    st.markdown("### Short-term forecast of O₃ and NO₂ using satellite and reanalysis data")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page:",
        ["Data Upload & Preview", "Data Preprocessing", "Model Training", "Forecasting & Results"]
    )
    
    # Initialize components
    data_handler = DataHandler()
    preprocessor = DataPreprocessor()
    forecaster = PollutantForecaster()
    visualizer = Visualizer()
    
    if page == "Data Upload & Preview":
        data_upload_page(data_handler, visualizer)
    elif page == "Data Preprocessing":
        preprocessing_page(preprocessor, visualizer)
    elif page == "Model Training":
        model_training_page(forecaster, visualizer)
    elif page == "Forecasting & Results":
        forecasting_page(forecaster, visualizer)

def data_upload_page(data_handler, visualizer):
    st.header("📁 Data Upload & Preview")
    
    # File upload section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Satellite/Reanalysis Data")
        uploaded_file = st.file_uploader(
            "Choose a CSV or NetCDF file",
            type=['csv', 'nc', 'netcdf'],
            help="Upload satellite data containing O₃, NO₂ concentrations and meteorological variables"
        )
        
        if uploaded_file is not None:
            try:
                # Load data based on file type
                if uploaded_file.name.endswith('.csv'):
                    data = data_handler.load_csv_data(uploaded_file)
                else:
                    data = data_handler.load_netcdf_data(uploaded_file)
                
                # Cache data in session state to avoid re-processing
                if 'data_cache_key' not in st.session_state or st.session_state.data_cache_key != uploaded_file.name:
                    st.session_state.raw_data = data
                    st.session_state.data_cache_key = uploaded_file.name
                    st.session_state.data_loaded = True
                else:
                    # Data already cached
                    pass
                st.success(f"✅ Data loaded successfully! Shape: {data.shape}")
                
            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
                return
    
    with col2:
        st.subheader("Data Requirements")
        st.info("""
        **Required columns:**
        - `datetime` or `date`: Timestamp
        - `o3`: Ozone concentration (µg/m³)
        - `no2`: NO₂ concentration (µg/m³)
        - `temperature`: Temperature (°C)
        - `wind_speed`: Wind speed (m/s)
        - `humidity`: Relative humidity (%)
        - `solar_radiation`: Solar radiation (W/m²)
        
        **Optional columns:**
        - `latitude`, `longitude`: Spatial coordinates
        - `pressure`: Atmospheric pressure (hPa)
        """)
    
    # Data preview section
    if st.session_state.data_loaded:
        st.subheader("📊 Data Preview")
        
        data = st.session_state.raw_data
        
        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(data))
        with col2:
            st.metric("Columns", len(data.columns))
        with col3:
            missing_pct = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
            st.metric("Missing Data %", f"{missing_pct:.1f}%")
        with col4:
            if 'datetime' in data.columns or 'date' in data.columns:
                date_col = 'datetime' if 'datetime' in data.columns else 'date'
                data[date_col] = pd.to_datetime(data[date_col])
                date_range = (data[date_col].max() - data[date_col].min()).days
                st.metric("Date Range (days)", date_range)
        
        # Data sample
        st.subheader("Data Sample")
        st.dataframe(data.head(10), width='stretch')
        
        # Data quality check
        st.subheader("Data Quality Check")
        quality_check = data_handler.check_data_quality(data)
        
        for check, status in quality_check.items():
            if status:
                st.success(f"✅ {check}")
            else:
                st.warning(f"⚠️ {check}")
        
        # Basic visualizations
        if all(col in data.columns for col in ['o3', 'no2']):
            st.subheader("Pollutant Concentration Overview")
            fig = visualizer.plot_pollutant_overview(data)
            st.plotly_chart(fig, use_container_width=True)

def preprocessing_page(preprocessor, visualizer):
    st.header("🔧 Data Preprocessing")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data first in the 'Data Upload & Preview' page.")
        return
    
    data = st.session_state.raw_data.copy()
    
    # Preprocessing options
    st.subheader("Preprocessing Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Missing Value Handling")
        missing_method = st.selectbox(
            "Method for handling missing values:",
            ["forward_fill", "backward_fill", "interpolate", "drop"]
        )
        
        st.subheader("Feature Engineering")
        create_lag_features = st.checkbox("Create lag features", value=True)
        lag_periods = st.slider("Lag periods", 1, 7, 3) if create_lag_features else 0
        
        create_rolling_features = st.checkbox("Create rolling averages", value=True)
        rolling_window = st.slider("Rolling window size", 2, 24, 6) if create_rolling_features else 0
        
        st.subheader("Advanced Features")
        advanced_features = st.checkbox("Enable advanced domain-specific features", value=True, 
                                       help="Includes interaction terms, atmospheric stability indicators, and photochemical potential")
        
        if advanced_features:
            st.info("🧠 Advanced features include: heat index, wind chill, photochemical potential, atmospheric stability, pollutant ratios, and enhanced temporal encoding")
    
    with col2:
        st.subheader("Temporal Alignment")
        resample_freq = st.selectbox(
            "Resample frequency:",
            ["1H", "3H", "6H", "12H", "1D"],
            index=3
        )
        
        st.subheader("Data Splitting")
        train_split = st.slider("Training data percentage", 0.6, 0.9, 0.8)
        
    # Preprocessing button
    if st.button("🚀 Run Preprocessing", type="primary"):
        with st.spinner("Processing data..."):
            try:
                # Apply preprocessing
                processed_data = preprocessor.preprocess_data(
                    data,
                    missing_method=missing_method,
                    resample_freq=resample_freq,
                    create_lag_features=create_lag_features,
                    lag_periods=lag_periods,
                    create_rolling_features=create_rolling_features,
                    rolling_window=rolling_window,
                    train_split=train_split,
                    use_advanced_features=advanced_features
                )
                
                st.session_state.processed_data = processed_data
                st.session_state.preprocessing_params = {
                    'missing_method': missing_method,
                    'resample_freq': resample_freq,
                    'lag_periods': lag_periods,
                    'rolling_window': rolling_window,
                    'train_split': train_split
                }
                
                st.success("✅ Data preprocessing completed!")
                
                # Display preprocessing results
                st.subheader("Preprocessing Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Training Samples", len(processed_data['X_train']))
                with col2:
                    st.metric("Test Samples", len(processed_data['X_test']))
                with col3:
                    st.metric("Features", processed_data['X_train'].shape[1])
                
                # Feature importance preview
                st.subheader("Feature Overview")
                feature_df = pd.DataFrame({
                    'Feature': processed_data['feature_names'],
                    'Type': ['Original' if not any(x in feat for x in ['lag', 'rolling']) 
                            else 'Engineered' for feat in processed_data['feature_names']]
                })
                st.dataframe(feature_df, width='stretch')
                
                # Visualize processed data
                st.subheader("Processed Data Visualization")
                fig = visualizer.plot_processed_data(processed_data)
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error during preprocessing: {str(e)}")

def model_training_page(forecaster, visualizer):
    st.header("🤖 Model Training")
    
    if 'processed_data' not in st.session_state:
        st.warning("⚠️ Please complete data preprocessing first.")
        return
    
    processed_data = st.session_state.processed_data
    
    # Model configuration
    st.subheader("Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("LSTM/GRU Configuration")
        model_type = st.selectbox("Model Type", ["LSTM", "GRU"])
        lstm_units = st.slider("Units", 32, 256, 64)
        lstm_layers = st.slider("Layers", 1, 3, 2)
        dropout_rate = st.slider("Dropout Rate", 0.0, 0.5, 0.2)
        
    with col2:
        st.subheader("Random Forest Configuration")
        n_estimators = st.slider("Number of Estimators", 50, 500, 100)
        max_depth = st.slider("Max Depth", 5, 30, 10)
        
        st.subheader("Training Configuration")
        epochs = st.slider("Epochs", 5, 50, 20)
        batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
    
    # Target selection
    st.subheader("Target Variables")
    targets = st.multiselect(
        "Select pollutants to predict:",
        ["o3", "no2"],
        default=["o3", "no2"]
    )
    
    if not targets:
        st.warning("⚠️ Please select at least one target variable.")
        return
    
    # Training button
    if st.button("🎯 Train Models", type="primary"):
        with st.spinner("Training models... This may take a few minutes."):
            try:
                # Train models for each target
                models = {}
                training_history = {}
                
                for target in targets:
                    st.info(f"Training models for {target.upper()}...")
                    
                    # Prepare target-specific data
                    target_data = forecaster.prepare_target_data(processed_data, target)
                    
                    # Train LSTM model
                    lstm_model, lstm_history = forecaster.train_lstm_model(
                        target_data,
                        model_type=model_type,
                        units=lstm_units,
                        num_layers=lstm_layers,
                        dropout_rate=dropout_rate,
                        epochs=epochs,
                        batch_size=batch_size
                    )
                    
                    # Train Random Forest model
                    rf_model = forecaster.train_random_forest(
                        target_data,
                        n_estimators=n_estimators,
                        max_depth=max_depth
                    )
                    
                    # Train ensemble model
                    ensemble_model = forecaster.train_ensemble_model(
                        target_data, lstm_model, rf_model
                    )
                    
                    models[target] = {
                        'lstm': lstm_model,
                        'random_forest': rf_model,
                        'ensemble': ensemble_model
                    }
                    training_history[target] = lstm_history
                
                st.session_state.models = models
                st.session_state.training_history = training_history
                st.session_state.model_trained = True
                st.session_state.training_targets = targets
                
                st.success("✅ All models trained successfully!")
                
                # Display training results
                st.subheader("Training Results")
                
                for target in targets:
                    st.subheader(f"{target.upper()} Model Performance")
                    
                    # Training history plot
                    if target in training_history:
                        fig = visualizer.plot_training_history(training_history[target], target)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Model comparison
                    target_data = forecaster.prepare_target_data(processed_data, target)
                    model_comparison = forecaster.evaluate_models(models[target], target_data)
                    
                    comparison_df = pd.DataFrame(model_comparison).T
                    st.dataframe(comparison_df, width='stretch')
                
            except Exception as e:
                st.error(f"❌ Error during training: {str(e)}")

def forecasting_page(forecaster, visualizer):
    st.header("🔮 Forecasting & Results")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train models first.")
        return
    
    models = st.session_state.models
    processed_data = st.session_state.processed_data
    targets = st.session_state.training_targets
    
    # Forecasting configuration
    st.subheader("Forecast Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        forecast_days = st.slider("Forecast horizon (days)", 1, 3, 2)
        model_selection = st.selectbox(
            "Model for prediction:",
            ["Ensemble", "LSTM", "Random Forest"]
        )
    
    with col2:
        confidence_interval = st.slider("Confidence Interval (%)", 80, 99, 95)
        show_baseline = st.checkbox("Show baseline (persistence) model", value=True)
    
    # Generate forecasts button
    if st.button("🚀 Generate Forecasts", type="primary"):
        with st.spinner("Generating forecasts..."):
            try:
                predictions = {}
                baseline_predictions = {}
                
                for target in targets:
                    # Generate predictions
                    target_data = forecaster.prepare_target_data(processed_data, target)
                    
                    if model_selection.lower() == "ensemble":
                        pred = forecaster.predict_ensemble(
                            models[target], target_data, forecast_days
                        )
                    elif model_selection.lower() == "lstm":
                        pred = forecaster.predict_lstm(
                            models[target]['lstm'], target_data, forecast_days
                        )
                    else:
                        pred = forecaster.predict_random_forest(
                            models[target]['random_forest'], target_data, forecast_days
                        )
                    
                    predictions[target] = pred
                    
                    # Generate baseline predictions if requested
                    if show_baseline:
                        baseline_predictions[target] = forecaster.predict_baseline(
                            target_data, forecast_days
                        )
                
                st.session_state.predictions = predictions
                st.session_state.baseline_predictions = baseline_predictions if show_baseline else None
                st.session_state.predictions_made = True
                
                st.success("✅ Forecasts generated successfully!")
                
                # Display results
                display_forecast_results(
                    predictions, baseline_predictions if show_baseline else None,
                    targets, processed_data, visualizer, forecast_days
                )
                
            except Exception as e:
                st.error(f"❌ Error generating forecasts: {str(e)}")
    
    # Display existing results if available
    elif st.session_state.predictions_made:
        predictions = st.session_state.predictions
        baseline_predictions = st.session_state.get('baseline_predictions', None)
        
        display_forecast_results(
            predictions, baseline_predictions, targets, processed_data, visualizer, forecast_days
        )

def display_forecast_results(predictions, baseline_predictions, targets, processed_data, visualizer, forecast_days):
    st.subheader("📊 Forecast Results")
    
    # Safety thresholds (WHO guidelines)
    safety_thresholds = {
        'o3': 100,  # µg/m³
        'no2': 40   # µg/m³
    }
    
    for target in targets:
        st.subheader(f"{target.upper()} Forecast")
        
        # Create forecast visualization
        fig = visualizer.plot_forecast_results(
            predictions[target], 
            target,
            baseline_predictions[target] if baseline_predictions else None,
            safety_thresholds.get(target, None)
        )
        st.plotly_chart(fig, width='stretch')
        
        # Model evaluation metrics
        col1, col2, col3 = st.columns(3)
        
        if 'actual' in predictions[target] and 'predicted' in predictions[target]:
            metrics = calculate_metrics(
                predictions[target]['actual'], 
                predictions[target]['predicted']
            )
            
            with col1:
                st.metric("RMSE", f"{metrics['rmse']:.2f}")
            with col2:
                st.metric("MAE", f"{metrics['mae']:.2f}")
            with col3:
                st.metric("R²", f"{metrics['r2']:.3f}")
        
        # Safety alerts
        threshold = safety_thresholds.get(target, None)
        if threshold and 'predicted' in predictions[target]:
            predicted_values = predictions[target]['predicted']
            if isinstance(predicted_values, (list, np.ndarray)):
                max_predicted = max(predicted_values)
                if max_predicted > threshold:
                    st.error(
                        f"🚨 ALERT: Predicted {target.upper()} levels ({max_predicted:.1f} µg/m³) "
                        f"exceed WHO safety threshold ({threshold} µg/m³)"
                    )
                else:
                    st.success(
                        f"✅ Predicted {target.upper()} levels remain within safe limits"
                    )
    
    # Forecast summary table
    st.subheader("📋 Forecast Summary")
    summary_data = []
    
    for target in targets:
        if 'predicted' in predictions[target]:
            pred_values = predictions[target]['predicted']
            if isinstance(pred_values, (list, np.ndarray)) and len(pred_values) > 0:
                summary_data.append({
                    'Pollutant': target.upper(),
                    'Current Level': f"{pred_values[0]:.1f} µg/m³",
                    'Max Forecast': f"{max(pred_values):.1f} µg/m³",
                    'Avg Forecast': f"{np.mean(pred_values):.1f} µg/m³",
                    'Trend': '📈' if len(pred_values) > 1 and pred_values[-1] > pred_values[0] else '📉'
                })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, width='stretch')
        
        # Data Export Section
        st.subheader("📥 Export Forecast Data")
        col1, col2, col3 = st.columns(3)
        
        # Generate export data once for all formats
        export_csv = export_predictions_to_csv(predictions, targets)
        export_json = export_predictions_to_json(predictions, targets)
        export_summary = export_summary_report(predictions, targets, summary_df)
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M')
        
        with col1:
            # Direct CSV download
            st.download_button(
                label="📊 Download CSV Report",
                data=export_csv,
                file_name=f"pollutant_forecast_{timestamp}.csv",
                mime="text/csv",
                help="Download forecast data as CSV file"
            )
        
        with col2:
            # Direct JSON download
            st.download_button(
                label="📋 Download JSON Report",
                data=export_json,
                file_name=f"pollutant_forecast_{timestamp}.json",
                mime="application/json",
                help="Download forecast data as JSON file"
            )
        
        with col3:
            # Direct text summary download
            st.download_button(
                label="📈 Download Summary Report",
                data=export_summary,
                file_name=f"forecast_summary_{timestamp}.txt",
                mime="text/plain",
                help="Download formatted summary report"
            )

if __name__ == "__main__":
    main()
