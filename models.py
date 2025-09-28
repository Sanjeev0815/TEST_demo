import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# TensorFlow imports with fallback
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    keras = None
    layers = None
    print("TensorFlow not available. LSTM/GRU models will not be functional.")

class PollutantForecaster:
    """Main class for training and using air pollutant forecasting models."""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.sequence_length = 24  # 24 hours for hourly data
    
    def prepare_target_data(self, processed_data, target):
        """Prepare data for a specific target variable."""
        if target not in processed_data['y_train'].columns:
            raise ValueError(f"Target {target} not found in training data")
        
        # Extract target-specific data
        y_train = processed_data['y_train'][target]
        y_test = processed_data['y_test'][target]
        
        return {
            'X_train': processed_data['X_train'],
            'X_test': processed_data['X_test'],
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': processed_data['feature_names']
        }
    
    def train_lstm_model(self, target_data, model_type='LSTM', units=64, num_layers=2,
                        dropout_rate=0.2, epochs=20, batch_size=32):
        """
        Train LSTM or GRU model for time series forecasting.
        """
        if not TENSORFLOW_AVAILABLE:
            raise RuntimeError("TensorFlow is not available. Cannot train LSTM/GRU models.")
            
        # Prepare sequence data
        X_train_seq, y_train_seq = self._create_sequences(
            target_data['X_train'], target_data['y_train']
        )
        X_test_seq, y_test_seq = self._create_sequences(
            target_data['X_test'], target_data['y_test']
        )
        
        if len(X_train_seq) == 0:
            raise ValueError("Not enough data to create sequences. Try reducing sequence length.")
        
        # Build model
        model = keras.Sequential()
        
        # First layer
        if model_type.upper() == 'LSTM':
            model.add(layers.LSTM(
                units,
                return_sequences=(num_layers > 1),
                input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])
            ))
        else:  # GRU
            model.add(layers.GRU(
                units,
                return_sequences=(num_layers > 1),
                input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])
            ))
        
        model.add(layers.Dropout(dropout_rate))
        
        # Additional layers
        for i in range(1, num_layers):
            if model_type.upper() == 'LSTM':
                model.add(layers.LSTM(units, return_sequences=(i < num_layers - 1)))
            else:
                model.add(layers.GRU(units, return_sequences=(i < num_layers - 1)))
            model.add(layers.Dropout(dropout_rate))
        
        # Output layer
        model.add(layers.Dense(1))
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        # Train model
        history = model.fit(
            X_train_seq, y_train_seq,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test_seq, y_test_seq),
            verbose=0,
            shuffle=False  # Important for time series
        )
        
        return model, history
    
    def train_random_forest(self, target_data, n_estimators=100, max_depth=10):
        """Train Random Forest model for meteorological feature regression."""
        
        X_train = target_data['X_train']
        y_train = target_data['y_train']
        
        # Initialize and train Random Forest
        rf_model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        
        rf_model.fit(X_train, y_train)
        
        return rf_model
    
    def train_ensemble_model(self, target_data, lstm_model, rf_model):
        """Train ensemble model combining LSTM and Random Forest predictions."""
        
        # Get predictions from both models
        lstm_pred_train = self._predict_lstm_single(lstm_model, target_data['X_train'])
        rf_pred_train = rf_model.predict(target_data['X_train'])
        
        # Align predictions (LSTM predictions are shorter due to sequence requirement)
        min_length = min(len(lstm_pred_train), len(rf_pred_train))
        lstm_pred_train = lstm_pred_train[-min_length:]
        rf_pred_train = rf_pred_train[-min_length:]
        y_train_aligned = target_data['y_train'].iloc[-min_length:]
        
        # Combine predictions as features for ensemble
        ensemble_features = np.column_stack([lstm_pred_train, rf_pred_train])
        
        # Train simple linear combiner
        from sklearn.linear_model import LinearRegression
        ensemble_model = LinearRegression()
        ensemble_model.fit(ensemble_features, y_train_aligned)
        
        return {
            'ensemble_combiner': ensemble_model,
            'lstm_model': lstm_model,
            'rf_model': rf_model
        }
    
    def _create_sequences(self, X, y):
        """Create sequences for LSTM/GRU training."""
        X_sequences = []
        y_sequences = []
        
        for i in range(self.sequence_length, len(X)):
            X_sequences.append(X.iloc[i-self.sequence_length:i].values)
            y_sequences.append(y.iloc[i])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def _predict_lstm_single(self, model, X):
        """Get predictions from LSTM model."""
        X_seq, _ = self._create_sequences(X, pd.Series(np.zeros(len(X))))
        if len(X_seq) == 0:
            return np.array([])
        predictions = model.predict(X_seq, verbose=0)
        return predictions.flatten()
    
    def predict_lstm(self, model, target_data, forecast_days=1):
        """Generate predictions using LSTM model."""
        X_test = target_data['X_test']
        y_test = target_data['y_test']
        
        # Get historical predictions
        test_predictions = self._predict_lstm_single(model, X_test)
        
        # Align with actual values
        min_length = min(len(test_predictions), len(y_test))
        test_predictions = test_predictions[-min_length:]
        y_test_aligned = y_test.iloc[-min_length:]
        
        # Generate future predictions (simplified - using last available data)
        future_predictions = []
        last_sequence = X_test.iloc[-self.sequence_length:].values.reshape(1, self.sequence_length, -1)
        
        for day in range(forecast_days):
            next_pred = model.predict(last_sequence, verbose=0)[0, 0]
            future_predictions.append(next_pred)
            
            # Update sequence for next prediction (simplified)
            # In practice, you'd need to update with actual meteorological forecasts
        
        return {
            'historical_actual': y_test_aligned.values,
            'historical_predicted': test_predictions,
            'future_predictions': future_predictions,
            'actual': y_test_aligned.values,
            'predicted': test_predictions
        }
    
    def predict_random_forest(self, model, target_data, forecast_days=1):
        """Generate predictions using Random Forest model."""
        X_test = target_data['X_test']
        y_test = target_data['y_test']
        
        # Historical predictions
        test_predictions = model.predict(X_test)
        
        # Future predictions (using last available features)
        future_predictions = []
        for day in range(forecast_days):
            # Use last available features for future prediction
            last_features = X_test.iloc[-1:].values
            future_pred = model.predict(last_features)[0]
            future_predictions.append(future_pred)
        
        return {
            'historical_actual': y_test.values,
            'historical_predicted': test_predictions,
            'future_predictions': future_predictions,
            'actual': y_test.values,
            'predicted': test_predictions
        }
    
    def predict_ensemble(self, models, target_data, forecast_days=1):
        """Generate predictions using ensemble model."""
        lstm_model = models['ensemble']['lstm_model']
        rf_model = models['ensemble']['rf_model']
        ensemble_combiner = models['ensemble']['ensemble_combiner']
        
        X_test = target_data['X_test']
        y_test = target_data['y_test']
        
        # Get predictions from both models
        lstm_pred = self._predict_lstm_single(lstm_model, X_test)
        rf_pred = rf_model.predict(X_test)
        
        # Align predictions
        min_length = min(len(lstm_pred), len(rf_pred))
        lstm_pred = lstm_pred[-min_length:]
        rf_pred = rf_pred[-min_length:]
        y_test_aligned = y_test.iloc[-min_length:]
        
        # Combine predictions
        ensemble_features = np.column_stack([lstm_pred, rf_pred])
        ensemble_pred = ensemble_combiner.predict(ensemble_features)
        
        # Future predictions
        future_predictions = []
        for day in range(forecast_days):
            # Get future predictions from individual models
            last_sequence = X_test.iloc[-self.sequence_length:].values.reshape(1, self.sequence_length, -1)
            lstm_future = lstm_model.predict(last_sequence, verbose=0)[0, 0]
            rf_future = rf_model.predict(X_test.iloc[-1:].values)[0]
            
            # Combine future predictions
            future_features = np.array([[lstm_future, rf_future]])
            ensemble_future = ensemble_combiner.predict(future_features)[0]
            future_predictions.append(ensemble_future)
        
        return {
            'historical_actual': y_test_aligned.values,
            'historical_predicted': ensemble_pred,
            'future_predictions': future_predictions,
            'actual': y_test_aligned.values,
            'predicted': ensemble_pred
        }
    
    def predict_baseline(self, target_data, forecast_days=1):
        """Generate baseline predictions (persistence model)."""
        y_test = target_data['y_test']
        
        # Persistence model: assume pollutant level remains constant
        last_value = y_test.iloc[-1]
        
        baseline_pred = np.full(len(y_test), last_value)
        future_predictions = [last_value] * forecast_days
        
        return {
            'historical_actual': y_test.values,
            'historical_predicted': baseline_pred,
            'future_predictions': future_predictions,
            'actual': y_test.values,
            'predicted': baseline_pred
        }
    
    def evaluate_models(self, models, target_data):
        """Evaluate all models and return comparison metrics."""
        X_test = target_data['X_test']
        y_test = target_data['y_test']
        
        results = {}
        
        # Evaluate LSTM
        if 'lstm' in models:
            lstm_pred = self._predict_lstm_single(models['lstm'], X_test)
            if len(lstm_pred) > 0:
                y_test_lstm = y_test.iloc[-len(lstm_pred):]
                results['LSTM'] = {
                    'RMSE': np.sqrt(mean_squared_error(y_test_lstm, lstm_pred)),
                    'MAE': mean_absolute_error(y_test_lstm, lstm_pred),
                    'R²': r2_score(y_test_lstm, lstm_pred)
                }
        
        # Evaluate Random Forest
        if 'random_forest' in models:
            rf_pred = models['random_forest'].predict(X_test)
            results['Random Forest'] = {
                'RMSE': np.sqrt(mean_squared_error(y_test, rf_pred)),
                'MAE': mean_absolute_error(y_test, rf_pred),
                'R²': r2_score(y_test, rf_pred)
            }
        
        # Evaluate Ensemble
        if 'ensemble' in models:
            lstm_pred = self._predict_lstm_single(models['ensemble']['lstm_model'], X_test)
            rf_pred = models['ensemble']['rf_model'].predict(X_test)
            
            if len(lstm_pred) > 0:
                min_length = min(len(lstm_pred), len(rf_pred))
                lstm_pred = lstm_pred[-min_length:]
                rf_pred = rf_pred[-min_length:]
                y_test_aligned = y_test.iloc[-min_length:]
                
                ensemble_features = np.column_stack([lstm_pred, rf_pred])
                ensemble_pred = models['ensemble']['ensemble_combiner'].predict(ensemble_features)
                
                results['Ensemble'] = {
                    'RMSE': np.sqrt(mean_squared_error(y_test_aligned, ensemble_pred)),
                    'MAE': mean_absolute_error(y_test_aligned, ensemble_pred),
                    'R²': r2_score(y_test_aligned, ensemble_pred)
                }
        
        return results
