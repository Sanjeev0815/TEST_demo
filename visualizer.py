import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class Visualizer:
    """Handles all visualization tasks for the air pollutant forecasting dashboard."""
    
    def __init__(self):
        self.colors = {
            'o3': '#FF6B6B',
            'no2': '#4ECDC4',
            'actual': '#2E86AB',
            'predicted': '#A23B72',
            'baseline': '#F18F01',
            'threshold': '#C73E1D'
        }
    
    def plot_pollutant_overview(self, data):
        """Create overview plots of pollutant concentrations."""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Ozone (O₃) Concentrations', 'Nitrogen Dioxide (NO₂) Concentrations'),
            shared_xaxes=True
        )
        
        # Prepare datetime column
        if 'datetime' in data.columns:
            x_axis = pd.to_datetime(data['datetime'])
        else:
            x_axis = data.index
        
        # O3 plot
        if 'o3' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=data['o3'],
                    mode='lines',
                    name='O₃',
                    line=dict(color=self.colors['o3'], width=2)
                ),
                row=1, col=1
            )
        
        # NO2 plot
        if 'no2' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=data['no2'],
                    mode='lines',
                    name='NO₂',
                    line=dict(color=self.colors['no2'], width=2)
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            title='Air Pollutant Concentration Overview',
            height=600,
            showlegend=True
        )
        
        fig.update_yaxes(title_text="Concentration (µg/m³)", row=1, col=1)
        fig.update_yaxes(title_text="Concentration (µg/m³)", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        
        return fig
    
    def plot_processed_data(self, processed_data):
        """Visualize processed data with train/test split."""
        try:
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Training Data', 'Test Data'),
                shared_xaxes=True
            )
            
            # Training data
            if 'y_train' in processed_data and not processed_data['y_train'].empty:
                y_train = processed_data['y_train']
                train_index = y_train.index
                
                for i, col in enumerate(y_train.columns):
                    color = self.colors.get(col, f'hsl({i*60}, 70%, 50%)')
                    fig.add_trace(
                        go.Scatter(
                            x=train_index,
                            y=y_train[col],
                            mode='lines',
                            name=f'{col.upper()} (Train)',
                            line=dict(color=color, width=2)
                        ),
                        row=1, col=1
                    )
            
            # Test data
            if 'y_test' in processed_data and not processed_data['y_test'].empty:
                y_test = processed_data['y_test']
                test_index = y_test.index
                
                for i, col in enumerate(y_test.columns):
                    color = self.colors.get(col, f'hsl({i*60}, 70%, 50%)')
                    fig.add_trace(
                        go.Scatter(
                            x=test_index,
                            y=y_test[col],
                            mode='lines',
                            name=f'{col.upper()} (Test)',
                            line=dict(color=color, width=2, dash='dash')
                        ),
                        row=2, col=1
                    )
            
            fig.update_layout(
                title='Processed Data: Train/Test Split',
                height=600,
                showlegend=True
            )
            
            fig.update_yaxes(title_text="Concentration (µg/m³)")
            fig.update_xaxes(title_text="Date", row=2, col=1)
            
            return fig
            
        except Exception as e:
            # Return a simple error plot if something goes wrong
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating visualization: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(size=16, color="red")
            )
            fig.update_layout(
                title='Processed Data Visualization Error',
                height=400
            )
            return fig
    
    def plot_training_history(self, history, target):
        """Plot training history for neural network models."""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Model Loss', 'Model MAE'),
            shared_xaxes=True
        )
        
        epochs = list(range(1, len(history.history['loss']) + 1))
        
        # Loss plot
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=history.history['loss'],
                mode='lines',
                name='Training Loss',
                line=dict(color='#FF6B6B', width=2)
            ),
            row=1, col=1
        )
        
        if 'val_loss' in history.history:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=history.history['val_loss'],
                    mode='lines',
                    name='Validation Loss',
                    line=dict(color='#4ECDC4', width=2)
                ),
                row=1, col=1
            )
        
        # MAE plot
        if 'mae' in history.history:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=history.history['mae'],
                    mode='lines',
                    name='Training MAE',
                    line=dict(color='#FF6B6B', width=2, dash='dot')
                ),
                row=1, col=2
            )
        
        if 'val_mae' in history.history:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=history.history['val_mae'],
                    mode='lines',
                    name='Validation MAE',
                    line=dict(color='#4ECDC4', width=2, dash='dot')
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            title=f'Training History - {target.upper()}',
            height=400,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Epoch")
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="MAE", row=1, col=2)
        
        return fig
    
    def plot_forecast_results(self, predictions, target, baseline_predictions=None, safety_threshold=None):
        """Plot forecast results with actual vs predicted values."""
        fig = go.Figure()
        
        # Create time index for plotting
        n_historical = len(predictions.get('actual', []))
        n_future = len(predictions.get('future_predictions', []))
        
        # Historical data
        if 'actual' in predictions and 'predicted' in predictions:
            historical_times = list(range(n_historical))
            
            # Actual values
            fig.add_trace(
                go.Scatter(
                    x=historical_times,
                    y=predictions['actual'],
                    mode='lines',
                    name='Actual',
                    line=dict(color=self.colors['actual'], width=3)
                )
            )
            
            # Predicted values
            fig.add_trace(
                go.Scatter(
                    x=historical_times,
                    y=predictions['predicted'],
                    mode='lines',
                    name='Predicted',
                    line=dict(color=self.colors['predicted'], width=2, dash='dash')
                )
            )
        
        # Future predictions
        if 'future_predictions' in predictions and predictions['future_predictions']:
            future_times = list(range(n_historical, n_historical + n_future))
            
            fig.add_trace(
                go.Scatter(
                    x=future_times,
                    y=predictions['future_predictions'],
                    mode='lines+markers',
                    name='Future Forecast',
                    line=dict(color=self.colors['predicted'], width=3),
                    marker=dict(size=8)
                )
            )
            
            # Add vertical line to separate historical and future
            fig.add_vline(
                x=n_historical-0.5,
                line_dash="dot",
                line_color="gray",
                annotation_text="Forecast Start"
            )
        
        # Baseline predictions
        if baseline_predictions:
            if 'predicted' in baseline_predictions:
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(baseline_predictions['predicted']))),
                        y=baseline_predictions['predicted'],
                        mode='lines',
                        name='Baseline (Persistence)',
                        line=dict(color=self.colors['baseline'], width=2, dash='dot'),
                        opacity=0.7
                    )
                )
        
        # Safety threshold
        if safety_threshold:
            fig.add_hline(
                y=safety_threshold,
                line_dash="dash",
                line_color=self.colors['threshold'],
                annotation_text=f"WHO Safety Threshold ({safety_threshold} µg/m³)"
            )
        
        fig.update_layout(
            title=f'{target.upper()} Concentration Forecast',
            xaxis_title='Time Steps',
            yaxis_title='Concentration (µg/m³)',
            height=500,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
    
    def plot_model_comparison(self, model_results):
        """Create comparison plot of different models' performance."""
        models = list(model_results.keys())
        metrics = ['RMSE', 'MAE', 'R²']
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=metrics,
            shared_yaxes=False
        )
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        for i, metric in enumerate(metrics):
            values = [model_results[model][metric] for model in models]
            
            fig.add_trace(
                go.Bar(
                    x=models,
                    y=values,
                    name=metric,
                    marker_color=colors[i % len(colors)]
                ),
                row=1, col=i+1
            )
        
        fig.update_layout(
            title='Model Performance Comparison',
            height=400,
            showlegend=False
        )
        
        return fig
    
    def plot_feature_importance(self, model, feature_names, top_n=15):
        """Plot feature importance for tree-based models."""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            
            # Create DataFrame for easier handling
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=True).tail(top_n)
            
            fig = go.Figure(go.Bar(
                x=importance_df['importance'],
                y=importance_df['feature'],
                orientation='h',
                marker_color='#4ECDC4'
            ))
            
            fig.update_layout(
                title=f'Top {top_n} Feature Importances',
                xaxis_title='Importance',
                yaxis_title='Features',
                height=500
            )
            
            return fig
        
        return None
    
    def create_correlation_heatmap(self, data):
        """Create correlation heatmap of variables."""
        # Select numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        correlation_matrix = data[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Variable Correlation Matrix',
            height=600,
            width=600
        )
        
        return fig
