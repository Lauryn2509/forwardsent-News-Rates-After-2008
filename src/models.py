"""
Machine Learning Models Module for NLP x Finance Project

This module contains model training, evaluation, and visualization functions
for predicting Federal Reserve rates using NLP features.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Tuple, Optional, List
import os

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not installed. Install with: pip install xgboost")


class FedRatePredictor:
    """
    A wrapper class for training and evaluating Fed rate prediction models.
    Supports XGBoost, Ridge Regression, and SVR.
    """

    def __init__(self, model_type: str = "xgboost", model_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the predictor with specified model type and parameters.

        Args:
            model_type: Type of model ('xgboost', 'ridge', 'svr')
            model_params: Dictionary of model-specific parameters
        """
        self.model_type = model_type.lower()
        self.scaler = StandardScaler()
        self.use_scaling = False
        
        # Default parameters for each model type
        if self.model_type == "xgboost":
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost is not installed. Please install with: pip install xgboost")
            default_params = {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "random_state": 42,
                "reg_alpha": 0.1,  # L1 regularization
                "reg_lambda": 1.0,  # L2 regularization
            }
        elif self.model_type == "ridge":
            default_params = {
                "alpha": 1.0,
                "random_state": 42,
                "max_iter": 1000,
            }
            self.use_scaling = True
        elif self.model_type == "svr":
            default_params = {
                "kernel": "rbf",
                "C": 1.0,
                "gamma": "scale",
                "epsilon": 0.1,
            }
            self.use_scaling = True
        else:
            raise ValueError(f"Unsupported model type: {model_type}. Use 'xgboost', 'ridge', or 'svr'")

        if model_params:
            default_params.update(model_params)

        # Initialize the appropriate model
        if self.model_type == "xgboost":
            self.model = xgb.XGBRegressor(**default_params)
        elif self.model_type == "ridge":
            self.model = Ridge(**default_params)
        elif self.model_type == "svr":
            self.model = SVR(**default_params)
            
        self.is_trained = False

    def train_and_evaluate(
        self,
        X: np.ndarray,
        y: pd.Series,
        dates: pd.Series,
        label: str,
        color: str = "blue",
    ) -> Dict[str, Any]:
        """
        Train model and evaluate performance with visualizations.

        Args:
            X: Feature matrix
            y: Target variable (Fed rates)
            dates: Corresponding dates
            label: Model identifier (e.g., 'TF-IDF', 'SBERT')
            color: Color for visualizations

        Returns:
            dict: Evaluation metrics and results
        """
        print(f"  Training {label} model with {self.model_type.upper()}...")

        # Train-test split (maintaining temporal order consideration)
        X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
            X, y, dates, test_size=0.2, random_state=42
        )

        # Apply scaling if needed (for Ridge and SVR)
        if self.use_scaling:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test

        # Model training
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True

        # Predictions
        y_pred = self.model.predict(X_test_scaled)

        # Evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"    {label} Results ({self.model_type.upper()}):")
        print(f"       • MSE: {mse:.4f}")
        print(f"       • MAE: {mae:.4f}")
        print(f"       • R²:  {r2:.4f}")

        # Generate visualizations
        model_label = f"{label} ({self.model_type.upper()})"
        self._create_scatter_plot(y_test, y_pred, model_label, color, mse, mae, r2)
        self._create_time_series_plot(dates_test, y_test, y_pred, model_label, color)

        return {
            "mse": mse,
            "mae": mae,
            "r2": r2,
            "model": self.model,
            "model_type": self.model_type,
            "predictions": y_pred,
            "actual": y_test,
            "dates_test": dates_test,
        }

    def _create_scatter_plot(
        self,
        y_test: pd.Series,
        y_pred: np.ndarray,
        label: str,
        color: str,
        mse: float,
        mae: float,
        r2: float,
    ):
        """Create scatter plot of predictions vs reality."""
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.7, color=color, s=50, label="Predictions")
        plt.plot(
            [y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            "r--",
            alpha=0.8,
            linewidth=2,
            label="Perfect Prediction Line",
        )

        plt.xlabel("Actual Fed Rate (%)", fontsize=12)
        plt.ylabel("Predicted Fed Rate (%)", fontsize=12)
        plt.title(
            f"Fed Rate Predictions vs Reality - {label} Model",
            fontsize=14,
            fontweight="bold",
        )
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)

        # Add performance metrics to plot
        textstr = f"MSE: {mse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}"
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
        plt.text(
            0.05,
            0.95,
            textstr,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
        )

        plt.tight_layout()

        # Ensure figures directory exists
        os.makedirs("figures", exist_ok=True)
        filename = (
            f"figures/scatter_{label.lower().replace(' ', '_').replace('-', '_')}.png"
        )
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()

    def _create_time_series_plot(
        self,
        dates_test: pd.Series,
        y_test: pd.Series,
        y_pred: np.ndarray,
        label: str,
        color: str,
    ):
        """Create time series plot comparing actual vs predicted rates."""
        df_visu = pd.DataFrame(
            {"date": dates_test, "actual_rate": y_test, "predicted_rate": y_pred}
        ).sort_values("date")

        plt.figure(figsize=(12, 7))
        plt.plot(
            df_visu["date"],
            df_visu["actual_rate"],
            label="Actual Fed Rate",
            linewidth=2.5,
            color="darkblue",
            marker="o",
            markersize=4,
        )
        plt.plot(
            df_visu["date"],
            df_visu["predicted_rate"],
            label=f"Predicted ({label})",
            linewidth=2,
            color=color,
            linestyle="--",
            marker="s",
            markersize=4,
        )

        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Fed Rate (%)", fontsize=12)
        plt.title(
            f"Fed Rate Time Series: Actual vs Predicted ({label})",
            fontsize=14,
            fontweight="bold",
        )
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.xticks(rotation=45)

        # Add shaded area for prediction error
        plt.fill_between(
            df_visu["date"],
            df_visu["actual_rate"],
            df_visu["predicted_rate"],
            alpha=0.2,
            color=color,
            label="Prediction Error",
        )

        plt.tight_layout()
        filename = (
            f"figures/curve_{label.lower().replace(' ', '_').replace('-', '_')}.png"
        )
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.

        Args:
            X: Feature matrix

        Returns:
            np.ndarray: Predictions

        Raises:
            RuntimeError: If model hasn't been trained yet
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")

        return self.model.predict(X)

    def get_feature_importance(
        self, feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get feature importance from XGBoost models or coefficients from linear models.

        Args:
            feature_names: List of feature names (optional)

        Returns:
            pd.DataFrame: Feature importance/coefficients dataframe
        """
        if not self.is_trained:
            raise RuntimeError(
                "Model must be trained before getting feature importance"
            )

        if self.model_type == "xgboost":
            importance = self.model.feature_importances_
        elif self.model_type == "ridge":
            importance = np.abs(self.model.coef_)  # Absolute coefficients
        elif self.model_type == "svr":
            # SVR doesn't have feature importance directly
            print("Warning: SVR doesn't provide feature importance. Consider using XGBoost or Ridge for interpretability.")
            return pd.DataFrame({"feature": ["N/A"], "importance": [0.0]})

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance))]

        df_importance = pd.DataFrame(
            {"feature": feature_names, "importance": importance}
        ).sort_values("importance", ascending=False)

        return df_importance


class EnsemblePredictor:
    """
    Ensemble predictor that combines XGBoost, Ridge, and SVR models.
    """
    
    def __init__(self):
        """Initialize ensemble with the three optimized models."""
        self.models = {
            "xgboost": FedRatePredictor("xgboost"),
            "ridge": FedRatePredictor("ridge"),
            "svr": FedRatePredictor("svr")
        }
        self.weights = None
        self.is_trained = False
    
    def train_and_evaluate(
        self,
        X: np.ndarray,
        y: pd.Series,
        dates: pd.Series,
        label: str,
        color: str = "purple",
    ) -> Dict[str, Any]:
        """
        Train ensemble and evaluate performance.
        """
        print(f"  Training {label} ensemble (XGBoost + Ridge + SVR)...")
        
        # Train individual models and collect predictions
        individual_results = {}
        validation_predictions = []
        
        # Train-test split (same as individual models for fair comparison)
        X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
            X, y, dates, test_size=0.2, random_state=42
        )
        
        for model_name, model in self.models.items():
            # Train individual model
            model_results = model.train_and_evaluate(
                X, y, dates, f"{label}-{model_name}", "gray"
            )
            individual_results[model_name] = model_results
            
            # Get predictions on test set
            if model.use_scaling:
                X_test_scaled = model.scaler.transform(X_test)
            else:
                X_test_scaled = X_test
            validation_predictions.append(model.model.predict(X_test_scaled))
        
        # Simple average ensemble
        validation_predictions = np.array(validation_predictions)
        y_pred_ensemble = np.mean(validation_predictions, axis=0)
        
        # Evaluation metrics
        mse = mean_squared_error(y_test, y_pred_ensemble)
        mae = mean_absolute_error(y_test, y_pred_ensemble)
        r2 = r2_score(y_test, y_pred_ensemble)
        
        print(f"    {label} Ensemble Results:")
        print(f"       • MSE: {mse:.4f}")
        print(f"       • MAE: {mae:.4f}")
        print(f"       • R²:  {r2:.4f}")
        
        # Generate visualizations
        model_label = f"{label} (ENSEMBLE)"
        self._create_ensemble_plots(y_test, y_pred_ensemble, model_label, color, mse, mae, r2, dates_test)
        
        self.is_trained = True
        
        return {
            "mse": mse,
            "mae": mae,
            "r2": r2,
            "predictions": y_pred_ensemble,
            "actual": y_test,
            "dates_test": dates_test,
            "individual_results": individual_results,
        }
    
    def _create_ensemble_plots(self, y_test, y_pred, label, color, mse, mae, r2, dates_test):
        """Create plots for ensemble results."""
        # Scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.7, color=color, s=50, label="Ensemble Predictions")
        plt.plot(
            [y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            "r--",
            alpha=0.8,
            linewidth=2,
            label="Perfect Prediction Line",
        )
        plt.xlabel("Actual Fed Rate (%)", fontsize=12)
        plt.ylabel("Predicted Fed Rate (%)", fontsize=12)
        plt.title(f"Fed Rate Predictions vs Reality - {label}", fontsize=14, fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        
        textstr = f"MSE: {mse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}"
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment="top", bbox=props)
        
        plt.tight_layout()
        plt.savefig("figures/scatter_ensemble.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        # Time series plot
        df_visu = pd.DataFrame({
            "date": dates_test, 
            "actual_rate": y_test, 
            "predicted_rate": y_pred
        }).sort_values("date")
        
        plt.figure(figsize=(12, 7))
        plt.plot(df_visu["date"], df_visu["actual_rate"], label="Actual Fed Rate",
                linewidth=2.5, color="darkblue", marker="o", markersize=4)
        plt.plot(df_visu["date"], df_visu["predicted_rate"], label=f"Predicted ({label})",
                linewidth=2, color=color, linestyle="--", marker="s", markersize=4)
        
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Fed Rate (%)", fontsize=12)
        plt.title(f"Fed Rate Time Series: Actual vs Predicted ({label})", fontsize=14, fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.xticks(rotation=45)
        
        plt.fill_between(df_visu["date"], df_visu["actual_rate"], df_visu["predicted_rate"],
                        alpha=0.2, color=color, label="Prediction Error")
        
        plt.tight_layout()
        plt.savefig("figures/curve_ensemble.png", dpi=300, bbox_inches="tight")
        plt.close()


def compare_models(results_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Compare multiple model results and create a summary table.

    Args:
        results_dict: Dictionary with model names as keys and results as values

    Returns:
        pd.DataFrame: Comparison table
    """
    comparison_data = []

    for model_name, results in results_dict.items():
        comparison_data.append(
            {
                "Model": model_name,
                "MSE": results["mse"],
                "MAE": results["mae"],
                "R²": results["r2"],
            }
        )

    return pd.DataFrame(comparison_data)


def print_model_insights(comparison_df: pd.DataFrame):
    """
    Print business insights based on model comparison results.

    Args:
        comparison_df: DataFrame with model comparison results
    """
    print("KEY INSIGHTS")
    print("=" * 20)

    best_model = comparison_df.loc[comparison_df["R²"].idxmax(), "Model"]
    best_r2 = comparison_df["R²"].max()

    if best_r2 < 0:
        print("• All models show negative R² scores, indicating predictions")
        print("  are currently worse than using the mean Fed rate as prediction")
        print(f"• {best_model} performs relatively better with R² = {best_r2:.4f}")
        print("• The relationship between news sentiment and Fed rates appears")
        print("  more complex than captured by current feature engineering")
        print("• Future improvements: sentiment analysis, economic indicators,")
        print("  temporal modeling, and larger datasets")
    else:
        print(f"• {best_model} shows the best performance with R² = {best_r2:.4f}")
        print("• The model successfully captures some relationship between")
        print("  news sentiment and Fed rate changes")

    print()
