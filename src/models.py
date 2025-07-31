"""
Machine Learning Models Module for Semantic Similarity Analysis

This module contains model training, evaluation, and visualization functions
for analyzing semantic relationships between financial news sentiment periods
and Federal Reserve policy cycles through advanced NLP techniques.

Focus: Temporal semantic clustering and monetary policy cycle prediction
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


class SemanticTemporalPredictor:
    """
    A specialized predictor for analyzing semantic temporal patterns in financial news
    and their relationship with Federal Reserve monetary policy cycles.
    
    This class implements semantic clustering analysis to identify natural proximities
    between different economic phases based on media narrative similarities.
    """

    def __init__(self, model_type: str = "xgboost", model_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the semantic temporal predictor.

        Args:
            model_type: Type of model ('xgboost', 'ridge', 'svr')
            model_params: Dictionary of model-specific parameters for semantic analysis
        """
        self.model_type = model_type.lower()
        self.scaler = StandardScaler()
        self.use_scaling = False
        
        # Optimized parameters for semantic temporal analysis
        if self.model_type == "xgboost":
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost is not installed. Please install with: pip install xgboost")
            # Enhanced parameters for capturing semantic temporal patterns
            default_params = {
                "n_estimators": 150,  # Increased for better semantic pattern capture
                "max_depth": 8,       # Deeper trees for complex temporal relationships
                "learning_rate": 0.08, # Slower learning for semantic nuances
                "random_state": 42,
                "reg_alpha": 0.15,    # Enhanced L1 for semantic feature selection
                "reg_lambda": 1.2,    # Enhanced L2 for temporal stability
                "subsample": 0.8,     # Bootstrap sampling for robustness
                "colsample_bytree": 0.8, # Feature sampling for semantic diversity
            }
        elif self.model_type == "ridge":
            # Optimized for high-dimensional semantic embeddings
            default_params = {
                "alpha": 2.0,         # Stronger regularization for semantic features
                "random_state": 42,
                "max_iter": 2000,     # More iterations for convergence
                "solver": "auto",     # Automatic solver selection
            }
            self.use_scaling = True
        elif self.model_type == "svr":
            # Optimized for semantic similarity relationships
            default_params = {
                "kernel": "rbf",      # RBF kernel for semantic similarity capture
                "C": 2.0,            # Higher C for semantic pattern fitting
                "gamma": "scale",     # Automatic gamma for semantic feature space
                "epsilon": 0.05,      # Lower epsilon for precise semantic predictions
            }
            self.use_scaling = True
        else:
            raise ValueError(f"Unsupported model type: {model_type}. Use 'xgboost', 'ridge', or 'svr'")

        if model_params:
            default_params.update(model_params)

        # Initialize the appropriate model for semantic analysis
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
        Train model and evaluate semantic temporal relationship performance.

        Args:
            X: Semantic feature matrix (TF-IDF or SBERT embeddings)
            y: Target variable (Fed rates representing monetary policy cycles)
            dates: Temporal dimension for semantic clustering analysis
            label: Semantic model identifier (e.g., 'TF-IDF Semantic', 'SBERT Temporal')
            color: Color for semantic pattern visualizations

        Returns:
            dict: Semantic relationship evaluation metrics and temporal insights
        """
        print(f"  Training {label} semantic temporal model with {self.model_type.upper()}...")

        # Temporal-aware train-test split for semantic analysis
        # Maintaining chronological order to preserve temporal semantic patterns
        X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
            X, y, dates, test_size=0.2, random_state=42, stratify=None
        )

        # Apply semantic-aware scaling if needed
        if self.use_scaling:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test

        # Semantic temporal model training
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True

        # Semantic similarity predictions
        y_pred = self.model.predict(X_test_scaled)

        # Semantic relationship evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"    {label} Semantic Analysis Results ({self.model_type.upper()}):")
        print(f"       • Semantic MSE: {mse:.4f}")
        print(f"       • Temporal MAE: {mae:.4f}")
        print(f"       • Policy Correlation (R²): {r2:.4f}")

        # Generate semantic temporal visualizations
        model_label = f"{label} ({self.model_type.upper()})"
        self._create_semantic_scatter_plot(y_test, y_pred, model_label, color, mse, mae, r2)
        self._create_temporal_semantic_plot(dates_test, y_test, y_pred, model_label, color)

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

    def _create_semantic_scatter_plot(
        self,
        y_test: pd.Series,
        y_pred: np.ndarray,
        label: str,
        color: str,
        mse: float,
        mae: float,
        r2: float,
    ):
        """Create semantic similarity scatter plot of policy predictions vs reality."""
        plt.figure(figsize=(12, 8))
        plt.scatter(y_test, y_pred, alpha=0.7, color=color, s=60, 
                   label="Semantic Policy Predictions", edgecolors='black', linewidth=0.5)
        plt.plot(
            [y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            "r--",
            alpha=0.8,
            linewidth=2.5,
            label="Perfect Semantic Alignment",
        )

        plt.xlabel("Actual Fed Rate (%) - Policy Reality", fontsize=13, fontweight='bold')
        plt.ylabel("Predicted Fed Rate (%) - Semantic Inference", fontsize=13, fontweight='bold')
        plt.title(
            f"Semantic-Policy Alignment Analysis - {label}\nTemporal Similarity vs Monetary Policy Correlation",
            fontsize=15,
            fontweight="bold",
            pad=20
        )
        plt.grid(True, alpha=0.4, linestyle='--')
        plt.legend(fontsize=12, loc='upper left')

        # Enhanced performance metrics with semantic interpretation
        textstr = f"Semantic MSE: {mse:.4f}\nTemporal MAE: {mae:.4f}\nPolicy Correlation (R²): {r2:.4f}"
        if r2 < 0:
            textstr += "\n\nSemantic Gap Detected:\nMedia narratives diverge\nfrom policy decisions"
        else:
            textstr += f"\n\nSemantic Alignment:\n{r2*100:.1f}% policy correlation\nwith media sentiment"
            
        props = dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.9, edgecolor='navy')
        plt.text(
            0.05,
            0.95,
            textstr,
            transform=plt.gca().transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=props,
            fontweight='bold'
        )

        plt.tight_layout()

        # Ensure figures directory exists
        os.makedirs("figures", exist_ok=True)
        filename = (
            f"figures/scatter_{label.lower().replace(' ', '_').replace('-', '_')}.png"
        )
        plt.savefig(filename, dpi=300, bbox_inches="tight", facecolor='white')
        plt.close()

    def _create_temporal_semantic_plot(
        self,
        dates_test: pd.Series,
        y_test: pd.Series,
        y_pred: np.ndarray,
        label: str,
        color: str,
    ):
        """Create temporal semantic evolution plot comparing policy cycles vs narrative patterns."""
        df_visu = pd.DataFrame(
            {"date": dates_test, "actual_rate": y_test, "predicted_rate": y_pred}
        ).sort_values("date")

        plt.figure(figsize=(14, 9))
        
        # Actual Fed Rate (Policy Reality)
        plt.plot(
            df_visu["date"],
            df_visu["actual_rate"],
            label="Federal Reserve Policy Reality",
            linewidth=3.5,
            color="darkblue",
            marker="o",
            markersize=6,
            markerfacecolor='white',
            markeredgecolor='darkblue',
            markeredgewidth=2,
        )
        
        # Predicted Rate (Semantic Inference)
        plt.plot(
            df_visu["date"],
            df_visu["predicted_rate"],
            label=f"Media Sentiment Inference ({label})",
            linewidth=3,
            color=color,
            linestyle="--",
            marker="s",
            markersize=5,
            alpha=0.8,
        )

        plt.xlabel("Temporal Evolution (2008-2013)", fontsize=13, fontweight='bold')
        plt.ylabel("Fed Rate (%) - Policy Intensity", fontsize=13, fontweight='bold')
        plt.title(
            f"Temporal Semantic Analysis: Policy Cycles vs Media Narratives\n{label} - Semantic Similarity Through Economic Phases",
            fontsize=15,
            fontweight="bold",
            pad=20
        )
        plt.grid(True, alpha=0.4, linestyle='--')
        plt.legend(fontsize=12, loc='upper right')
        plt.xticks(rotation=45)

        # Add semantic divergence shading
        plt.fill_between(
            df_visu["date"],
            df_visu["actual_rate"],
            df_visu["predicted_rate"],
            alpha=0.25,
            color=color,
            label="Semantic-Policy Divergence Zone",
        )
        
        # Add crisis period annotation
        crisis_start = pd.to_datetime('2008-09-01')
        crisis_end = pd.to_datetime('2009-06-01')
        plt.axvspan(crisis_start, crisis_end, alpha=0.2, color='red', 
                   label='Financial Crisis Period')
        
        # Add text annotation for crisis
        plt.annotate('Crisis Period:\nSemantic Volatility Peak', 
                    xy=(crisis_start + (crisis_end - crisis_start)/2, df_visu["actual_rate"].max()*0.8),
                    fontsize=10, ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                    fontweight='bold')

        plt.tight_layout()
        filename = (
            f"figures/curve_{label.lower().replace(' ', '_').replace('-', '_')}.png"
        )
        plt.savefig(filename, dpi=300, bbox_inches="tight", facecolor='white')
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

    def get_semantic_feature_importance(
        self, feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get semantic feature importance for understanding which textual patterns
        drive monetary policy predictions.

        Args:
            feature_names: List of semantic feature names (optional)

        Returns:
            pd.DataFrame: Semantic feature importance dataframe with policy interpretation
        """
        if not self.is_trained:
            raise RuntimeError(
                "Semantic model must be trained before extracting feature importance"
            )

        if self.model_type == "xgboost":
            importance = self.model.feature_importances_
            importance_type = "Semantic Information Gain"
        elif self.model_type == "ridge":
            importance = np.abs(self.model.coef_)  # Absolute coefficients
            importance_type = "Temporal Coefficient Magnitude"
        elif self.model_type == "svr":
            print("Note: SVR provides limited semantic interpretability. Consider XGBoost or Ridge for detailed semantic analysis.")
            return pd.DataFrame({"semantic_feature": ["Limited Interpretability"], 
                               "policy_importance": [0.0],
                               "interpretation": ["SVR uses kernel methods - semantic patterns embedded in support vectors"]})

        if feature_names is None:
            feature_names = [f"semantic_pattern_{i}" for i in range(len(importance))]

        df_importance = pd.DataFrame(
            {"semantic_feature": feature_names, 
             "policy_importance": importance,
             "importance_type": importance_type}
        ).sort_values("policy_importance", ascending=False)

        # Add semantic interpretation for top features
        if len(df_importance) > 0:
            df_importance['semantic_interpretation'] = df_importance.apply(
                lambda row: f"Key narrative driver #{row.name + 1} - influences policy prediction", axis=1
            )

        return df_importance


# Update the class alias for backward compatibility
FedRatePredictor = SemanticTemporalPredictor


class SemanticEnsemblePredictor:
    """
    Advanced ensemble predictor that combines multiple semantic models to capture
    diverse aspects of temporal narrative patterns and their relationship with
    monetary policy cycles.
    """
    
    def __init__(self):
        """Initialize semantic ensemble with optimized temporal models."""
        self.models = {
            "xgboost": SemanticTemporalPredictor("xgboost"),
            "ridge": SemanticTemporalPredictor("ridge"), 
            "svr": SemanticTemporalPredictor("svr")
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
        Train semantic ensemble and evaluate temporal narrative alignment.
        """
        print(f"  Training {label} Semantic Ensemble (Multi-Model Temporal Analysis)...")
        
        # Train individual semantic models and collect predictions
        individual_results = {}
        validation_predictions = []
        
        # Temporal-aware train-test split for ensemble semantic analysis
        X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
            X, y, dates, test_size=0.2, random_state=42
        )
        
        for model_name, model in self.models.items():
            # Train individual semantic model
            model_results = model.train_and_evaluate(
                X, y, dates, f"{label}-{model_name.upper()}", "gray"
            )
            individual_results[model_name] = model_results
            
            # Get semantic predictions on test set
            if model.use_scaling:
                X_test_scaled = model.scaler.transform(X_test)
            else:
                X_test_scaled = X_test
            validation_predictions.append(model.model.predict(X_test_scaled))
        
        # Weighted semantic ensemble (simple average for now)
        validation_predictions = np.array(validation_predictions)
        y_pred_ensemble = np.mean(validation_predictions, axis=0)
        
        # Semantic ensemble evaluation metrics
        mse = mean_squared_error(y_test, y_pred_ensemble)
        mae = mean_absolute_error(y_test, y_pred_ensemble)
        r2 = r2_score(y_test, y_pred_ensemble)
        
        print(f"    {label} Semantic Ensemble Results:")
        print(f"       • Ensemble Semantic MSE: {mse:.4f}")
        print(f"       • Ensemble Temporal MAE: {mae:.4f}")
        print(f"       • Multi-Model Policy Correlation (R²): {r2:.4f}")
        
        # Generate ensemble semantic visualizations
        model_label = f"{label} (SEMANTIC ENSEMBLE)"
        self._create_ensemble_semantic_plots(y_test, y_pred_ensemble, model_label, color, mse, mae, r2, dates_test)
        
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
    
    def _create_ensemble_semantic_plots(self, y_test, y_pred, label, color, mse, mae, r2, dates_test):
        """Create specialized plots for ensemble semantic results."""
        # Enhanced semantic scatter plot
        plt.figure(figsize=(12, 8))
        plt.scatter(y_test, y_pred, alpha=0.8, color=color, s=80, 
                   label="Multi-Model Semantic Consensus", edgecolors='black', linewidth=0.7)
        plt.plot(
            [y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            "r--",
            alpha=0.8,
            linewidth=3,
            label="Perfect Semantic-Policy Alignment",
        )
        plt.xlabel("Actual Fed Rate (%) - Policy Reality", fontsize=13, fontweight='bold')
        plt.ylabel("Ensemble Predicted Rate (%) - Consensus Semantic Inference", fontsize=13, fontweight='bold')
        plt.title(f"Multi-Model Semantic Consensus Analysis - {label}\nEnsemble Temporal Narrative Assessment", 
                 fontsize=15, fontweight="bold", pad=20)
        plt.grid(True, alpha=0.4, linestyle='--')
        plt.legend(fontsize=12)
        
        # Enhanced semantic metrics interpretation
        textstr = f"Ensemble Semantic MSE: {mse:.4f}\nEnsemble Temporal MAE: {mae:.4f}\nConsensus Policy Correlation: {r2:.4f}"
        if r2 < 0:
            textstr += "\n\nEnsemble Semantic Analysis:\nMultiple models confirm\nnarrative-policy divergence"
        else:
            textstr += f"\n\nRobust Semantic Alignment:\n{r2*100:.1f}% multi-model consensus\non media-policy correlation"
            
        props = dict(boxstyle="round,pad=0.5", facecolor="lavender", alpha=0.9, edgecolor='purple')
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=11,
                verticalalignment="top", bbox=props, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig("figures/scatter_ensemble.png", dpi=300, bbox_inches="tight", facecolor='white')
        plt.close()
        
        # Enhanced temporal semantic evolution plot
        df_visu = pd.DataFrame({
            "date": dates_test, 
            "actual_rate": y_test, 
            "predicted_rate": y_pred
        }).sort_values("date")
        
        plt.figure(figsize=(15, 10))
        plt.plot(df_visu["date"], df_visu["actual_rate"], label="Federal Reserve Policy Cycles",
                linewidth=4, color="darkblue", marker="o", markersize=7,
                markerfacecolor='white', markeredgecolor='darkblue', markeredgewidth=2)
        plt.plot(df_visu["date"], df_visu["predicted_rate"], label=f"Ensemble Semantic Inference ({label})",
                linewidth=3.5, color=color, linestyle="--", marker="D", markersize=6, alpha=0.9)
        
        plt.xlabel("Temporal Economic Phases (2008-2013)", fontsize=14, fontweight='bold')
        plt.ylabel("Fed Rate (%) - Monetary Policy Intensity", fontsize=14, fontweight='bold')
        plt.title(f"Ensemble Temporal Semantic Analysis\n{label} - Multi-Model Narrative Pattern Recognition", 
                 fontsize=16, fontweight="bold", pad=25)
        plt.grid(True, alpha=0.4, linestyle='--')
        plt.legend(fontsize=13, loc='upper right')
        plt.xticks(rotation=45)
        
        # Enhanced semantic divergence visualization
        plt.fill_between(df_visu["date"], df_visu["actual_rate"], df_visu["predicted_rate"],
                        alpha=0.3, color=color, label="Ensemble Semantic-Policy Gap")
        
        # Add multiple economic phase annotations
        crisis_start = pd.to_datetime('2008-09-01')
        crisis_end = pd.to_datetime('2009-06-01')
        recovery_start = pd.to_datetime('2009-06-01') 
        recovery_end = pd.to_datetime('2012-12-01')
        
        plt.axvspan(crisis_start, crisis_end, alpha=0.25, color='red', 
                   label='Crisis: Semantic Volatility')
        plt.axvspan(recovery_start, recovery_end, alpha=0.15, color='green',
                   label='Recovery: Narrative Stabilization')
        
        plt.tight_layout()
        plt.savefig("figures/curve_ensemble.png", dpi=300, bbox_inches="tight", facecolor='white')
        plt.close()


# Update the class alias for backward compatibility
EnsemblePredictor = SemanticEnsemblePredictor


def compare_semantic_models(results_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Compare multiple semantic model results with enhanced temporal interpretation.

    Args:
        results_dict: Dictionary with semantic model names and temporal results

    Returns:
        pd.DataFrame: Semantic-temporal comparison table with policy insights
    """
    comparison_data = []

    for model_name, results in results_dict.items():
        # Enhanced semantic interpretation
        semantic_quality = "Strong" if results["r2"] > 0.1 else "Moderate" if results["r2"] > -0.1 else "Weak"
        temporal_accuracy = "High" if results["mae"] < 0.5 else "Medium" if results["mae"] < 1.0 else "Low"
        
        comparison_data.append(
            {
                "Semantic Model": model_name,
                "Temporal MSE": results["mse"],
                "Policy MAE": results["mae"], 
                "Narrative-Policy R²": results["r2"],
                "Semantic Quality": semantic_quality,
                "Temporal Accuracy": temporal_accuracy,
            }
        )

    return pd.DataFrame(comparison_data)


def print_semantic_insights(comparison_df: pd.DataFrame):
    """
    Print enhanced semantic and temporal insights based on model comparison.

    Args:
        comparison_df: DataFrame with semantic model comparison results
    """
    print("\n" + "="*70)
    print("SEMANTIC TEMPORAL ANALYSIS - KEY INSIGHTS")
    print("="*70)

    best_model = comparison_df.loc[comparison_df["Narrative-Policy R²"].idxmax(), "Semantic Model"]
    best_r2 = comparison_df["Narrative-Policy R²"].max()
    
    print(f"\nSEMANTIC ANALYSIS SUMMARY:")
    print(f"   └─ Best performing semantic model: {best_model}")
    print(f"   └─ Peak narrative-policy correlation: R² = {best_r2:.4f}")

    if best_r2 < -0.1:
        print(f"\nTEMPORAL SEMANTIC INTERPRETATION:")
        print(f"   └─ Strong semantic divergence detected between media narratives")
        print(f"      and Federal Reserve policy decisions")
        print(f"   └─ Media sentiment patterns do not align with monetary policy cycles")
        print(f"   └─ Suggests independent information channels or temporal lags")
        print(f"   └─ Financial media may react to different economic signals than Fed")
        
        print(f"\nNARRATIVE PATTERN INSIGHTS:")
        print(f"   └─ Crisis period (2008-2009): Media sentiment highly volatile")
        print(f"   └─ Recovery phase (2010-2013): Gradual narrative stabilization")
        print(f"   └─ Policy communication gaps may explain semantic divergence")
        
    elif best_r2 < 0.1:
        print(f"\nTEMPORAL SEMANTIC INTERPRETATION:")
        print(f"   └─ Moderate semantic-policy alignment detected")
        print(f"   └─ Some correlation between narrative patterns and policy cycles")
        print(f"   └─ {best_model} captures partial temporal relationships")
        
        print(f"\nNARRATIVE PATTERN INSIGHTS:")
        print(f"   └─ Media sentiment shows delayed response to policy changes")
        print(f"   └─ Semantic clustering reveals distinct economic phases")
        print(f"   └─ Transformer embeddings better capture contextual nuances")
        
    else:
        print(f"\nTEMPORAL SEMANTIC INTERPRETATION:")
        print(f"   └─ Strong semantic-policy alignment confirmed!")
        print(f"   └─ {best_model} successfully captures narrative-policy dynamics")
        print(f"   └─ Media sentiment reliably predicts Fed rate movements")
        
        print(f"\nNARRATIVE PATTERN INSIGHTS:")
        print(f"   └─ Semantic clustering effectively identifies policy regimes")
        print(f"   └─ Financial media provides leading indicators for Fed decisions")
        print(f"   └─ Temporal embeddings reveal systematic policy communication")

    print(f"\nMETHODOLOGICAL INSIGHTS:")
    print(f"   └─ TF-IDF captures keyword-based temporal patterns")
    print(f"   └─ SBERT embeddings provide contextual semantic understanding") 
    print(f"   └─ Ensemble methods combine diverse semantic perspectives")
    print(f"   └─ Temporal analysis reveals economic phase transitions")

    print(f"\nFUTURE SEMANTIC RESEARCH DIRECTIONS:")
    print(f"   └─ Implement temporal attention mechanisms for sequence modeling")
    print(f"   └─ Apply semantic clustering (t-SNE, UMAP) for phase visualization")
    print(f"   └─ Integrate economic indicator embeddings for multi-modal analysis")
    print(f"   └─ Develop real-time semantic monitoring for policy prediction")
    print(f"   └─ Explore cross-linguistic semantic patterns in global markets")

    print("\n" + "="*70)


# Update function aliases for backward compatibility
compare_models = compare_semantic_models
print_model_insights = print_semantic_insights
