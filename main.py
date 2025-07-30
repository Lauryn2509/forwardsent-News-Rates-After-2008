"""
Financial News Sentiment Analysis & Fed Rate Prediction

This script demonstrates how Natural Language Processing can be applied to financial markets
by predicting Federal Reserve interest rates using news headlines sentiment analysis.

The pipeline includes:
1. Data loading and preprocessing
2. NLP feature extraction (TF-IDF and SBERT)
3. Machine learning model training and evaluation
4. Results visualization and analysis

Author: Enzo Montariol
Project: NLP x Finance - Fed Rate Prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from src.data_loader import build_dataset
from src.feature_engineering import compute_tfidf_embeddings, compute_sbert_embeddings
from src.models import FedRatePredictor, EnsemblePredictor, compare_models, print_model_insights

# Ensure figures directory exists
os.makedirs("figures", exist_ok=True)

print("Financial News Sentiment Analysis & Fed Rate Prediction")
print("=" * 60)
print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ========================================
# STEP 1: Data Loading and Preprocessing
# ========================================
print("STEP 1: Loading and preprocessing data...")

df = build_dataset(
    fed_csv_path="data/fed_rates.csv",
    headlines_csv_path="data/headlines_fixed.csv",
    save_path="data/merged_dataset.csv",
)

# Reload the merged dataset for consistency
df = pd.read_csv("data/merged_dataset.csv")
print(
    f"Dataset loaded: {len(df)} observations from {df['date'].min()} to {df['date'].max()}"
)
print(f"Fed rate range: {df['fed_rate'].min():.2f}% to {df['fed_rate'].max():.2f}%")
print()

# ========================================
# STEP 2: NLP Feature Engineering
# ========================================
print("STEP 2: Extracting NLP features...")

# TF-IDF Features
print("  Computing TF-IDF embeddings...")
X_tfidf = compute_tfidf_embeddings(df["headlines"], max_features=500)

# Save TF-IDF features
pd.DataFrame(X_tfidf).to_csv("data/X_tfidf.csv", index=False)
print(f"TF-IDF features saved - Shape: {X_tfidf.shape}")

# Skip SBERT for now to avoid segmentation fault
print("  Skipping SBERT for this test run...")
X_sbert = None
print()

# ========================================
# STEP 3: Model Training and Evaluation
# ========================================
print("STEP 3: Training and evaluating optimized models...")

# Target variable and dates for visualization
y = df["fed_rate"]
dates = pd.to_datetime(df["date"])

# ========================================
# Model Training with Optimized Algorithms (TF-IDF only)
# ========================================

print("\nTraining individual optimized models on TF-IDF features...")

# Train models on TF-IDF features
print("\nTraining models on TF-IDF features:")

# XGBoost
xgboost_predictor = FedRatePredictor("xgboost")
tfidf_xgb_results = xgboost_predictor.train_and_evaluate(
    X_tfidf, y, dates, label="TF-IDF", color="orange"
)

# Ridge Regression
ridge_predictor_tfidf = FedRatePredictor("ridge")
tfidf_ridge_results = ridge_predictor_tfidf.train_and_evaluate(
    X_tfidf, y, dates, label="TF-IDF", color="red"
)

# SVR
svr_predictor_tfidf = FedRatePredictor("svr")
tfidf_svr_results = svr_predictor_tfidf.train_and_evaluate(
    X_tfidf, y, dates, label="TF-IDF", color="blue"
)

# Train ensemble model
print("\nTraining ensemble model:")
tfidf_ensemble = EnsemblePredictor()
tfidf_ensemble_results = tfidf_ensemble.train_and_evaluate(
    X_tfidf, y, dates, label="TF-IDF", color="purple"
)

# ========================================
# STEP 4: Comprehensive Results Summary
# ========================================
print()
print("COMPREHENSIVE RESULTS SUMMARY")
print("=" * 50)

# Create comparison table for TF-IDF models only
all_results = {
    "TF-IDF + XGBoost": tfidf_xgb_results,
    "TF-IDF + Ridge": tfidf_ridge_results,
    "TF-IDF + SVR": tfidf_svr_results,
    "TF-IDF + Ensemble": tfidf_ensemble_results,
}

comparison_df = compare_models(all_results)
print(comparison_df.to_string(index=False, float_format="%.4f"))
print()

# Find best overall model
best_overall = comparison_df.loc[comparison_df["R²"].idxmax(), "Model"]
best_r2 = comparison_df["R²"].max()
print(f"Best performing model: {best_overall} (R² = {best_r2:.4f})")

# Feature importance for XGBoost model
print("\nFeature Importance Analysis:")
print("Top 10 most important TF-IDF features (XGBoost):")
importance_df = xgboost_predictor.get_feature_importance()
print(importance_df.head(10).to_string(index=False))

# ========================================
# STEP 5: Enhanced Business Insights
# ========================================
print()
print_model_insights(comparison_df)

print("Optimized models analysis completed successfully!")
print(f"Results saved in:")
print(f"   • Visualizations: ./figures/ (including ensemble plots)")
print(f"   • Features: ./data/X_tfidf.csv")
print(f"   • Dataset: ./data/merged_dataset.csv")
print()
print("Key improvements with optimized models:")
print("   • XGBoost: Better handling of non-linear relationships")
print("   • Ridge: Regularization for high-dimensional features")
print("   • SVR: Robust to outliers and small datasets")
print("   • Ensemble: Combines strengths of all approaches")
print()
print("Compare with your previous Random Forest results!")
print("   Previous SBERT + RF: R² = -0.0628")
print("   Previous TF-IDF + RF: R² = -0.1456")
