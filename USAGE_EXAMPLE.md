# Financial NLP Analysis Example

This notebook demonstrates how to use the NLP x Finance project components.

## Quick Start

```python
# Import necessary modules
from src.data_loader import build_dataset
from src.feature_engineering import compute_tfidf_embeddings, compute_sbert_embeddings
from src.models import FedRatePredictor, compare_models
import pandas as pd

# Load and process data
df = build_dataset(
    fed_csv_path="data/fed_rates.csv",
    headlines_csv_path="data/headlines_fixed.csv"
)

# Extract features
X_tfidf = compute_tfidf_embeddings(df["headlines"], max_features=500)
X_sbert = compute_sbert_embeddings(df["headlines"])

# Train models
tfidf_predictor = FedRatePredictor()
results = tfidf_predictor.train_and_evaluate(
    X_tfidf, df["fed_rate"], pd.to_datetime(df["date"]), 
    label="TF-IDF", color='orange'
)

print(f"Model R² Score: {results['r2']:.4f}")
```

## Custom Model Parameters

```python
# Use custom Random Forest parameters
custom_params = {
    'n_estimators': 200,
    'max_depth': 15,
    'min_samples_split': 3
}

predictor = FedRatePredictor(model_params=custom_params)
```

## Feature Importance Analysis

```python
# Get feature importance (after training)
importance_df = predictor.get_feature_importance()
print(importance_df.head(10))
```

For the complete pipeline, simply run:
```bash
python main.py
```
