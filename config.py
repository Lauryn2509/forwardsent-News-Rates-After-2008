# Configuration file for NLP x Finance Project

# Data paths
FED_RATES_PATH = "data/fed_rates.csv"
HEADLINES_PATH = "data/headlines_fixed.csv"
MERGED_DATA_PATH = "data/merged_dataset.csv"

# Feature engineering parameters
TFIDF_MAX_FEATURES = 500
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"

# Model parameters
MODEL_PARAMS = {
    "n_estimators": 100,
    "random_state": 42,
    "max_depth": 10,
    "min_samples_split": 5,
}

# Visualization parameters
FIGURE_SIZE = (10, 6)
DPI = 300
COLORS = {"tfidf": "orange", "sbert": "green", "actual": "darkblue"}

# Train-test split parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
