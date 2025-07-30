"""
Feature Engineering Module for NLP x Finance Project

This module contains functions for extracting numerical features from text data
using various NLP techniques including TF-IDF and sentence transformers (SBERT).
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from scipy import sparse
from typing import Union


def compute_tfidf_embeddings(texts: pd.Series, max_features: int = 500) -> np.ndarray:
    """
    Compute TF-IDF vectors for a series of texts.

    Args:
        texts: Series of text documents
        max_features: Maximum number of features to extract

    Returns:
        numpy.ndarray: TF-IDF feature matrix
    """
    # Clean the text data
    texts = texts.fillna("").astype(str)
    texts = texts.apply(lambda x: x.strip() if isinstance(x, str) else "")

    # Check if we have any non-empty texts
    non_empty_texts = [text for text in texts if text.strip()]
    if len(non_empty_texts) == 0:
        raise ValueError("All texts are empty after cleaning")

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=1,  # Minimum document frequency
        lowercase=True,
        strip_accents="ascii",
    )

    X = vectorizer.fit_transform(texts)

    # Always return dense array for consistency
    if sparse.issparse(X):
        return X.toarray()
    else:
        return np.array(X)


def compute_sbert_embeddings(
    texts: pd.Series, model_name: str = "all-MiniLM-L6-v2"
) -> np.ndarray:
    """
    Compute SBERT (Sentence-BERT) embeddings for text data.

    Args:
        texts: Series of text documents
        model_name: Name of the sentence transformer model to use

    Returns:
        numpy.ndarray: SBERT embeddings matrix
    """
    print(f"  Loading SBERT model: {model_name}")
    model = SentenceTransformer(model_name)

    # Clean and prepare texts
    text_list = texts.fillna("").astype(str).tolist()

    print(f"  Encoding {len(text_list)} documents...")
    embeddings = model.encode(text_list, show_progress_bar=True)

    return np.array(embeddings)
