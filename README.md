# Financial News Sentiment Analysis & Fed Rate Prediction

A Natural Language Processing project that predicts Federal Reserve interest rates using news headlines sentiment analysis. This project demonstrates the application of NLP techniques in quantitative finance by analyzing the relationship between media sentiment and monetary policy decisions.

## Project Overview

This project explores how financial news headlines can be used to predict Federal Reserve rate changes. By applying various NLP techniques (TF-IDF and SBERT embeddings), we create features from news text data and use machine learning models to forecast Fed rate movements.

### Key Features
- **Multi-source Data Integration**: Combines Fed rate data with financial news headlines
- **Advanced NLP Processing**: Uses both traditional (TF-IDF) and modern (SBERT) text vectorization
- **Time Series Analysis**: Analyzes temporal relationships between news sentiment and rate changes
- **Comparative Modeling**: Evaluates different text representation methods
- **Visualization**: Generates insightful plots showing model performance and predictions

## Dataset

- **Fed Rates Data**: Historical Federal Reserve interest rates from FRED (Federal Reserve Economic Data)
- **News Headlines**: Financial news headlines aggregated by month
- **Time Period**: 2008-2013 (covering the financial crisis and recovery period)
- **Total Observations**: 37 monthly data points

## Technical Stack

- **Python 3.11+**
- **Machine Learning**: scikit-learn, Random Forest Regressor
- **NLP**: TF-IDF Vectorization, Sentence Transformers (SBERT)
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib
- **Text Processing**: Custom preprocessing pipeline

## Project Structure

```
nlp_laulouks/
├── main.py                 # Main execution script
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
├── data/                  # Data files
│   ├── fed_rates.csv      # Federal Reserve rates
│   ├── headlines_fixed.csv # Cleaned news headlines
│   ├── merged_dataset.csv # Combined dataset
│   ├── X_tfidf.csv       # TF-IDF features
│   └── X_sbert.csv       # SBERT embeddings
├── src/                   # Source code modules
│   ├── data_loader.py     # Data loading and preprocessing
│   ├── feature_engineering.py # NLP feature extraction
│   ├── models.py          # Machine learning models
│   └── text_cleaner.py    # Text preprocessing utilities
└── figures/               # Generated visualizations
    ├── scatter_tfidf.png  # TF-IDF prediction scatter plot
    ├── scatter_sbert.png  # SBERT prediction scatter plot
    ├── curve_tfidf.png    # TF-IDF time series plot
    └── curve_sbert.png    # SBERT time series plot
```

## Quick Start

### Prerequisites
```bash
python --version  # Ensure Python 3.11+
pip --version     # Ensure pip is installed
```

### Installation
```bash
# Clone the repository
git clone https://github.com/enzomontariol/nlp_laulouks.git
cd nlp_laulouks

# Install dependencies
pip install -r requirements.txt
```

### Running the Analysis
```bash
# Execute the complete pipeline
python main.py
```

This will:
1. Load and merge Fed rates with news headlines data
2. Generate TF-IDF and SBERT text embeddings
3. Train Random Forest models on both feature sets
4. Generate evaluation metrics and visualizations
5. Save results and plots to respective directories

## Model Performance

The project compares two text representation approaches:

| Method | MSE | R² Score | Description |
|--------|-----|----------|-------------|
| TF-IDF | 0.0100 | -0.1456 | Traditional bag-of-words approach with term frequency weighting |
| SBERT | 0.0093 | -0.0628 | Modern transformer-based sentence embeddings |

**Note**: Negative R² scores indicate that the models are currently underperforming compared to a simple mean baseline, suggesting opportunities for model improvement through feature engineering, hyperparameter tuning, or alternative approaches.

## Résultats Visuels et Analyse

### Graphiques de Dispersion (Prédictions vs Réalité)

#### TF-IDF Model Performance
![TF-IDF Scatter Plot](figures/scatter_tf_idf.png)

**Analyse**: Ce graphique de dispersion compare les prédictions du modèle TF-IDF avec les valeurs réelles des taux de la Fed. On observe :
- Une dispersion importante des points autour de la ligne de régression idéale
- Le modèle a tendance à sous-estimer les taux élevés (points au-dessus de la ligne)
- Les prédictions sont concentrées dans une gamme plus étroite que les valeurs réelles
- Le R² négatif (-0.1456) confirme que le modèle performe moins bien qu'une simple moyenne

#### SBERT Model Performance  
![SBERT Scatter Plot](figures/scatter_sbert.png)

**Analyse**: Le modèle SBERT montre une amélioration notable par rapport au TF-IDF :
- Dispersion légèrement réduite des prédictions
- Meilleure capture des variations dans les taux moyens
- R² moins négatif (-0.0628) indiquant une performance supérieure
- Les embeddings SBERT capturent mieux la sémantique des headlines financières

### Évolution Temporelle des Prédictions

#### TF-IDF Time Series Analysis
![TF-IDF Time Series](figures/curve_tf_idf.png)

**Analyse**: Cette courbe temporelle révèle :
- **Période 2008-2009**: Le modèle ne capture pas bien la chute drastique des taux pendant la crise financière
- **2010-2011**: Sous-estimation persistante de la stabilité des taux bas
- **2012-2013**: Amélioration relative mais toujours des écarts significatifs
- Le modèle TF-IDF peine à saisir les changements de régime monétaire

#### SBERT Time Series Analysis
![SBERT Time Series](figures/curve_sbert.png)

**Analyse**: Les embeddings SBERT montrent :
- **Meilleure réactivité** aux changements de sentiment dans les news
- **Suivi plus fidèle** des tendances générales des taux
- **Moins de retard** dans la détection des changements de politique monétaire
- Les représentations contextuelles capturent mieux les nuances du langage financier

### Observations Clés des Visualisations

1. **Supériorité de SBERT**: Les graphiques confirment quantitativement et visuellement que SBERT surpasse TF-IDF
2. **Défis de Prédiction**: Les deux modèles montrent les difficultés à prédire les mouvements de taux basés uniquement sur les headlines
3. **Patterns Temporels**: La période de crise (2008-2009) est particulièrement difficile à modéliser
4. **Variance Limitée**: Les modèles tendent à prédire dans une gamme plus restreinte que la réalité

Ces résultats suggèrent que bien que SBERT soit prometteur, des améliorations sont nécessaires pour capturer pleinement la complexité de la relation entre sentiment médiatique et politique monétaire.

## Key Insights

1. **SBERT vs TF-IDF**: SBERT embeddings show slightly better performance (lower MSE, higher R²)
2. **Temporal Patterns**: The visualizations reveal interesting relationships between news sentiment cycles and Fed rate changes
3. **Model Limitations**: Current models suggest the relationship between news headlines and Fed rates is more complex than captured by simple regression

## Visualizations

The project generates four key visualizations:

1. **Scatter Plots**: Show predicted vs actual Fed rates for both TF-IDF and SBERT models
2. **Time Series Plots**: Display temporal evolution of predictions compared to actual rates

## Technical Details

### NLP Pipeline
1. **Text Preprocessing**: Cleaning and normalization of headlines
2. **Feature Extraction**: 
   - TF-IDF: 500-dimensional sparse vectors
   - SBERT: 384-dimensional dense embeddings using 'all-MiniLM-L6-v2' model
3. **Model Training**: Random Forest Regressor with 100 estimators

### Data Processing
- Monthly aggregation of daily headlines
- Temporal alignment of news data with Fed rate decisions
- Robust handling of missing values and text encoding issues

## Future Improvements

- [ ] **Enhanced Feature Engineering**: Sentiment analysis, named entity recognition
- [ ] **Advanced Models**: LSTM, Transformer-based time series models
- [ ] **Expanded Dataset**: Include more recent data and additional economic indicators
- [ ] **Cross-Validation**: Implement time series cross-validation
- [ ] **Model Interpretability**: Add SHAP values or attention visualization
- [ ] **Real-time Predictions**: Build API for live Fed rate predictions

## Dependencies

See `requirements.txt` for the complete list. Key libraries include:
- `pandas>=1.5.0`
- `numpy<2.0.0` (for PyTorch compatibility)
- `scikit-learn>=1.2.0`
- `sentence-transformers>=2.2.0`
- `matplotlib>=3.5.0`

## Contributing

Contributions are welcome! Areas for improvement:
- Model architecture enhancements
- Additional data sources integration
- Feature engineering improvements
- Documentation and code quality

## License

This project is for educational and research purposes. Please cite appropriately if used in academic work.

## Contact

For questions or collaborations, please reach out through GitHub issues or contact the repository owner.

---

*This project demonstrates the intersection of Natural Language Processing and Quantitative Finance, showcasing how textual data can inform financial modeling and decision-making.*