"""
Configuration file for the lens modeling pipeline.
"""

# Data paths
CHATLOGS_PATH = "../../chatlogsM"  # Path to chatlogsM folder
DEEPFACE_CSV_PATH = "data/deepfacelongform.csv"
OUTPUT_DIR = "data"

# Preprocessing parameters
TFIDF_PARAMS = {
    'ngram_range': (1, 2),
    'min_df': 2,  
    'max_df': 0.95,  
    'lowercase': True,
    'strip_accents': 'unicode'
}

# Topic modeling parameters
NMF_PARAMS = {
    'n_components': 10,
    'solver': 'mu',
    'beta_loss': 'kullback-leibler',
    'max_iter': 700,
    'random_state': 42
}

TOP_WORDS_PER_TOPIC = 50

# Lens model parameters
LENS_MODEL_PARAMS = {
    # Logistic regression parameters
    'logistic_regression': {
        'random_state': 42,     # Reproducible model fitting
        'max_iter': 2000,       # Increased iterations for better convergence
        'penalty': None,        # No regularization (L1/L2)
        'solver': 'lbfgs'       # Default solver for no penalty
    },
    
    # Feature selection
    'use_all_topics': True,     # Use all 20 topics as features
    'feature_selection': None,  # No feature selection (use all topics)
    
    # Evaluation settings
    'scoring': 'accuracy',      # Primary evaluation metric
    'additional_metrics': ['precision', 'recall', 'f1', 'roc_auc']
}

# SpaCy model
SPACY_MODEL = "en_core_web_lg"

# Output file names
TOPIC_WEIGHTS_FILE = "topic_weights.csv"
TOPIC_WORDS_FILE = "topic_words.csv"
CUE_MATRIX_FILE = "cue_matrix.csv" 