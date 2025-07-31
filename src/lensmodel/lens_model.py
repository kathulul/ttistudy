"""
Lens Model module for analyzing topic-gender associations.
Merges topic weights with DeepFace gender perception data and performs lens model analysis.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
from config import DEEPFACE_CSV_PATH, OUTPUT_DIR, CUE_MATRIX_FILE, LENS_MODEL_PARAMS

def run_lens_model():
    """
    Main function to run the complete lens model analysis.
    
    Returns:
        tuple: (cue_matrix, results)
    """
    print("="*60)
    print("LENS MODEL ANALYSIS")
    print("="*60)
    
    try:
        # Load DeepFace gender perception data
        print("Loading DeepFace gender perception data...")
        if not os.path.exists(DEEPFACE_CSV_PATH):
            raise FileNotFoundError(f"DeepFace data not found at {DEEPFACE_CSV_PATH}")
        
        deepface_df = pd.read_csv(DEEPFACE_CSV_PATH)
        expected_columns = ['PID', 'Image', 'Gender']
        missing_columns = [col for col in expected_columns if col not in deepface_df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in DeepFace data: {missing_columns}")
        
        print(f"Loaded {len(deepface_df)} DeepFace judgments")
        print(f"Unique PIDs: {deepface_df['PID'].nunique()}")
        
        # Load topic weights
        print("Loading topic weights...")
        topic_weights_path = os.path.join(OUTPUT_DIR, "topic_weights.csv")
        if not os.path.exists(topic_weights_path):
            raise FileNotFoundError(f"Topic weights not found at {topic_weights_path}")
        
        topic_weights_df = pd.read_csv(topic_weights_path, index_col='pid')
        print(f"Loaded topic weights for {len(topic_weights_df)} transcripts")
        
        # Create cue matrix by merging data
        print("Creating cue matrix...")
        cue_matrix = deepface_df.merge(topic_weights_df, left_on='PID', right_index=True, how='inner')
        print(f"Cue matrix shape: {cue_matrix.shape}")
        print(f"Gender distribution: {dict(cue_matrix['Gender'].value_counts())}")
        
        # Run lens model analysis
        print("Running lens model analysis...")
        topic_columns = [col for col in cue_matrix.columns if col.startswith('topic_')]
        X = cue_matrix[topic_columns]
        y = cue_matrix['Gender']
        
        # Fit model on all data (no train/test split for lens model analysis)
        print("Fitting lens model on all data...")
        model = LogisticRegression(**LENS_MODEL_PARAMS['logistic_regression'])
        model.fit(X, y)
        
        # Make predictions on all data
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]
        accuracy = accuracy_score(y, y_pred)
        
        # Get feature importance and odds ratios
        feature_importance = pd.DataFrame({
            'topic': topic_columns,
            'coefficient': model.coef_[0],
            'abs_coefficient': np.abs(model.coef_[0])
        }).sort_values('abs_coefficient', ascending=False)
        feature_importance['odds_ratio'] = np.exp(feature_importance['coefficient'])
        
        results = {
            'model': model,
            'accuracy': accuracy,
            'feature_importance': feature_importance,
            'classification_report': classification_report(y, y_pred),
            'confusion_matrix': confusion_matrix(y, y_pred),
            'X_test': X,
            'y_test': y,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        # Save results
        print("Saving lens model results...")
        cue_matrix.to_csv(os.path.join(OUTPUT_DIR, CUE_MATRIX_FILE), index=False)
        
        feature_importance_path = os.path.join(OUTPUT_DIR, "lens_model_feature_importance.csv")
        results['feature_importance'].to_csv(feature_importance_path, index=False)
        
        performance_path = os.path.join(OUTPUT_DIR, "lens_model_performance.txt")
        with open(performance_path, 'w') as f:
            f.write("LENS MODEL ANALYSIS RESULTS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model Accuracy: {results['accuracy']:.3f}\n\n")
            f.write("Classification Report:\n")
            f.write(results['classification_report'])
            f.write("\nConfusion Matrix:\n")
            f.write(str(results['confusion_matrix']))
            f.write("\n\nTop 10 Most Important Topics:\n")
            f.write(results['feature_importance'].head(10).to_string())
        
        print(f"Cue matrix saved to: {CUE_MATRIX_FILE}")
        print(f"Feature importance saved to: lens_model_feature_importance.csv")
        print(f"Performance summary saved to: lens_model_performance.txt")
        
        # Print summary
        print("\n" + "="*60)
        print("LENS MODEL ANALYSIS RESULTS")
        print("="*60)
        
        print(f"\nModel Fit (All Data): {results['accuracy']:.3f}")
        print(f"Total Observations: {len(y)}")
        print(f"Gender Distribution: {dict(y.value_counts())}")
        
        print(f"\nTop 10 Most Predictive Topics:")
        top_topics = results['feature_importance'].head(10)
        for _, row in top_topics.iterrows():
            topic_name = row['topic']
            coefficient = row['coefficient']
            odds_ratio = row['odds_ratio']
            direction = "MALE" if coefficient > 0 else "FEMALE"
            print(f"  {topic_name}: {coefficient:.3f} (OR: {odds_ratio:.3f}) → {direction}")
        
        print(f"\nClassification Report (All Data):")
        print(results['classification_report'])
        
        print(f"\nConfusion Matrix (All Data):")
        print(results['confusion_matrix'])
        
        print("\n" + "="*60)
        print("✅ LENS MODEL ANALYSIS COMPLETED!")
        print("="*60)
        
        return cue_matrix, results
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        raise

if __name__ == "__main__":
    run_lens_model() 