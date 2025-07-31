"""
Main pipeline for lens modeling analysis.
Runs preprocessing, topic modeling, and lens model analysis in sequence.
"""

import os
import sys
from preprocessing import run_preprocessing
from topic_modeling import run_topic_modeling
from lens_model import run_lens_model
from config import OUTPUT_DIR

def main():
    """
    Main pipeline function that runs the complete analysis.
    """
    print("="*60)
    print("LENS MODELING PIPELINE")
    print("="*60)
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        # Step 1: Preprocessing
        print("\n🔄 STEP 1: PREPROCESSING")
        print("-" * 40)
        tfidf_matrix, vectorizer, transcript_ids = run_preprocessing()
        
        # Step 2: Topic Modeling
        print("\n🔄 STEP 2: TOPIC MODELING")
        print("-" * 40)
        W_normalized, H, top_words = run_topic_modeling(
            tfidf_matrix, vectorizer, transcript_ids
        )
        
        # Step 3: Lens Model Analysis
        print("\n🔄 STEP 3: LENS MODEL ANALYSIS")
        print("-" * 40)
        cue_matrix, lens_results = run_lens_model()
        
        print("\n" + "="*60)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        print(f"\nOutput files saved in: {OUTPUT_DIR}/")
        print("- transcripts.csv: Cleaned transcripts with PIDs")
        print("- topic_weights.csv: Normalized topic weights per transcript")
        print("- topic_words.csv: Top words for each topic")
        print("- cue_matrix.csv: Topic weights merged with gender data")
        print("- lens_model_feature_importance.csv: Topic importance for gender prediction")
        print("- lens_model_performance.txt: Model performance summary")
        
        print(f"\nResults summary:")
        print(f"- Number of transcripts: {len(transcript_ids)}")
        print(f"- Number of topics: {W_normalized.shape[1]}")
        print(f"- Vocabulary size: {len(vectorizer.vocabulary_)}")
        print(f"- Number of gender judgments: {len(cue_matrix)}")
        print(f"- Lens model accuracy: {lens_results['accuracy']:.3f}")
        
    except Exception as e:
        print(f"\n ERROR: {str(e)}")
        print("Please check your data and configuration.")
        sys.exit(1)

if __name__ == "__main__":
    main() 