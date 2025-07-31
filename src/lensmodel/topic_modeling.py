"""
Topic modeling module using Non-negative Matrix Factorization (NMF).
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from config import NMF_PARAMS, OUTPUT_DIR, TOPIC_WEIGHTS_FILE, TOPIC_WORDS_FILE, TOP_WORDS_PER_TOPIC
import os

def run_topic_modeling(tfidf_matrix, vectorizer, transcript_ids):
    """
    Run NMF topic modeling and save results.
    
    Args:
        tfidf_matrix: Sparse matrix from TF-IDF vectorizer
        vectorizer: Fitted TF-IDF vectorizer
        transcript_ids: List of transcript IDs
        
    Returns:
        tuple: (W_normalized, H, top_words)
    """
    print("Running NMF topic modeling...")
    
    # Initialize and fit NMF
    nmf = NMF(**NMF_PARAMS)
    W = nmf.fit_transform(tfidf_matrix)  # Document-topic matrix
    H = nmf.components_  # Topic-word matrix
    
    print(f"NMF completed! W shape: {W.shape}, H shape: {H.shape}")
    
    # Normalize topic weights so each row sums to 1
    W_normalized = W / (W.sum(axis=1, keepdims=True) + 1e-10)
    
    # Get top words for each topic
    feature_names = vectorizer.get_feature_names_out()
    top_words = {}
    topic_words_data = []
    
    for topic_idx in range(H.shape[0]):
        # Get indices of top words for this topic
        top_indices = H[topic_idx].argsort()[-TOP_WORDS_PER_TOPIC:][::-1]
        top_words[topic_idx] = [feature_names[i] for i in top_indices]
        
        # Prepare data for CSV
        for rank, word in enumerate(top_words[topic_idx], 1):
            topic_words_data.append({
                'topic_id': topic_idx,
                'rank': rank,
                'word': word,
                'weight': H[topic_idx][vectorizer.vocabulary_[word]]
            })
    
    # Save topic weights
    topic_weights_df = pd.DataFrame(
        W_normalized,
        index=transcript_ids,
        columns=[f'topic_{i}' for i in range(W.shape[1])]
    )
    topic_weights_df.index.name = 'pid'
    topic_weights_df.to_csv(os.path.join(OUTPUT_DIR, TOPIC_WEIGHTS_FILE))
    
    # Save topic words
    topic_words_df = pd.DataFrame(topic_words_data)
    topic_words_df.to_csv(os.path.join(OUTPUT_DIR, TOPIC_WORDS_FILE), index=False)
    
    print(f"Topic weights saved to: {TOPIC_WEIGHTS_FILE}")
    print(f"Topic words saved to: {TOPIC_WORDS_FILE}")
    
    # Print topic summary
    print("\n" + "="*60)
    print("TOPIC MODELING RESULTS")
    print("="*60)
    
    for topic_id, words in top_words.items():
        print(f"\nTopic {topic_id}:")
        print(f"  Top words: {', '.join(words[:10])}")
        if len(words) > 10:
            print(f"  Additional: {', '.join(words[10:15])}")
    
    print("\nTopic modeling completed!")
    return W_normalized, H, top_words

if __name__ == "__main__":
    print("Please run this module from the main pipeline or import it.") 