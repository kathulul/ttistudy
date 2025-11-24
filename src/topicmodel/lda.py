""" the topic model using LDA RETIRED BC IT DOES NOT CONVERGE XD"""
import sys
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from helper import load_chatlogs, clean_text

# Initialize spaCy 
_nlp = None
def get_nlp():
    """Lazy load spaCy model."""
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        except OSError:
            print("Warning: spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
            print("Falling back to basic tokenization without lemmatization.")
            _nlp = False
    return _nlp


def fit_lda(
    df: pd.DataFrame,
    n_topics: int = 2,
    alpha: float = None,  # If None, uses 1.0/n_topics for symmetric prior
    beta: float = 0.01,
    max_iter: int = 500,
    random_state: int = 42,
    min_df: int = 3,
    max_df: float = 0.90,
    max_features: int = 1000,
    remove_redacted: bool = True,
    remove_you_said: bool = True,
    output_path: str = "doc_topic_matrix.csv",
    verbose: int = 1,
    check_convergence: bool = True
) -> tuple:
    """
    Complete LDA pipeline: preprocess, vectorize, fit model, extract topics, and save results.
    
    Returns:
        tuple: (lda_model, doc_topic_df, topic_words)
            - lda_model: Fitted LDA model
            - doc_topic_df: Document-topic matrix as DataFrame with folder IDs and topic weights (also saved to CSV)
            - topic_words: Dictionary mapping topic_id -> list of (word, weight) tuples
    """
    # Preprocess and filter texts using clean_text from helper.py, then apply LDA-specific preprocessing
    nlp = get_nlp()  # for lemmatization
    
    def preprocess_for_lda(text):
        # Basic cleaning (universal)
        text = clean_text(text, remove_redacted=remove_redacted, remove_you_said=remove_you_said)
        
        # LDA-specific preprocessing
        # Remove standalone numbers (keep words with numbers like "3D")
        text = re.sub(r'\b\d+\b', '', text)
        
        # Remove punctuation (keep only alphanumeric and spaces)
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Normalize whitespace again (after punctuation removal)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Lemmatization
        if nlp is not None and nlp is not False:
            doc = nlp(text.lower())
            text = ' '.join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])
        
        return text
    
    processed_texts = df['text'].apply(preprocess_for_lda)
    
    non_empty_mask = processed_texts.str.len() > 0
    processed_texts = processed_texts[non_empty_mask]
    filtered_df = df.loc[non_empty_mask].copy()
    
    if len(processed_texts) == 0:
        raise ValueError("No non-empty texts after preprocessing!")
    
    print(f"Processing {len(processed_texts)} documents...")
    
    # Removed many words that could be topic-relevant (e.g., 'like', 'think', 'want', 'see')
    custom_stopwords = [
        'yeah', 'yes', 'yea', 'yep', 'okay', 'ok', 'hmm', 'hmmm', 'um', 'uh', 'ah',
        'hey', 'hi', 'hello', 'thanks', 'thank', 'welcome',
        'gotta', 'gonna', 'wanna', 'lemme', 'im', 'ive', 'youre', 'theyre', 'were',
        'havent', 'hasnt', 'hadnt', 'isnt', 'arent', 'wasnt', 'werent',
        'dont', 'cant', 'wont', 'wouldnt', 'couldnt', 'shouldnt'
    ]
    
    # Combine English stopwords with custom conversational fillers
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    extended_stopwords = list(ENGLISH_STOP_WORDS) + custom_stopwords
    
    # Use CountVectorizer (LDA works better with counts than TF-IDF)
    # Relaxed token pattern to allow 2+ char words to preserve more content
    vectorizer = CountVectorizer(
        min_df=min_df, max_df=max_df, max_features=max_features,
        lowercase=True, stop_words=extended_stopwords, token_pattern=r'\b[a-zA-Z]{2,}\b'
    )
    
    doc_term_matrix = vectorizer.fit_transform(processed_texts)
    feature_names = vectorizer.get_feature_names_out()
    
    if len(feature_names) == 0:
        raise ValueError("No features extracted! Check preprocessing parameters.")
    
    if alpha is None:
        alpha = 1.0 / n_topics
    

    # Fit LDA model
    lda = LatentDirichletAllocation(
        n_components=n_topics, 
        doc_topic_prior=alpha, 
        topic_word_prior=beta,
        max_iter=max_iter, 
        random_state=random_state, 
        verbose=verbose, 
        n_jobs=-1
    )
    
    print(f"\nFitting LDA with {n_topics} topics (alpha={alpha:.3f}, beta={beta})...")
    lda.fit(doc_term_matrix)
    doc_topic_matrix = lda.transform(doc_term_matrix)
    
    doc_lengths = [len(text.split()) for text in processed_texts]
    sparsity = 1.0 - (doc_term_matrix.nnz / (doc_term_matrix.shape[0] * doc_term_matrix.shape[1]))
    
    print(f"Vocabulary size: {len(feature_names)}, Document-term matrix: {doc_term_matrix.shape}")
    print(f"Data diagnostics:")
    print(f"  - Avg words per document: {np.mean(doc_lengths):.1f} (min: {min(doc_lengths)}, max: {max(doc_lengths)})")
    print(f"  - Matrix sparsity: {sparsity:.1%} ({doc_term_matrix.nnz} non-zero entries)")
    print(f"  - Sample processed text (first 200 chars): {processed_texts.iloc[0][:200]}...")
    
    # Optional: Check if model converged early (can be removed by setting check_convergence=False)
    if check_convergence:
        actual_iterations = lda.n_iter_
        if actual_iterations < max_iter:
            print(f"✓ Model converged early after {actual_iterations} iterations (max_iter={max_iter})")
        else:
            print(f"⚠ Model reached max_iter={max_iter} without convergence")
    
    print(f"Document-topic matrix: {doc_topic_matrix.shape}, Perplexity: {lda.perplexity(doc_term_matrix):.2f}\n")
    
    # Extract topic words (normalize to probabilities for interpretability)
    # sklearn's lda.components_ contains raw learned parameters (Dirichlet posterior parameters)
    # These are NOT probabilities - they need to be normalized to get P(word|topic)
    topic_words = {}
    for topic_idx, topic in enumerate(lda.components_):
        # Normalize raw parameters to probabilities (sum to 1.0)
        # This gives us P(word | topic) = probability of word given topic
        topic_sum = topic.sum()
        topic_probs = topic / topic_sum  # Simple normalization (not softmax - that's for logits)
        top_indices = topic_probs.argsort()[-10:][::-1]
        topic_words[topic_idx] = [(feature_names[i], topic_probs[i]) for i in top_indices]
    
    # Save topic words to CSV
    topic_words_rows = []
    for topic_id, words in topic_words.items():
        for rank, (word, weight) in enumerate(words, 1):
            # weight is P(word|topic) - probability distribution (0.0 to 1.0, sums to 1.0 per topic)
            topic_words_rows.append({
                'topic_id': topic_id,
                'rank': rank,
                'word': word,
                'weight': weight  # P(word|topic) - probability (0.0 to 1.0)
            })
    topic_words_df = pd.DataFrame(topic_words_rows)
    topic_words_path = output_path.replace('.csv', '_topic_words.csv')
    topic_words_df.to_csv(topic_words_path, index=False)
    print(f"Saved topic words to {topic_words_path}")
    
    # Create and save document-topic DataFrame
    doc_topic_df = pd.DataFrame(
        doc_topic_matrix,
        columns=[f"topic_{i}" for i in range(n_topics)]
    )
    doc_topic_df.insert(0, 'folder', filtered_df['folder'].values)
    doc_topic_df.to_csv(output_path, index=False)
    print(f"Saved document-topic matrix to {output_path}")
    
    return lda, doc_topic_df, topic_words


if __name__ == "__main__":
    # Load data
    chatlogs = load_chatlogs()
    print(f"Loaded {len(chatlogs)} chatlogs\n")
    
    # Run complete LDA pipeline
    # Using TF-IDF now (better for conversational text)
    lda_model, doc_topic_df, topic_words = fit_lda(chatlogs)
    
    print(f"\nDocument-topic matrix preview:")
    print(doc_topic_df.head())
    print(f"\nMatrix shape: {doc_topic_df.shape} (documents x topics)")
