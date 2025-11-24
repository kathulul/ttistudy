"""Topic model using Contextualized Topic Models"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import pandas as pd
from helper import load_chatlogs, clean_text
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessingStopwords
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.models.ctm import CombinedTM
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from collections import Counter


def prepare_chatlogs():
    chatlogs = load_chatlogs()
    chatlogs['text'] = chatlogs['text'].apply(clean_text)
    return chatlogs


def preprocessing(chatlogs, min_df=2, max_df=0.95):
    documents = chatlogs['text'].tolist()
    sp = WhiteSpacePreprocessingStopwords(documents, "english")
    preprocessed, unpreprocessed, _, retained_indices = sp.preprocess()
    
    # Stopword filtering for BoW only
    custom_stopwords = ['yeah', 'yes', 'yea', 'yep', 'okay', 'ok', 'hmm', 'hmmm', 'um', 'uh', 'ah',
                       'hey', 'hi', 'hello', 'thanks', 'thank', 'welcome',
                       'gotta', 'gonna', 'wanna', 'lemme', 'im', 'ive', 'youre', 'theyre', 'were',
                       'havent', 'hasnt', 'hadnt', 'isnt', 'arent', 'wasnt', 'werent',
                       'dont', 'cant', 'wont', 'wouldnt', 'couldnt', 'shouldnt',
                       'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                       'really', 'very', 'quite', 'just', 'also', 'even', 'still', 'well', 'good', 'clearly',
                       'like', 'thing', 'things', 'actually', 'maybe', 'definitely', 'probably', 
                       'kinda', 'sorta', 'sort', 'let', 'lets', 'don', 'didn', 'doesn', 'wasn', 'weren']
    stopwords = set(ENGLISH_STOP_WORDS) | set(custom_stopwords)
    
    filtered_docs = []
    word_doc_count = Counter()
    for doc in preprocessed:
        words = [w for w in doc.split() if w.lower() not in stopwords]
        filtered_docs.append(words)
        word_doc_count.update(set(words))  # Count unique words per document
    
    # Calculate document frequency thresholds
    n_docs = len(filtered_docs)
    min_docs = max(1, int(min_df * n_docs)) if min_df < 1 else min_df
    max_docs = int(max_df * n_docs) if max_df < 1 else max_df
    
    # Filter by document frequency and construct final output
    valid_words = {word for word, count in word_doc_count.items() 
                   if min_docs <= count <= max_docs}
    final_preprocessed = [' '.join(w for w in words if w in valid_words) for words in filtered_docs]
    vocab = sorted(valid_words)
    retained_chatlogs = chatlogs.iloc[retained_indices].copy()
    
    print(f"  Vocabulary size: {len(vocab)}")
    print(f"  Retained {len(retained_chatlogs)}/{len(chatlogs)} documents")
    return final_preprocessed, unpreprocessed, vocab, retained_chatlogs


def estimate_truncation(unpreprocessed, max_seq_length):
    """Estimate which documents may be truncated (rough estimate: 1.3 tokens per word)"""
    truncated_count = 0
    for doc in unpreprocessed:
        word_count = len(doc.split())
        estimated_tokens = word_count * 1.3
        if estimated_tokens > max_seq_length:
            truncated_count += 1
    
    if truncated_count > 0:
        print(f"  Warning: ~{truncated_count} documents may be truncated (exceeding max_seq_length={max_seq_length})")
    else:
        print(f"  All documents fit within max_seq_length={max_seq_length}")
    return truncated_count


def prepare_data(unpreprocessed, preprocessed, embedding_model="paraphrase-distilroberta-base-v2", max_seq_length=512):
    print(f"  Using embedding model: {embedding_model}")
    print(f"  Max sequence length: {max_seq_length}")
    estimate_truncation(unpreprocessed, max_seq_length)
    
    qt = TopicModelDataPreparation(embedding_model, max_seq_length=max_seq_length)
    training_dataset = qt.fit(text_for_contextual=unpreprocessed, text_for_bow=preprocessed)
    print(f"  Training dataset ready with {len(unpreprocessed)} documents")
    return training_dataset, qt


def fit_contexttm(training_dataset, qt, num_epochs, patience, delta, n_components, contextual_size=768, batch_size=39):
    print(f"  Topics: {n_components}")
    print(f"  BoW size: {len(qt.vocab)}")
    print(f"  Contextual size: {contextual_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Max epochs: {num_epochs}")
    if patience > 0:
        print(f"  Early stopping: patience={patience}, delta={delta}")
    
    ctm = CombinedTM(
        bow_size=len(qt.vocab),
        contextual_size=contextual_size,
        n_components=n_components,
        batch_size=batch_size,
        num_epochs=num_epochs
    )
    ctm.fit(training_dataset, patience=patience, delta=delta)
    print("  Training complete!")
    return ctm, qt


def save_ctm_results(ctm, training_dataset, retained_chatlogs, n_words_per_topic=10, n_samples=20):
    print(f"  Extracting top {n_words_per_topic} words per topic...")
    topic_lists = ctm.get_topic_lists(n_words_per_topic)
    n_topics = len(topic_lists)
    
    topic_words_rows = []
    for topic_id, words in enumerate(topic_lists):
        for rank, word in enumerate(words, 1):
            topic_words_rows.append({'topic_id': topic_id, 'rank': rank, 'word': word})
    
    topic_words_df = pd.DataFrame(topic_words_rows)
    words_path = f"CTMexport/n{n_topics}words.csv"
    topic_words_df.to_csv(words_path, index=False)
    print(f"  Saved topic words to {words_path}")
    
    print(f"  Getting document-topic distributions (n_samples={n_samples})...")
    doc_topic_matrix = ctm.get_thetas(training_dataset, n_samples=n_samples)
    doc_topic_df = pd.DataFrame(doc_topic_matrix, columns=[f"topic_{i}" for i in range(n_topics)])
    doc_topic_df.insert(0, 'folder', retained_chatlogs['folder'].values)
    matrix_path = f"CTMexport/n{n_topics}matrix.csv"
    doc_topic_df.to_csv(matrix_path, index=False)
    print(f"  Saved document-topic matrix to {matrix_path}")
    print(f"  Shape: {doc_topic_df.shape} (documents x topics)")
    
    return doc_topic_df, topic_words_df


if __name__ == "__main__":
    N_COMPONENTS = 2
    MAX_SEQ_LENGTH = 258
    N_WORDS_PER_TOPIC = 10
    N_SAMPLES = 20
    EMBEDDING_MODEL = "paraphrase-distilroberta-base-v2"
    NUM_EPOCHS = 25
    BATCH_SIZE = 39
    
    print("Loading chatlogs...")
    chatlogs = prepare_chatlogs()
    print(f"Loaded {len(chatlogs)} chatlogs")
    
    print("\nPreprocessing...")
    preprocessed, unpreprocessed, vocab, retained_chatlogs = preprocessing(chatlogs)
    
    print("\nPreparing data...")
    training_dataset, qt = prepare_data(unpreprocessed, preprocessed, EMBEDDING_MODEL, MAX_SEQ_LENGTH)
    
    print("\nTraining model...")

    ctm, qt = fit_contexttm(training_dataset, qt, NUM_EPOCHS, 0, 0, N_COMPONENTS, batch_size=BATCH_SIZE)
    
    print("\nSaving results...")
    doc_topic_df, topic_words_df = save_ctm_results(ctm, training_dataset, retained_chatlogs, N_WORDS_PER_TOPIC, N_SAMPLES)
    
    print("\n" + "="*60)
    print("CTM Pipeline Complete!")
    print("="*60)
    print(f"\nDocument-topic matrix shape: {doc_topic_df.shape}")
    print(f"Topic words saved for {doc_topic_df.shape[1] - 1} topics")

