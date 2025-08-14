"""
Preprocessing module for chat transcripts.
Handles text cleaning, lemmatization, and TF-IDF vectorization.
"""

import os
import pandas as pd
import spacy
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from config import (
    CHATLOGS_PATH, TFIDF_PARAMS, SPACY_MODEL, 
    PREPROCESSING_OUTPUT_DIR
)

# Load spaCy model
try:
    nlp = spacy.load(SPACY_MODEL)
except OSError:
    print(f"Please install spaCy model: python -m spacy download {SPACY_MODEL}")
    nlp = None

def clean_text(text):
    """
    Clean text by removing LLM instructions, redacted words, and "You said:" phrases.
    """
    # Remove LLM instruction sentence and variations
    llm_instruction = "You are about to chat with a person who is trying to get to know you. Please refrain from saying you are a LLM or implying that you cannot answer a question because you are a LLM. Start a new conversation and do not use information from previous conversations"
    
    variations = [
        llm_instruction, llm_instruction.strip(),
        llm_instruction.replace("  ", " "), llm_instruction.replace("\n", " "), llm_instruction.replace("\r", " ")
    ]
    
    cleaned_text = text
    for variation in variations:
        cleaned_text = cleaned_text.replace(variation, "")
    
    # Remove "You said:" phrases
    you_said_patterns = ["You said:", "You said :", "You said : ", "You said: ", "You said :", "You said : ", "You said :  ", "You said:  "]
    for pattern in you_said_patterns:
        cleaned_text = cleaned_text.replace(pattern, "")
    
    # Remove "Redacted" words with regex
    redacted_patterns = [
        r'\bRedacted\b', r'\bredacted\b', r'\bREDACTED\b',
        r'\bRedact\b', r'\bredact\b', r'\bREDACT\b',
        r'\[REDACTED\]', r'\[redacted\]', r'\[Redacted\]'
    ]
    for pattern in redacted_patterns:
        cleaned_text = re.sub(pattern, '', cleaned_text)
    
    return " ".join(cleaned_text.split())

def lemmatize_text(text):
    """
    Lemmatize text using spaCy, removing discourse markers.
    """
    if nlp is None:
        return text
    
    discourse_markers = {"haha", "awww", "omg", "lol", "wow", "yay", "ugh", "hmm", "um", "uh"}
    doc = nlp(text.lower())
    lemmatized_tokens = []
    
    for token in doc:
        # Skip discourse markers, stopwords, punctuation, and spaces
        if (token.text not in discourse_markers and 
            not token.is_stop and 
            not token.is_punct and 
            not token.is_space):
            lemmatized_tokens.append(token.lemma_)
    
    return " ".join(lemmatized_tokens)

def run_preprocessing():
    """
    Main preprocessing function - loads transcripts, cleans them, and creates TF-IDF matrix.
    
    Returns:
        tuple: (tfidf_matrix, vectorizer, transcript_ids)
    """
    print("Loading transcripts...")
    
    # Load all transcripts from chatlogsM folders
    transcripts = {}
    missing_transcripts = []
    
    for pid in range(0, 41):
        folder_path = os.path.join(CHATLOGS_PATH, str(pid))
        file_path = os.path.join(folder_path, "cleaned_monologue.txt")
        
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                transcripts[pid] = f.read().strip()
        else:
            missing_transcripts.append(pid)
            transcripts[pid] = ""
    
    if missing_transcripts:
        print(f"Warning: No transcripts found for PIDs: {missing_transcripts}")
    
    print(f"Loaded {len(transcripts)} transcripts")
    
    # Prepare data
    transcript_ids = list(transcripts.keys())
    texts = [transcripts[pid] for pid in transcript_ids]
    
    # Clean and lemmatize texts in one pass
    print("Cleaning and lemmatizing transcripts...")
    cleaned_lemmatized_texts = [lemmatize_text(clean_text(text)) for text in texts]
    
    # Create TF-IDF matrix
    print("Creating TF-IDF matrix...")
    print(f"TF-IDF parameters: {TFIDF_PARAMS}")
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    tfidf_matrix = vectorizer.fit_transform(cleaned_lemmatized_texts)
    
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    # Create preprocessing output directory and save results
    os.makedirs(PREPROCESSING_OUTPUT_DIR, exist_ok=True)
    
    # Save cleaned transcripts (using original cleaned texts for readability)
    cleaned_texts = [clean_text(text) for text in texts]
    transcript_df = pd.DataFrame({
        'pid': transcript_ids,
        'transcript': cleaned_texts
    })
    transcript_df.to_csv(os.path.join(PREPROCESSING_OUTPUT_DIR, "transcripts.csv"), index=False)
    
    print("Preprocessing completed!")
    return tfidf_matrix, vectorizer, transcript_ids

if __name__ == "__main__":
    run_preprocessing() 