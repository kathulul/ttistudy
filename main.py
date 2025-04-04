import csv 
import os  
import re
import unidecode
from presidio_analyzer import AnalyzerEngine
from typing import Optional
from io import StringIO
from collections import defaultdict

# Initialize Presidio analyzer at module level (singleton)
try:
    # Initialize analyzer with default settings
    ANALYZER = AnalyzerEngine()
except Exception as e:
    raise RuntimeError(f"Failed to initialize Presidio Analyzer: {e}")

# Constants already defined at module level (good!)
ENTITY_TYPES = [
    "PERSON", "PROFESSION", "LOCATION", "CITY", "COUNTRY", 
    "STATE", "GENDER", "AGE", "NATIONALITY", "ETHNICITY",
    "TITLE", "ORGANIZATION"
]

ENTITY_REPLACEMENTS = {
    "PERSON": "[PERSON]",
    "PROFESSION": "[OCCUPATION]",
    "LOCATION": "[LOCATION]",
    "CITY": "[LOCATION]",
    "COUNTRY": "[LOCATION]",
    "STATE": "[LOCATION]",
    "GENDER": "[GENDER]",
    "AGE": "[AGE]",
    "NATIONALITY": "[NATIONALITY]",
    "ETHNICITY": "[ETHNICITY]",
    "TITLE": "[OCCUPATION]",
    "ORGANIZATION": "[ORGANIZATION]",
}

# Cache compiled regex patterns
PRONOUN_REPLACEMENTS = {
    r'\b(he|she)\b': lambda m: 'they' if m.group().islower() else 'They',
    r'\b(him|her)\b': lambda m: 'them' if m.group().islower() else 'Them',
    r'\b(his|hers)\b': lambda m: 'their' if m.group().islower() else 'Their',
    r'\b(himself|herself)\b': lambda m: 'themself' if m.group().islower() else 'Themself'
}

GENDER_TERM_REPLACEMENTS = {
    r'\b(man|woman)\b': 'person',
    r'\b(boy|girl)\b': 'person',
    r'\b(brother|sister)\b': 'sibling',
    r'\b(son|daughter)\b': 'child',
    r'\b(mother|father)\b': 'parent',
    r'\b(gay|lesbian)\b': 'queer',
}

# Cache compiled patterns
COMPILED_PRONOUN_PATTERNS = {
    pattern: re.compile(pattern) 
    for pattern in PRONOUN_REPLACEMENTS.keys()
}

COMPILED_GENDER_PATTERNS = {
    pattern: re.compile(pattern, flags=re.IGNORECASE) 
    for pattern in GENDER_TERM_REPLACEMENTS.keys()
}

# Cache unwanted patterns
UNWANTED_PATTERNS = [
    r'(?m)^(?:(?:GPT-?[0-9])|(?:[0-9]+(?:\.[0-9]+)?(?:o|-?turbo)?))\s*(?:mini|plus|max|turbo)?\s*$',
    r'(?m)^Search\s*$',
    r'ChatGPT can make mistakes\. Check important info\.',
    r'(?m)^\d+/\d+\s*$'
]

COMPILED_UNWANTED_PATTERNS = [
    re.compile(pattern) for pattern in UNWANTED_PATTERNS
]

def validate_input(text: str, function_name: str) -> None:
    """Validate input text."""
    if not isinstance(text, str):
        raise ValueError(f"{function_name}: Input must be a string")
    if not text.strip():
        raise ValueError(f"{function_name}: Input cannot be empty")

def log_presidio_detections(text: str, confidence_threshold: float = 0.8) -> str:
    """
    Analyzes text with Presidio and returns a formatted detection log.
    
    Args:
        text (str): Text to analyze
        confidence_threshold (float): Minimum confidence score for detections (0.0 to 1.0)
    
    Returns:
        str: Formatted detection log
    
    Raises:
        ValueError: If input is invalid
        Exception: If Presidio analysis fails
    """
    try:
        # Validate input
        validate_input(text, "log_presidio_detections")
        
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1")
        
        # Create detection log
        log_buffer = StringIO()
        log_buffer.write("PRESIDIO DETECTION LOG\n")
        log_buffer.write("=" * 50 + "\n")
        log_buffer.write(f"Confidence Threshold: {confidence_threshold}\n")
        log_buffer.write("=" * 50 + "\n")
        
        # Analyze text for all entities using module-level analyzer
        results = ANALYZER.analyze(
            text=text,
            entities=ENTITY_TYPES,
            language='en'
        )
        
        # Filter and sort results by confidence
        filtered_results = [r for r in results if r.score >= confidence_threshold]
        filtered_results.sort(key=lambda x: (-x.score, x.start))  # Sort by confidence (desc) and position
        
        if not filtered_results:
            log_buffer.write("\nNo entities detected above confidence threshold.\n")
        else:
            for result in filtered_results:
                detected_text = text[result.start:result.end]
                log_buffer.write(f"\nEntity Type: {result.entity_type}")
                log_buffer.write(f"Detected Text: '{detected_text}'")
                log_buffer.write(f"Confidence Score: {result.score:.3f}")
                log_buffer.write(f"Position: {result.start}-{result.end}")
                if result.entity_type in ENTITY_REPLACEMENTS:
                    log_buffer.write(f"Replacement: '{ENTITY_REPLACEMENTS[result.entity_type]}'")
                log_buffer.write("-" * 30)
        
        return log_buffer.getvalue()
        
    except Exception as e:
        error_msg = f"Error in log_presidio_detections: {str(e)}"
        raise Exception(error_msg) from e

def clean_chatlog(chatlog: str, confidence_threshold: float = 0.8, progress_callback=None) -> str:
    """
    Clean chatlog by removing identifiable information and standardizing format.
    
    Args:
        chatlog (str): The input chatlog text
        confidence_threshold (float): Minimum confidence score for entity detection (0.0 to 1.0)
    
    Returns:
        str: Cleaned chatlog text with:
            - Identifiable entities replaced with placeholders
            - Gendered pronouns replaced with neutral alternatives
            - Gender-specific terms replaced with neutral alternatives
            - Unwanted lines removed
            - Multiple newlines cleaned up
    
    Raises:
        ValueError: If input is invalid or empty
        RuntimeError: If cleaning process fails
    """
    try:
        validate_input(chatlog, "clean_chatlog")
        
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1")
        
        # Additional sanitization
        chatlog = chatlog.strip()  # Remove leading/trailing whitespace
        chatlog = re.sub(r'\r\n?', '\n', chatlog)  # Normalize line endings
        
        # Convert to ASCII using unidecode
        cleaned_text = unidecode.unidecode(chatlog)
        
        # Remove unwanted lines using cached patterns
        for pattern in COMPILED_UNWANTED_PATTERNS:
            cleaned_text = pattern.sub('', cleaned_text)
        
        # Analyze text for all entities using module-level analyzer
        results = ANALYZER.analyze(
            text=cleaned_text,
            entities=ENTITY_TYPES,
            language='en'
        )
        
        # Filter results by confidence threshold
        results = [r for r in results if r.score >= confidence_threshold]
        
        # Sort results in reverse order to prevent index shifting
        results = sorted(results, key=lambda x: x.start, reverse=True)
        
        # Replace entities
        replacement_counts = defaultdict(int)
        for result in results:
            if result.entity_type in ENTITY_REPLACEMENTS:
                replacement = ENTITY_REPLACEMENTS[result.entity_type]
                cleaned_text = cleaned_text[:result.start] + replacement + cleaned_text[result.end:]
                replacement_counts[result.entity_type] += 1
        
        # Process remaining replacements using cached patterns
        for pattern, compiled_pattern in COMPILED_PRONOUN_PATTERNS.items():
            cleaned_text = compiled_pattern.sub(PRONOUN_REPLACEMENTS[pattern], cleaned_text)
        
        for pattern, compiled_pattern in COMPILED_GENDER_PATTERNS.items():
            cleaned_text = compiled_pattern.sub(GENDER_TERM_REPLACEMENTS[pattern], cleaned_text)
        
        # Clean up multiple newlines
        cleaned_text = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_text)
        
        if progress_callback:
            progress_callback(0.5, "Processing entity replacements")
        
        if replacement_counts:
            print(f"Replacement statistics: {dict(replacement_counts)}")
        
        return cleaned_text
        
    except Exception as e:
        error_msg = f"Error in clean_chatlog: {str(e)}"
        raise Exception(error_msg) from e

def process_chatlogs(p_csv:str):
    # First loop: Process the CSV and create directories
    participant_info = {}
    with open(p_csv, 'r') as file:
        csv_reader = csv.DictReader(file)
        i = 0
        
        for row in csv_reader:
            if i < 2:
                i += 1
                continue
            response_id = row['ResponseId']
            participant_info[response_id] = {
                'Gender': row['Gender'],
                'Race': row['Race'],
                'Chatlog': row['Chatlog']
            }
            # Create directory for this participant if it doesn't exist
            pid_dir = os.path.join('chatlogs', str(i - 2))
            os.makedirs(pid_dir, exist_ok=True)
            
            # Save raw chatlog to a file
            chatlog_path = os.path.join(pid_dir, 'raw_chatlog.txt')
            with open(chatlog_path, 'w') as chatlog_file:
                chatlog_file.write(row['Chatlog'])
            i += 1

    # Second loop: Process each directory
    for pid in os.listdir('chatlogs'):
        pid_path = os.path.join('chatlogs', pid)
        if os.path.isdir(pid_path):
            chatlog_file_path = os.path.join(pid_path, 'raw_chatlog.txt')
            if os.path.exists(chatlog_file_path):
                try:
                    with open(chatlog_file_path, 'r') as chatlog_file:
                        chatlog_content = chatlog_file.read().strip()
                        
                        # Skip empty chatlogs
                        if not chatlog_content:
                            print(f"Warning: Empty chatlog found in {pid_path}, skipping...")
                            continue
                        
                        # Get the detection log
                        detection_log_output = log_presidio_detections(chatlog_content)
                        
                        # Save detection log for this directory
                        detection_log_path = os.path.join(pid_path, 'presidio_detections.txt')
                        with open(detection_log_path, 'w') as log_file:
                            log_file.write(detection_log_output)
                        
                        # Clean the chatlog
                        cleaned_chatlog = clean_chatlog(chatlog_content)
                        
                        # Save cleaned chatlog
                        cleaned_chatlog_path = os.path.join(pid_path, 'cleaned_chatlog.txt')
                        with open(cleaned_chatlog_path, 'w') as cleaned_file:
                            cleaned_file.write(cleaned_chatlog)
                            
                except Exception as e:
                    print(f"Error processing chatlog in {pid_path}: {str(e)}")
                    continue

if __name__ == "__main__":
    process_chatlogs("data/testdata.csv")

