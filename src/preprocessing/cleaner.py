"""Module for processing and cleaning chatlogs.

This module provides functionality for:
- Creating directory structures for chatlogs
- Processing and cleaning chatlog content
- Detecting and redacting sensitive information using Presidio
- Generating detection logs
- Handling gender-neutral language conversion

Example:
    >>> from preprocessing.cleaner import create_directories, process_chatlogs
    >>> create_directories("data/chatlogs.csv", "output_dir")
    >>> process_chatlogs("output_dir")
"""

import csv 
import os  
import re
import unidecode
from presidio_analyzer import AnalyzerEngine
from io import StringIO
from collections import defaultdict

# Initialize Presidio Analyzer
try:
    ANALYZER = AnalyzerEngine()
except Exception as e:
    raise RuntimeError(f"Failed to initialize Presidio Analyzer: {e}")

# Entity types for detection
ENTITY_TYPES = [
    "PERSON", "PROFESSION", "LOCATION", "CITY", "COUNTRY", 
    "STATE", "GENDER", "AGE", "NATIONALITY", "ETHNICITY",
    "TITLE", "ORGANIZATION"
]

ENTITY_REPLACEMENTS = {
    "PERSON": "[REDACTED]",
    "PROFESSION": "[REDACTED]",
    "LOCATION": "[REDACTED]",
    "CITY": "[REDACTED]",
    "COUNTRY": "[REDACTED]",
    "STATE": "[REDACTED]",
    "GENDER": "[REDACTED]",
    "AGE": "[REDACTED]",
    "NATIONALITY": "[REDACTED]",
    "ETHNICITY": "[REDACTED]",
    "TITLE": "[REDACTED]",
    "ORGANIZATION": "[REDACTED]",
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
    r'\b(boyfriend|girlfriend)\b': 'partner',
    r'\b(king|queen)\b': '[REDACTED]',
    r'\b(lord|lady)\b': 'person',
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

NATIONALITY_ETHNICITY_TERMS = {
    # Nationalities and their derivatives (-ish, -ic, -ese, -i, -ian, etc.)
    r'\b(?:American(?:ized?)?|Canadian|French|Brit(?:ish|ain)|German(?:ic)?|Italian(?:ate)?|Span(?:ish|iard)|Mexic(?:an|o)|Brazil(?:ian)?|Chin(?:ese|a)|Japan(?:ese)?|Korean?|Ind(?:ian|ic)|Russ(?:ian?)|Austral(?:ian?)|Ir(?:ish|eland)|Scot(?:tish|land)|Welsh|Dutch|Pol(?:ish|and)|Swed(?:ish|en)|Norweg(?:ian)?|Dan(?:ish)|Finn(?:ish)|Turk(?:ish)?|Greek?|Portugal|Portuguese|Vietnam(?:ese)?|Thai(?:land)?|Filip(?:ino)?|Indonesia(?:n)?|Malay(?:sian)?|Niger(?:ian)?|Kenya(?:n)?|Ethiopia(?:n)?|Egypt(?:ian)?|Iran(?:ian)?|Iraq(?:i)?|Saudi|Israel(?:i)?)\b': '[REDACTED]',
    # Ethnic/Racial terms and derivatives
    r'\b(?:Lat(?:in[oa]|inx)|Hispanic|Asian?|Caucasian|White|Black|African?|Europe(?:an)?|Middle[- ]Eastern|Pacific Islander|Native American|Indigenous|Aborig(?:inal|ine)|Inuit|Muslim|Christian|Hindu|Buddhist|Sikh|Jain|Zoroastrian|Jewish|Muslim|Christian|Hindu|Buddhist|Sikh|Jain|Zoroastrian)\b': '[REDACTED]',
    # Regional identifiers and derivatives
    r'\b(?:West(?:ern)?|East(?:ern)?|North(?:ern)?|South(?:ern)?|Mediterranean|Nord(?:ic)?|Balt(?:ic)?|Slav(?:ic)?|Anglo(?:-.*)?|Celt(?:ic)?|German(?:ic)?|Latin[oa]?|Asia(?:n|tic)?|Afric(?:an)?|Europe(?:an)?|America(?:n)?)\b': '[REDACTED]',
}


def log_presidio_detections(text: str, confidence_threshold: float = 0.8) -> tuple[str, dict]:
    """
    Analyzes text with Presidio and regex patterns, returns formatted detection log with confidence intervals, statistics, and detected entities. 

    Args:
        text (str): The input text to analyze
        confidence_threshold (float): Minimum confidence score for entity detection (0.0 to 1.0)
    
    Returns:
        tuple[str, dict]: A tuple containing the formatted detection log and a dictionary of detected entities
    """
    try:      
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1")
        
        # Create detection log
        log_buffer = StringIO()
        log_buffer.write("PRESIDIO DETECTION LOG\n")
        log_buffer.write("=" * 50 + "\n")
        log_buffer.write(f"Confidence Threshold: {confidence_threshold}\n")
        log_buffer.write("=" * 50 + "\n")
        
        # Create dictionary to store detected entities
        detected_entities = defaultdict(set)
        replacement_counts = defaultdict(int)
        
        # Presidio Detections
        results = ANALYZER.analyze(
            text=text,
            entities=ENTITY_TYPES,
            language='en'
        )
        
        # Filter and sort results by confidence
        filtered_results = [r for r in results if r.score >= confidence_threshold]
        filtered_results.sort(key=lambda x: (-x.score, x.start))
        
        if not filtered_results:
            log_buffer.write("\nNo Presidio entities detected above confidence threshold.\n")
        else:
            for result in filtered_results:
                detected_text = text[result.start:result.end]
                # Add to detection log
                log_buffer.write(f"\nEntity Type: {result.entity_type}")
                log_buffer.write(f"Detected Text: '{detected_text}'")
                log_buffer.write(f"Confidence Score: {result.score:.3f}")
                log_buffer.write(f"Position: {result.start}-{result.end}")
                if result.entity_type in ENTITY_REPLACEMENTS:
                    log_buffer.write(f"Replacement: '{ENTITY_REPLACEMENTS[result.entity_type]}'")
                log_buffer.write("-" * 30)
                
                # Add to entity dictionary
                detected_entities[result.entity_type].add(detected_text)
                replacement_counts[result.entity_type] += 1
        
        # Gender Term Replacements
        log_buffer.write("\nGENDER TERM REPLACEMENTS\n")
        log_buffer.write("=" * 50 + "\n")
        
        for pattern, replacement in GENDER_TERM_REPLACEMENTS.items():
            compiled_pattern = re.compile(pattern, re.IGNORECASE)
            for match in compiled_pattern.finditer(text):
                detected_text = match.group(0)
                start, end = match.span()
                
                log_buffer.write(f"\nEntity Type: GENDER_TERM")
                log_buffer.write(f"Detected Text: '{detected_text}'")
                log_buffer.write(f"Position: {start}-{end}")
                log_buffer.write(f"Replacement: '{replacement}'")
                log_buffer.write("-" * 30)
        
        # Pronoun Replacements log
        log_buffer.write("\nPRONOUN REPLACEMENTS\n")
        log_buffer.write("=" * 50 + "\n")
        
        for pattern, replacement_func in PRONOUN_REPLACEMENTS.items():
            compiled_pattern = re.compile(pattern)
            for match in compiled_pattern.finditer(text):
                detected_text = match.group(0)
                start, end = match.span()
                # Get the actual replacement by calling the lambda function
                replacement = replacement_func(match)
                
                log_buffer.write(f"\nEntity Type: PRONOUN")
                log_buffer.write(f"Detected Text: '{detected_text}'")
                log_buffer.write(f"Position: {start}-{end}")
                log_buffer.write(f"Replacement: '{replacement}'")
                log_buffer.write("-" * 30)
        
        # 3. Nationality/Ethnicity Replacements
        log_buffer.write("\nNATIONALITY/ETHNICITY REPLACEMENTS\n")
        log_buffer.write("=" * 50 + "\n")
        
        for pattern, replacement in NATIONALITY_ETHNICITY_TERMS.items():
            compiled_pattern = re.compile(pattern, re.IGNORECASE)
            for match in compiled_pattern.finditer(text):
                detected_text = match.group(0)
                start, end = match.span()
                
                log_buffer.write(f"\nEntity Type: {replacement.strip('[]')}")
                log_buffer.write(f"Detected Text: '{detected_text}'")
                log_buffer.write(f"Position: {start}-{end}")
                log_buffer.write(f"Replacement: '{replacement}'")
                log_buffer.write("-" * 30)
        
        # Add statistics section
        log_buffer.write("\nREPLACEMENT STATISTICS\n")
        log_buffer.write("=" * 50 + "\n")
        for entity_type, count in replacement_counts.items():
            log_buffer.write(f"{entity_type}: {count} replacements\n")
        
        log_output = log_buffer.getvalue()
        log_buffer.close()  # Free memory
        return log_output, dict(detected_entities)
        
    except Exception as e:
        error_msg = f"Error in log_presidio_detections: {str(e)}"
        raise Exception(error_msg) from e

def clean_chatlog(chatlog: str, detected_entities: dict, confidence_threshold: float = 0.8, progress_callback=None) -> str:
    """
    Clean chatlog using both Presidio replacements and removing all instances of detected entities.
    
    Args:
        chatlog (str): The input chatlog text
        detected_entities (dict): Dictionary of detected entities to remove
        confidence_threshold (float): Minimum confidence score for entity detection (0.0 to 1.0)
        progress_callback (callable, optional): Callback function to report progress
    
    Returns:
        str: Cleaned chatlog text with:
            - Identifiable entities replaced with placeholders
            - Gendered pronouns replaced with neutral alternatives
            - Gender-specific terms replaced with neutral alternatives
            - Unwanted lines removed
            - Multiple newlines cleaned up
    
    Raises:
        ValueError: If confidence threshold is invalid
        RuntimeError: If cleaning process fails
    """
    try:
        
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1")
        
        # Do all existing cleaning first
        cleaned_text = chatlog.strip()  # Remove leading/trailing whitespace
        cleaned_text = re.sub(r'\r\n?', '\n', cleaned_text) # Normalize line endings
        cleaned_text = unidecode.unidecode(cleaned_text) # Convert to ASCII using unidecode
        
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
        
        # Replace all instances of detected entities with [REDACTED]
        for entity_type, words in detected_entities.items():
            for word in words:
                # Create pattern that matches whole word case-insensitively
                pattern = rf'\b{re.escape(word)}\b'
                cleaned_text = re.sub(pattern, '[REDACTED]', cleaned_text, flags=re.IGNORECASE)
        
        # Apply nationality/ethnicity replacements
        for pattern, replacement in NATIONALITY_ETHNICITY_TERMS.items():
            cleaned_text = re.sub(pattern, replacement, cleaned_text, flags=re.IGNORECASE)
        
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

def create_directories(p_csv: str, directory: str) -> None:
    """Create directory structure and save raw chatlogs from CSV file.
    
    This function:
    1. Reads the input CSV file
    2. Creates a directory for each participant
    3. Saves their raw chatlog to a file
    
    Args:
        p_csv (str): Path to the CSV file containing chatlogs
        directory (str): Base directory to store chatlogs
        
    Raises:
        FileNotFoundError: If the CSV file cannot be found
        ValueError: If the CSV file is empty or malformed
        
    Example:
        >>> create_directories("data/chatlogs.csv", "output_dir")
    """
    participant_info = {}
    with open(p_csv, 'r') as file:
        csv_reader = csv.DictReader(file)
        i = 0
        
        for row in csv_reader:
            # Skip header rows
            if i < 2:
                i += 1
                continue
                
            # Store participant information
            response_id = row['ResponseId']
            participant_info[response_id] = {
                'Gender': row['Gender'],
                'Race': row['Race'],
                'Chatlog': row['Chatlog']
            }
            
            # Create directory for participant
            pid_dir = os.path.join(directory, str(i - 2))
            os.makedirs(pid_dir, exist_ok=True)
            
            # Save raw chatlog to a file. COMMMENT OUT IF JUST CREATING DIRECTORIES 
            chatlog_path = os.path.join(pid_dir, 'raw_chatlog.txt')
            with open(chatlog_path, 'w') as chatlog_file:
                chatlog_file.write(row['Chatlog'])
            i += 1

def process_chatlogs(directory: str) -> None:
    """Process chatlogs in the specified directory.
    
    This function:
    1. Processes each chatlog in the directory
    2. Detects sensitive information using Presidio
    3. Generates detection logs
    4. Cleans the chatlog content
    5. Saves processed files
    
    Args:
        directory (str): Base directory containing chatlogs
        
    Raises:
        FileNotFoundError: If the directory doesn't exist
        ValueError: If chatlog files are empty or malformed
        
    Example:
        >>> process_chatlogs("output_dir")
    """
    for pid in os.listdir(directory):
        pid_path = os.path.join(directory, pid)
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
                        
                        # Get both the detection log and entity dictionary
                        detection_log_output, detected_entities = log_presidio_detections(chatlog_content)
                        
                        # Save detection log
                        detection_log_path = os.path.join(pid_path, 'presidio_detections.txt')
                        with open(detection_log_path, 'w') as log_file:
                            log_file.write(detection_log_output)
                        
                        # Clean the chatlog
                        cleaned_chatlog = clean_chatlog(chatlog_content, detected_entities)
                        
                        # Save cleaned chatlog
                        cleaned_chatlog_path = os.path.join(pid_path, 'cleaned_chatlog.txt')
                        with open(cleaned_chatlog_path, 'w') as cleaned_file:
                            cleaned_file.write(cleaned_chatlog)
                            
                except Exception as e:
                    print(f"Error processing chatlog in {pid_path}: {str(e)}")
                    continue