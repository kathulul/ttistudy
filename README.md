NOTES 

THE CLUSTERING STUFF IS NOT DONE.
WORK ON faceembedding.py. output.csv is the ACTUAL FACE EMBEDDINGS_CSV. work on the HBDSCAN and k-means clustering for faces, and next step: create some sort of visualization that clusters the faces together nicely. 



main.py: 
# Chatlog Anonymization Tool

A Python-based tool for anonymizing chatlogs by removing or replacing personally identifiable information (PII), gender-specific terms, and other sensitive data.

## Features

- **PII Detection and Replacement**: Uses Microsoft's Presidio Analyzer to detect and replace various types of personally identifiable information
- **Gender-Neutral Language**: Automatically replaces gendered pronouns and terms with neutral alternatives
- **Nationality/Ethnicity Handling**: Replaces nationality and ethnicity-related terms with placeholders
- **Comprehensive Logging**: Generates detailed logs of all replacements and detections
- **Configurable Confidence Threshold**: Allows adjustment of detection sensitivity

## Supported Entity Types

The tool can detect and replace the following types of information:
- Person names
- Professions
- Locations (cities, countries, states)
- Gender information
- Age
- Nationality
- Ethnicity
- Titles
- Organizations

## Installation

1. Clone the repository
2. Install required dependencies:
```bash
pip install presidio-analyzer unidecode
```

## Usage

The main script processes chatlogs from a CSV file and creates anonymized versions:

```python
python main.py
```

The script expects a CSV file with the following columns:
- ResponseId
- Gender
- Race
- Chatlog

## Output Structure

For each processed chatlog, the tool creates:
- `raw_chatlog.txt`: Original chatlog
- `presidio_detections.txt`: Detailed log of all detected entities
- `cleaned_chatlog.txt`: Anonymized version of the chatlog

## Configuration

The tool includes several configurable parameters:
- `confidence_threshold`: Minimum confidence score for entity detection (default: 0.8)
- `ENTITY_TYPES`: List of entity types to detect
- `ENTITY_REPLACEMENTS`: Mapping of entity types to replacement text
- `PRONOUN_REPLACEMENTS`: Rules for gender-neutral pronoun replacement
- `GENDER_TERM_REPLACEMENTS`: Rules for gender-neutral term replacement

## Error Handling

The tool includes comprehensive error handling for:
- Invalid input validation
- Presidio Analyzer initialization failures
- Processing errors during chatlog cleaning

## Requirements

- Python 3.x
- presidio-analyzer
- unidecode

## License

[Add your license information here] 

