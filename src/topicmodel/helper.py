import os
import sys
import pandas as pd
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import CHATLOGS_DIR

def load_chatlogs() -> pd.DataFrame:
    """
    Load the chatlogs from the directory.
    
    Returns:
        DataFrame with columns 'folder' and 'text'
    
    Raises:
        FileNotFoundError: If CHATLOGS_DIR doesn't exist
        ValueError: If no chatlogs are found
    """
    if not os.path.exists(CHATLOGS_DIR):
        raise FileNotFoundError(f"Chatlogs directory not found: {CHATLOGS_DIR}")
    
    all_texts = []
    
    # Loop through folders 0-40
    for folder_num in range(41):
        try:
            # Construct path to cleaned_monologue.txt in each folder
            file_path = os.path.join(CHATLOGS_DIR, str(folder_num), "cleaned_monologue.txt")
            
            # Read the text file
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    if text.strip():  # Only add non-empty texts
                        all_texts.append({
                            'folder': folder_num,
                            'text': text
                        })
            
        except (FileNotFoundError, IOError, UnicodeDecodeError) as e:
            print(f"Warning: Could not read folder {folder_num}: {e}")
            continue
    
    if len(all_texts) == 0:
        raise ValueError(f"No chatlogs found in {CHATLOGS_DIR}")
    
    chatlogs = pd.DataFrame(all_texts)
    return chatlogs


def clean_text(
    text: str,
    remove_redacted: bool = True,
    remove_you_said: bool = True
) -> str:
  
    # Remove "You said:" markers 
    if remove_you_said:
        text = re.sub(r'You said:\s*', '', text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Remove [REDACTED] tokens
    if remove_redacted:
        text = re.sub(r'\[REDACTED\]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

