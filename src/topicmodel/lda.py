""" the topic model using LDA """
import sys
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import CHATLOGS_DIR

def load_chatlogs() -> pd.DataFrame:
    """
    Load the chatlogs from the directory
    
    """
    all_texts = []
    
    # Loop through folders 0-40
    for folder_num in range(41):
        try:
            # Construct path to cleaned_monologue.txt in each folder
            file_path = f"{CHATLOGS_DIR}/{folder_num}/cleaned_monologue.txt"
            
            # Read the text file
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                all_texts.append({
                    'folder': folder_num,
                    'text': text
                })
                
        except FileNotFoundError:
            print(f"No cleaned_monologue.txt found in folder {folder_num}")
            continue
            
    # Create DataFrame from collected texts
    df = pd.DataFrame(all_texts)
    return df

    
if __name__ == "__main__":
    df = load_chatlogs()
    print(f"Loaded {len(df)} chatlogs")
    print(df.head())  # Show first 5 rows
    print(df.info())  # Show DataFrame info



