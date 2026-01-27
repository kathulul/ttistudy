import os
from helper import load_chatlogs, clean_text
import pandas as pd
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import CHATLOGS_DIR
from topicmodel.openai_client import OpenAIEmbedder, wants_openai_embeddings

def prepare_chatlogs():
    chatlogs = load_chatlogs()
    chatlogs['text'] = chatlogs['text'].apply(clean_text)
    return chatlogs

