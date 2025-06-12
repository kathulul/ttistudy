import os
from deepface import DeepFace
import logging
from typing import Optional, Tuple, Dict
import cv2
import numpy as np
from mtcnn import MTCNN
from config.settings import CONFIDENCE_THRESHOLD
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_pid_from_path(image_path: str) -> Optional[int]:
    """Extract PID from the parent directory name."""
    try:
        parent_dir = os.path.basename(os.path.dirname(image_path))
        return int(parent_dir)
    except (ValueError, IndexError):
        return None
    

def analyze_face(image_path: str) -> Optional[Dict]:
    """
    Analyze a single face image using DeepFace.
    Returns analysis results if successful, None otherwise.
    """
    try:
        # Analyze the image
        result = DeepFace.analyze(
            img_path=image_path,
            actions=['gender', 'race', 'age', 'emotion'],
            enforce_detection=False,  # Don't fail if face detection is uncertain
            detector_backend='retinaface'  # Use RetinaFace for better detection
        )
        
        # Extract relevant information
        analysis = {
            'gender': result[0]['dominant_gender'],
            'race': result[0]['dominant_race'],
            'age': result[0]['age'],
            'emotion': result[0]['dominant_emotion']
        }
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing {image_path}: {str(e)}")
        logger.error(f"Full error: {e.__class__.__name__}: {str(e)}")
        return None

def process_images(base_directory: str, output_file: str = 'face_analysis.csv') -> None:
    """
    Process all images in the directory and save results to CSV.
    """
    results = []
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(base_directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                pid = get_pid_from_path(image_path)
                
                if pid is None:
                    logger.warning(f"Could not extract PID from path {image_path}, skipping")
                    continue
                
                analysis = analyze_face(image_path)
                
                if analysis:
                    results.append({
                        'PID': pid,
                        'Image': os.path.splitext(os.path.basename(image_path))[0],
                        'Gender': analysis['gender'],
                        'Race': analysis['race'],
                        'Age': analysis['age'],
                        'Emotion': analysis['emotion']
                    })
                    logger.info(f"Processed {image_path}: Gender={analysis['gender']}, Race={analysis['race']}, Age={analysis['age']}, Emotion={analysis['emotion']}")
                else:
                    logger.warning(f"Failed to analyze {image_path}")
    
    # Create DataFrame and save to CSV
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values(['PID', 'Image'])
        df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
    else:
        logger.error("No successful analyses to save")

if __name__ == "__main__":
    process_images()
