import os
import pandas as pd
from deepface import DeepFace
import logging
from typing import Dict, Tuple, Optional
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_pid_from_path(image_path: str) -> Optional[int]:
    """Extract PID from the parent directory name."""
    try:
        # Split path and get the parent directory name
        parent_dir = os.path.basename(os.path.dirname(image_path))
        return int(parent_dir)
    except (ValueError, IndexError):
        return None

def analyze_face(image_path: str) -> Tuple[Dict, float]:
    """
    Analyze a single face image using DeepFace.
    Returns a tuple of (analysis results, confidence score).
    """
    try:
        # Analyze the image
        result = DeepFace.analyze(
            img_path=image_path,
            actions=['gender', 'race'],
            enforce_detection=True,
            detector_backend='opencv'
        )
        
        # Extract relevant information
        analysis = {
            'gender': result[0]['dominant_gender'],
            'race': result[0]['dominant_race']
        }
        
        # Calculate average confidence score
        confidence = (result[0]['gender']['confidence'] + result[0]['race']['confidence']) / 2
        
        return analysis, confidence
        
    except Exception as e:
        logger.error(f"Error analyzing {image_path}: {str(e)}")
        return None, 0.0

def process_images(base_directory: str = 'images', output_file: str = 'face_analysis.csv') -> None:
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
                
                analysis, confidence = analyze_face(image_path)
                
                if analysis:
                    results.append({
                        'PID': pid,
                        'Gender': analysis['gender'],
                        'Race': analysis['race'],
                        'Confidence': confidence
                    })
                    logger.info(f"Processed {image_path}: Gender={analysis['gender']}, Race={analysis['race']}, Confidence={confidence:.2f}")
                else:
                    logger.warning(f"Failed to analyze {image_path}")
    
    # Create DataFrame and save to CSV
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values('PID')
        df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
    else:
        logger.error("No successful analyses to save")

if __name__ == "__main__":
    process_images()
