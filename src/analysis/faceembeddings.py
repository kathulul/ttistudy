""" Not optimized right now. this script is to generate the dynamic PyChart graph that plots the face embeddings and has a hover so you can see how they cluster. not optimized rn"""


import os
import pandas as pd
from deepface import DeepFace
import logging
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from PIL import Image
import ast

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_pid_from_path(image_path: str) -> int:
    """Extract PID from the parent directory name."""
    return int(os.path.basename(os.path.dirname(image_path)))

def get_image_name(image_path: str) -> str:
    """Extract image name without extension."""
    return os.path.splitext(os.path.basename(image_path))[0]

def get_embedding(image_path: str) -> np.ndarray:
    """Get the face embedding for an image using DeepFace."""
    try:
        embedding = DeepFace.represent(
            img_path=image_path,
            model_name='Facenet',
            enforce_detection=False,
            detector_backend='retinaface'
        )
        # DeepFace.represent returns a list of dicts, get the embedding vector
        return embedding[0]['embedding']
    except Exception as e:
        logger.error(f"Error getting embedding for {image_path}: {e}")
        return None

def process_images(base_directory: str, output_file: str = 'face_embeddings.csv') -> pd.DataFrame:
    """
    Process images and save their embeddings to a CSV file.
    
    Args:
        base_directory: Directory containing the images
        output_file: Path to save the embeddings CSV
        
    Returns:
        DataFrame containing the processed embeddings
    """
    results = []
    for root, dirs, files in os.walk(base_directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                try:
                    pid = get_pid_from_path(image_path)
                except Exception:
                    logger.warning(f"Could not extract PID from {image_path}, skipping.")
                    continue
                image_name = get_image_name(image_path)
                embedding = get_embedding(image_path)
                if embedding is not None:
                    embedding_str = ','.join([str(x) for x in embedding])
                    results.append({
                        'PID': pid,
                        'Image': image_name,
                        'Embedding': embedding_str
                    })
                    logger.info(f"Processed {image_path}")
                else:
                    logger.warning(f"Failed to get embedding for {image_path}")
    
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values(['PID', 'Image'])
        df.to_csv(output_file, index=False)
        logger.info(f"Embeddings saved to {output_file}")
        return df
    else:
        logger.error("No embeddings to save.")
        return pd.DataFrame()

def perform_clustering(df: pd.DataFrame, eps: float = 0.3, min_samples: int = 3) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Perform clustering on face embeddings using DBSCAN and visualize with t-SNE.
    
    Args:
        df: DataFrame containing the embeddings
        eps: The maximum distance between two samples for them to be considered neighbors
        min_samples: The number of samples in a neighborhood for a point to be considered a core point
        
    Returns:
        Tuple of (results DataFrame, cluster labels, 2D embeddings for visualization)
    """
    # Convert string embeddings to numpy arrays and normalize
    embeddings = np.array([ast.literal_eval(emb) for emb in df['Embedding']])
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Perform clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(embeddings)
    labels = clustering.labels_
    
    # Add cluster information to results
    results_df = df.copy()
    results_df['Cluster'] = labels
    
    # Perform t-SNE for visualization
    tsne = TSNE(n_components=2, random_state=42, metric='cosine')
    embeddings_2d = tsne.fit_transform(embeddings)
    
    return results_df, labels, embeddings_2d

def visualize_clusters(results_df: pd.DataFrame, labels: np.ndarray, embeddings_2d: np.ndarray, 
                      eps: float = 0.3, min_samples: int = 3, output_file: str = 'face_clusters.png') -> None:
    """
    Create and save a visualization of the clusters with interactive image hover.
    
    Args:
        results_df: DataFrame containing the results
        labels: Cluster labels
        embeddings_2d: 2D embeddings for visualization
        eps: Eps parameter used for clustering
        min_samples: Min_samples parameter used for clustering
        output_file: Path to save the visualization
    """
    # Create the main window
    root = tk.Tk()
    root.title("Face Clusters Visualization")
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Create scatter plot
    scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                        c=labels, cmap='viridis', alpha=0.6)
    
    # Add colorbar
    plt.colorbar(scatter, label='Cluster')
    
    # Add labels and title
    ax.set_title(f'Face Clusters (t-SNE visualization)\neps={eps}, min_samples={min_samples}')
    ax.set_xlabel('t-SNE dimension 1')
    ax.set_ylabel('t-SNE dimension 2')
    ax.grid(True, alpha=0.3)
    
    # Store the annotation
    annotation = None
    
    def hover(event):
        nonlocal annotation
        if event.inaxes != ax:
            return

        # Get the index of the point closest to the mouse
        cont, ind = scatter.contains(event)
        if cont:
            pos = ind["ind"][0]
            x, y = embeddings_2d[pos]
            
            # Remove previous annotation if it exists
            if annotation:
                annotation.remove()
            
            # Get the image path from the results DataFrame
            pid = results_df.iloc[pos]['PID']
            image_name = results_df.iloc[pos]['Image']
            image_path = os.path.join("/Users/katherineliu/Documents/ttistudy/images", 
                                    str(pid), f"{image_name}.png")
            
            # Load and display the image
            try:
                img = Image.open(image_path)
                img = img.resize((100, 100))  # Resize image for display
                imagebox = OffsetImage(img, zoom=1)
                annotation = AnnotationBbox(imagebox, (x, y),
                                         xybox=(50, 50),
                                         xycoords='data',
                                         boxcoords="offset points",
                                         pad=0.5,
                                         arrowprops=dict(arrowstyle="->"))
                ax.add_artist(annotation)
                fig.canvas.draw_idle()
            except Exception as e:
                logger.error(f"Error loading image: {e}")
    
    # Connect the hover event
    fig.canvas.mpl_connect("motion_notify_event", hover)
    
    # Embed the plot in the tkinter window
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    
    # Save the static visualization
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Clustering visualization saved as '{output_file}'")
    
    # Start the tkinter event loop
    root.mainloop()

def print_cluster_statistics(results_df: pd.DataFrame) -> None:
    """
    Print statistics about the clustering results.
    
    Args:
        results_df: DataFrame containing the clustering results
    """
    labels = results_df['Cluster']
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    logger.info(f"Number of clusters: {n_clusters}")
    logger.info(f"Number of noise points: {n_noise}")
    
    # Print cluster sizes
    cluster_sizes = results_df['Cluster'].value_counts().sort_index()
    logger.info("\nCluster sizes:")
    for cluster, size in cluster_sizes.items():
        logger.info(f"Cluster {cluster}: {size} images")
    
    # Print distribution of images in each cluster
    logger.info("\nDistribution of images in each cluster:")
    print(pd.crosstab(results_df['Cluster'], results_df['Image']))
    
    # Print which images from the same PID ended up in the same cluster
    logger.info("\nImages from the same PID in the same cluster:")
    same_cluster_pids = results_df.groupby(['PID', 'Cluster']).size().reset_index(name='count')
    same_cluster_pids = same_cluster_pids[same_cluster_pids['count'] > 1]
    print(same_cluster_pids)

def save_dataframe(df: pd.DataFrame, filename: str = 'face_embeddings.csv') -> None:
    """
    Save the DataFrame to a CSV file.
    
    Args:
        df: DataFrame to save
        filename: Name of the file to save to
    """
    df.to_csv(filename, index=False)
    logger.info(f"DataFrame saved to {filename}")

def load_or_process_images(base_directory: str, cache_file: str = 'face_embeddings.csv') -> pd.DataFrame:
    """
    Load the DataFrame from cache if it exists, otherwise process images.
    
    Args:
        base_directory: Directory containing the images
        cache_file: Path to the cache file
        
    Returns:
        DataFrame containing the processed embeddings
    """
    if os.path.exists(cache_file):
        logger.info(f"Loading cached embeddings from {cache_file}")
        return pd.read_csv(cache_file)
    else:
        logger.info("No cache found, processing images...")
        df = process_images(base_directory, cache_file)
        return df

if __name__ == "__main__":
    # Load or process images
    df = load_or_process_images(base_directory="/Users/katherineliu/Documents/ttistudy/images")
    
    # Perform clustering on the embeddings
    results_df, labels, embeddings_2d = perform_clustering(df)
    visualize_clusters(results_df, labels, embeddings_2d)
    print_cluster_statistics(results_df)
