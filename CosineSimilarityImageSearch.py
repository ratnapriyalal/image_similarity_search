import cv2
import numpy as np
import os
import requests
from io import BytesIO
from PIL import Image

import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import tf_keras
import cv2
import matplotlib.pyplot as plt
import os
from sklearn.metrics.pairwise import cosine_similarity

# Set environment variable for legacy Keras compatibility
os.environ['TF_USE_LEGACY_KERAS'] = '1'

# Define model URL
TF_MODEL_URL = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'  # Feature extraction model
IMAGE_URL_LIST = image_url_list.copy()  # List of image URLs
IMAGE_URL = image_url_list[0]

# Load the feature extractor model
feature_extractor_layer = hub.KerasLayer(TF_MODEL_URL, trainable=False, name="feature_extractor")

# Build a Keras model for feature extraction
feature_extractor = tf_keras.Sequential([tf_keras.Input(shape=(224, 224, 3)), feature_extractor_layer])

# Function to download and preprocess an image from a URL
def load_and_preprocess_image_from_url(image_url=IMAGE_URL):
    try:
        # Adjust GitHub URLs if necessary
        if "github.com" in image_url:
            image_url = image_url.replace("github.com", "raw.githubusercontent.com").replace("/blob", "")

        response = requests.get(image_url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Convert to image
        img_data = BytesIO(response.content)
        img = Image.open(img_data).convert('RGB')  # Convert to RGB
        img = np.array(img)
        
        # Preprocess the image
        img = cv2.resize(img, (224, 224))  # Resize to 224x224
        img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
        return np.expand_dims(img, axis=0)  # Add batch dimension
    except Exception as e:
        print(f"Failed to preprocess image from URL {image_url}: {e}")
        return None

# Function to extract features from an image URL
def extract_features_from_url(image_url=IMAGE_URL):
    if "github.com" in image_url:
        image_url = image_url.replace("github.com", "raw.githubusercontent.com").replace("/blob", "")
    preprocessed_image = load_and_preprocess_image_from_url(image_url)
    if preprocessed_image is not None:
        return feature_extractor.predict(preprocessed_image)  # Use predict() for numerical input
    else:
        return None

# Build a database of image embeddings from URLs
def build_embedding_database_from_urls(image_urls=IMAGE_URL_LIST):
    """
    Builds a dictionary of image embeddings from a list of image URLs.
    Each entry contains the embedding and the URL of the image.
    
    Parameters:
        image_urls (list): A list of image URLs to extract embeddings from.
    
    Returns:
        dict: A dictionary where the keys are image filenames and the values are dictionaries with 
              'embedding' and 'url' of the images.
    """
    embeddings_db = {}
    for image_url in image_urls:
        if "github.com" in image_url:
            image_url = image_url.replace("github.com", "raw.githubusercontent.com").replace("/blob", "")
        
        embedding = extract_features_from_url(image_url)  # Extract features for each image
        if embedding is not None:
            image_name = image_url.split("/")[-1]  # Use the file name as the key in the database
            embeddings_db[image_name] = {
                "embedding": embedding,
                "url": image_url
            }
    return embeddings_db

# Perform similarity search
def find_similar_images(embeddings_db, query_embedding, top_k=5):
    """
    Finds the top K most similar images to a given query embedding.
    
    Parameters:
        embeddings_db (dict): A dictionary where keys are image names (e.g., '1.jpg')
                              and values are dictionaries containing 'embedding' and 'url'.
        query_embedding (numpy.ndarray): The embedding of the query image as a 2D numpy array.
        top_k (int): Number of most similar images to return.
    
    Returns:
        list: A list of tuples (image_name, similarity_score) sorted by similarity score in descending order.
    """
    similarity_scores = {}

    for image_name, data in embeddings_db.items():
        embedding = data["embedding"]
        # Compute cosine similarity; ensure query_embedding is 2D.
        score = cosine_similarity(query_embedding, embedding)[0][0]
        similarity_scores[image_name] = score

    # Sort by similarity score in descending order
    sorted_images = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_images[:top_k]

# Main code execution
query_embedding = extract_features_from_url(IMAGE_URL)
# Set up image database
embeddings_db = build_embedding_database_from_urls(IMAGE_URL_LIST)
# Query the database with a new image
similar_images = find_similar_images(embeddings_db, query_embedding, top_k=5)

# Display query image
query_image = load_and_preprocess_image_from_url(IMAGE_URL)[0]  # Remove batch dimension
plt.imshow(query_image)
plt.title("Query Image")
plt.axis("off")
plt.show()

# Display actual similar images (use the image URL to fetch and display the image)
for image_name, _ in similar_images:
    # Retrieve the image URL from the embeddings database
    image_url = embeddings_db.get(image_name, {}).get("url", None)  # Get the URL from the database
    
    if image_url:
        # Preprocess the similar image and display it
        similar_image = load_and_preprocess_image_from_url(image_url)[0]  # Preprocess the similar image
        plt.imshow(similar_image)
        plt.title(f"Similar Image: {image_name}")
        plt.axis("off")
        plt.show()
    else:
        print(f"Image URL for {image_name} not found.")
