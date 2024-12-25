import cv2
import numpy as np
import os
import requests
from io import BytesIO
from PIL import Image

import tensorflow as tf
import tensorflow_hub as hub
import tf_keras
import cv2
import matplotlib.pyplot as plt
import os
from sklearn.neighbors import NearestNeighbors

# Set environment variable for legacy Keras compatibility
os.environ['TF_USE_LEGACY_KERAS'] = '1'

# Define model URL
TF_MODEL_URL = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'  # Feature extraction model
IMAGE_URL_LIST = image_url_list.copy()
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
    embeddings_db = {}
    for image_url in image_urls:
        if "github.com" in image_url:
            image_url = image_url.replace("github.com", "raw.githubusercontent.com").replace("/blob", "")
        embedding = extract_features_from_url(image_url)
        if embedding is not None:
            image_name = image_url.split("/")[-1]  # Use the file name as the key
            embeddings_db[image_name] = embedding
    return embeddings_db

def find_similar_images_knn(embeddings_db, query_embedding, top_k=5):
    """
    Perform KNN search to find the top K similar images.

    Parameters:
        embeddings_db (dict): A dictionary where keys are image names (e.g., '1.jpg')
                              and values are embeddings stored as 2D numpy arrays.
        query_embedding (numpy.ndarray): The embedding of the query image.
        top_k (int): Number of most similar images to return.

    Returns:
        list: A list of tuples (image_name, distance) sorted by distance.
    """
    # Convert the embeddings database to a 2D numpy array
    image_names = list(embeddings_db.keys())
    embeddings = np.vstack([embedding.flatten() for embedding in embeddings_db.values()])  # Flatten in case of extra dimensions

    # Flatten the query embedding if necessary
    query_embedding = query_embedding.flatten()

    # Fit KNN
    knn = NearestNeighbors(n_neighbors=top_k, metric='euclidean')
    knn.fit(embeddings)

    # Perform the search
    distances, indices = knn.kneighbors([query_embedding])

    # Retrieve the image names and distances for the top-k results
    similar_images = [(image_names[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
    return similar_images


query_embedding = extract_features_from_url(IMAGE_URL)
# Set up image database
embeddings_db = build_embedding_database_from_urls(IMAGE_URL_LIST)
# Query the database with a new image
similar_images = find_similar_images_knn(embeddings_db, query_embedding, top_k=5)

# Display results
print("Most similar images:")
for image_name, dist in similar_images:
    print(f"{image_name}: KNN Distance = {dist}")
    similar_image = embeddings_db.get(image_name)

# Display query image
query_image = load_and_preprocess_image_from_url(IMAGE_URL)[0]  # Remove batch dimension
plt.imshow(query_image)
plt.title("Query Image")
plt.axis("off")
plt.show()

# Display similar images
for image_name, _ in similar_images:
    print("KNN Distance", _)
    
    # Retrieve the image from the database (using the image URL or original image)
    similar_image_url = [url for url in IMAGE_URL_LIST if image_name in url][0]
    similar_image = load_and_preprocess_image_from_url(similar_image_url)[0]  # Remove batch dimension
    
    if similar_image is not None:
        # Display the similar image
        plt.imshow(similar_image)
        plt.title(f"Similar Image: {image_name}")
        plt.axis("off")
        plt.show()
    else:
        print(f"Image {image_name} not found in the image dictionary.")
