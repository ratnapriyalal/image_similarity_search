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
import imagehash

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
def build_hash_database(image_urls):
    """
    Build a database of image hashes from URLs.

    Parameters:
        image_urls (list): List of image URLs.

    Returns:
        dict: A dictionary where keys are image names and values are their hashes.
    """
    hash_db = {}
    for image_url in image_urls:
        try:
            if "github.com" in image_url:
                image_url = image_url.replace("github.com", "raw.githubusercontent.com").replace("/blob", "")
            response = requests.get(image_url)
            response.raise_for_status()  # Raise an error for bad status codes
            img_data = BytesIO(response.content)
            image = Image.open(img_data).convert("RGB")
            hash_db[image_url.split("/")[-1]] = imagehash.average_hash(image)
        except Exception as e:
            print(f"Failed to process image {image_url}: {e}")
    return hash_db

def find_similar_images_hash(embeddings_db, image_url=IMAGE_URL, top_k=5):
    """
    Perform similarity search using image hashing.

    Parameters:
        embeddings_db (dict): A dictionary where keys are image names (e.g., '1.jpg')
                              and values are precomputed image hashes.
        image_url (str): URL of the query image.
        top_k (int): Number of most similar images to return.

    Returns:
        list: A list of tuples (image_name, hash_difference) sorted by hash difference.
    """
    try:
        # Transform GitHub URL to raw URL if necessary
        if "github.com" in image_url:
            image_url = image_url.replace("github.com", "raw.githubusercontent.com").replace("/blob", "")

        # Download the image from the URL
        response = requests.get(image_url)
        response.raise_for_status()  # Raise an error for bad status codes
        img_data = BytesIO(response.content)
        query_image = Image.open(img_data).convert("RGB")

        # Compute hash for the query image
        query_hash = imagehash.average_hash(query_image)

        # Compute hash differences with images in the database
        hash_differences = []
        for image_name, image_hash in embeddings_db.items():
            hash_difference = query_hash - image_hash  # Compute hash difference
            hash_differences.append((image_name, hash_difference))

        # Sort by hash difference (lower is better)
        hash_differences.sort(key=lambda x: x[1])
        return hash_differences[:top_k]

    except Exception as e:
        print(f"Error processing image from URL {image_url}: {e}")
        return []

query_embedding = extract_features_from_url(IMAGE_URL)
# Set up image database
hash_embeddings_db = build_hash_database(IMAGE_URL_LIST)
# Query the database with a new image
similar_images = find_similar_images_hash(hash_embeddings_db, IMAGE_URL, top_k=5)

# Display results
print("Most similar images:")
for image_name, hash_diff in similar_images:
    print(f"{image_name}: Hash Difference = {hash_diff}")
    similar_image = hash_embeddings_db.get(image_name)

# Display query image
query_image = load_and_preprocess_image_from_url(IMAGE_URL)[0]  # Remove batch dimension
plt.imshow(query_image)
plt.title("Query Image")
plt.axis("off")
plt.show()

for image_name, hash_diff in similar_images:
    # Retrieve the image URL
    image_url = [url for url in IMAGE_URL_LIST if image_name in url]
    if image_url:
        # Adjust GitHub URLs for direct download
        image_url[0] = image_url[0].replace("github.com", "raw.githubusercontent.com").replace("/blob", "")
        
        try:
            # Fetch the image
            response = requests.get(image_url[0])
            response.raise_for_status()
            
            # Validate content type
            if "image" not in response.headers["Content-Type"]:
                print(f"URL {image_url[0]} did not return an image. Content-Type: {response.headers['Content-Type']}")
                continue
            
            # Load and display the image
            img_data = BytesIO(response.content)
            original_image = Image.open(img_data).convert("RGB")
            plt.imshow(original_image)
            plt.title(f"Similar Image: {image_name} (Hash Diff: {hash_diff})")
            plt.axis("off")
            plt.show()
        except Exception as e:
            print(f"Failed to process image {image_name} from URL {image_url[0]}: {e}")
    else:
        print(f"Image {image_name} not found in the URL list.")
