from google.colab import files
import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import cv2
import matplotlib.pyplot as plt
import pandas as pd

# Step 1: Upload an image file
# The following code allows the user to upload an image file directly in Google Colab
uploaded = files.upload()  # Prompts the user to upload an image file

# Step 2: Define model and label map URLs
# TF_MODEL_URL points to the TensorFlow Hub model for classification
# LABEL_MAP_URL is a CSV file that maps label IDs to human-readable names
TF_MODEL_URL = 'https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_asia_V1/1'
LABEL_MAP_URL = 'https://www.gstatic.com/aihub/tfhub/labelmaps/landmarks_classifier_asia_V1_label_map.csv'

# Step 3: Load the label map
# The label map is read from the CSV file and stored in a dictionary for easy lookup
df = pd.read_csv(LABEL_MAP_URL)
label_map = dict(zip(df.id, df.name))  # Maps label IDs to label names

# Step 4: Load the TensorFlow Hub model
# Use TensorFlow's Functional API to define the input and connect it to the pre-trained model
inputs = tf.keras.Input(shape=(321, 321, 3), name='input_image')  # Input tensor with the correct shape
classifier_layer = hub.KerasLayer(TF_MODEL_URL, trainable=False, name="classifier_layer", output_key='predictions:logits')
outputs = classifier_layer(inputs)  # Connect the input tensor to the classifier layer

# Create a Keras model that uses the classifier layer
classifier = tf.keras.Model(inputs=inputs, outputs=outputs)

# Step 5: Define a function to preprocess the image
# This function reads the image, converts it to the required format, and normalizes it
def load_and_preprocess_image(image_path):
    """
    Load and preprocess an image for the classifier.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray: Preprocessed image ready for model input.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or unable to read.")
    RGBimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    RGBimg = cv2.resize(RGBimg, (321, 321))  # Resize to match model input size
    RGBimg = np.array(RGBimg) / 255.0  # Normalize pixel values to [0, 1]
    return np.reshape(RGBimg, (1, 321, 321, 3))  # Add batch dimension

# Step 6: Preprocess the uploaded image
# Retrieve the uploaded file's path and preprocess it for prediction
image_path = list(uploaded.keys())[0]  # Use the uploaded file's name
RGBimg = load_and_preprocess_image(image_path)

# Step 7: Make a prediction using the classifier
# The classifier outputs logits, which are used to determine the predicted label
prediction = classifier.predict(RGBimg)
predicted_label_id = np.argmax(prediction)  # Get the index of the highest probability
predicted_label = label_map[predicted_label_id]  # Map the index to a label name

# Step 8: Display the prediction and the image
# Visualize the image with the predicted label as the title
print(f"Prediction ID: {predicted_label_id}, Prediction Label: {predicted_label}")
plt.imshow(RGBimg[0])  # Display the image (remove batch dimension)
plt.title(f'Predicted: {predicted_label}')
plt.axis('off')
plt.show()
