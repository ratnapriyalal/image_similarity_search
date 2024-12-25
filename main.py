from ImageURLProcessing import fetch_image_urls
from CosineSimilarityImageSearch import build_embedding_database_from_urls, extract_features_from_url, find_similar_images, load_and_preprocess_image_from_url
import matplotlib.pyplot as plt

def main():
    # Step 1: Fetch the image URLs using ImageURLProcessing
    image_url_list = fetch_image_urls()

    # Ensure that the list is not empty
    if not image_url_list:
        print("No image URLs were found. Exiting.")
        return

    # Step 2: Choose a query image (we'll use the first image in the list)
    IMAGE_URL = image_url_list[0]
    
    # Step 3: Extract features from the query image
    query_embedding = extract_features_from_url(IMAGE_URL)
    
    # Ensure the query embedding is valid
    if query_embedding is None:
        print("Failed to extract features from the query image. Exiting.")
        return

    # Step 4: Build the embedding database from the image URLs
    embeddings_db = build_embedding_database_from_urls(image_url_list)

    # Step 5: Find the top 5 most similar images based on cosine similarity
    similar_images = find_similar_images(embeddings_db, query_embedding, top_k=5)

    # Step 6: Display the query image
    query_image = load_and_preprocess_image_from_url(IMAGE_URL)[0]
    plt.imshow(query_image)
    plt.title("Query Image")
    plt.axis("off")
    plt.show()

    # Step 7: Display the most similar images
    for image_name, _ in similar_images:
        image_url = embeddings_db.get(image_name, {}).get("url", None)
        if image_url:
            similar_image = load_and_preprocess_image_from_url(image_url)[0]
            plt.imshow(similar_image)
            plt.title(f"Similar Image: {image_name}")
            plt.axis("off")
            plt.show()
        else:
            print(f"Image URL for {image_name} not found.")

if __name__ == "__main__":
    main()
