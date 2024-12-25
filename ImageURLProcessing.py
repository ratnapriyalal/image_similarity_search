import requests
import os

# Define constants for GitHub repository and folder URL
IMAGE_DATABASE_URL = 'https://github.com/karpathy/neuraltalk/tree/master/'  # GitHub repository URL
FOLDER_PATH = 'example_images/'  # Folder path in the repository

# Define a default list of image URLs (COCO dataset sample images)
DEFAULT_IMAGE_LIST = [
    f"https://github.com/cocodataset/cocodataset.github.io/blob/master/images/cocoicons/{i}.jpg"
    for i in range(1, 91)
]

# Define the list of supported image file extensions
image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']

def is_valid_image(url):
    """
    Check if the URL points to a valid image file.

    Args:
        url (str): URL to check.

    Returns:
        bool: True if the URL points to a valid image, otherwise False.
    """
    # Check if URL ends with an image extension
    if not any(url.lower().endswith(ext) for ext in image_extensions):
        return False

    try:
        # Send a GET request to the URL
        response = requests.get(url, stream=True, timeout=5)
        
        # Check if the response status code indicates a valid image file (HTTP 200)
        if response.status_code == 200:
            # Verify that the content type is an image
            content_type = response.headers.get('Content-Type')
            if content_type and 'image' in content_type:
                return True
    except requests.exceptions.RequestException as e:
        # Handle connection errors, timeout, or invalid URL errors
        print(f"Error fetching {url}: {e}")
    return False

def get_github_folder_contents(repo_url=IMAGE_DATABASE_URL, folder_path=FOLDER_PATH):
    """
    Fetch contents of a GitHub repository folder using the GitHub API.

    Args:
        repo_url (str): GitHub repository URL (e.g., 'https://api.github.com/repos/username/repo').
        folder_path (str): Path of the folder inside the repository (e.g., 'folder/').

    Returns:
        list: List of valid image URLs from the folder.
    """
    # Construct the URL to access folder contents
    folder_api_url = f"{repo_url}{folder_path}"

    try:
        # Send a GET request to fetch folder contents
        response = requests.get(folder_api_url)
        
        # Check if the response status code is OK (200)
        if response.status_code == 200:
            # Parse the JSON response
            folder_contents = response.json()
            image_urls = []

            # Loop through folder contents to filter image files
            for item in folder_contents:
                if item['type'] == 'file':  # Ensure the item is a file
                    file_url = item['download_url']  # Get the raw file URL
                    if is_valid_image(file_url):
                        image_urls.append(file_url)

            return image_urls
        else:
            print(f"Error fetching folder contents: {response.status_code}")
            return []

    except requests.exceptions.RequestException as e:
        # Handle exceptions during the API request
        print(f"Error fetching GitHub folder: {e}")
        return []

# Fetch valid image URLs from the specified folder
image_links = get_github_folder_contents(IMAGE_DATABASE_URL, FOLDER_PATH)

if image_links:
    image_url_list = image_links.copy()  # Use the fetched image URLs
    print("Valid image URLs found:")
    for img_url in image_links:
        print(img_url)
else:
    # Fallback to default image list if no valid images are found
    print("No valid images found. Setting list to the default value.")
    image_url_list = DEFAULT_IMAGE_LIST.copy()
