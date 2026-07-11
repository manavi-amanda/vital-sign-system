import requests
import os

# ==========================================
# Face++ API Credentials
# ==========================================
# Replace these with your actual Face++ keys.
# Alternatively, you can use os.getenv() if you prefer environment variables.
API_KEY = "api_key"
API_SECRET = "api_secret"


def detect_age(image_path, default_age=24):
    """
    Sends an image to the Face++ API to estimate the person's age.

    Args:
        image_path (str): The local path to the image file.
        default_age (int): The age to return if detection fails or API limits are hit.

    Returns:
        int: The estimated age, or the default age if unsuccessful.
    """
    url = "https://api-us.faceplusplus.com/facepp/v3/detect"

    # Parameters for the API request
    data = {
        'api_key': API_KEY,
        'api_secret': API_SECRET,
        'return_attributes': 'age'
    }

    # Check if the file actually exists before sending
    if not os.path.exists(image_path):
        print(f"[Age Detector] Error: File {image_path} not found.")
        return default_age

    try:
        # Open the image file in binary read mode
        with open(image_path, 'rb') as image_file:
            files = {'image_file': image_file}

            # Send request to Face++
            response = requests.post(url, data=data, files=files, timeout=10)
            result = response.json()

            # 1. Handle API-level errors (e.g., concurrency limit, bad keys)
            if 'error_message' in result:
                print(f"[Age Detector] Face++ API Error: {result['error_message']}")
                return default_age

            # 2. Extract age if a face is found
            if 'faces' in result and len(result['faces']) > 0:
                # We grab the age of the first face detected
                age = result['faces'][0]['attributes']['age']['value']
                return age
            else:
                # 3. Handle cases where the image is sent, but no face is visible
                print("[Age Detector] No faces detected in the frame.")
                return default_age

    except requests.exceptions.RequestException as e:
        print(f"[Age Detector] Network error connecting to Face++: {e}")
        return default_age
    except Exception as e:
        print(f"[Age Detector] Unexpected error: {e}")
        return default_age