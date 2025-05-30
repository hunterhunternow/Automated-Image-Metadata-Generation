import requests
import os
import io
import base64
import csv
from PIL import Image
from google.cloud import vision
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# --- Configuration Constants ---
OUTPUT_CSV_FILENAME = "image_metadata.csv"
MAX_IMAGE_WIDTH = 1024
COMPRESSION_QUALITY = 85
ASTICA_API_TIMEOUT = 30 # seconds

# --- Helper to check if running in Colab ---
def is_running_in_colab():
    """Checks if the script is running in a Google Colab environment."""
    try:
        import google.colab
        return True
    except ImportError:
        return False

# --- Authentication Setup ---
def setup_google_credentials():
    """
    Sets up Google Cloud credentials.
    In Colab, prompts for upload if GOOGLE_APPLICATION_CREDENTIALS is not set.
    Otherwise, expects GOOGLE_APPLICATION_CREDENTIALS environment variable to be set.
    Returns True if credentials are set or appear to be set, False otherwise.
    """
    if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ and os.path.exists(os.environ["GOOGLE_APPLICATION_CREDENTIALS"]):
        print(f"Using Google credentials from GOOGLE_APPLICATION_CREDENTIALS: {os.environ['GOOGLE_APPLICATION_CREDENTIALS']}")
        return True

    if is_running_in_colab():
        print("Running in Google Colab. GOOGLE_APPLICATION_CREDENTIALS not found or invalid.")
        print("Please upload your service account key JSON file.")
        try:
            # Ensure google.colab.files is available
            from google.colab import files
            uploaded = files.upload()
            if not uploaded:
                print("No file uploaded for Google credentials. Google Cloud Vision API calls may fail.")
                return False
            for fn in uploaded.keys():
                # The uploaded file is now in the Colab environment's current directory
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = fn
                print(f"Google credentials set from uploaded file: {fn}")
                return True # Assuming one key file
        except Exception as e:
            print(f"Error during Colab file upload for credentials: {e}")
            return False
    else: # Not in Colab and env var not set or invalid
        print("Error: GOOGLE_APPLICATION_CREDENTIALS environment variable not set or path is invalid.")
        print("Please set it to the path of your service account key JSON file (e.g., in your .env file or system environment).")
        return False
    return False # Default case

# --- Image Processing Functions ---
def compress_image(image_path, max_width=MAX_IMAGE_WIDTH, quality=COMPRESSION_QUALITY):
    """Compresses and resizes an image in-place."""
    try:
        img = Image.open(image_path)
        original_size = os.path.getsize(image_path)

        if img.width > max_width:
            width_percent = (max_width / float(img.width))
            height_size = int((float(img.height) * float(width_percent)))
            img = img.resize((max_width, height_size), Image.LANCZOS) # LANCZOS is good for downscaling

        img.save(image_path, quality=quality, optimize=True)
        compressed_size = os.path.getsize(image_path)
        print(f"Compressed {os.path.basename(image_path)}: {original_size} bytes -> {compressed_size} bytes")
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path} for compression.")
    except Exception as e:
        print(f"Error compressing image {os.path.basename(image_path)}: {e}")

def get_image_tags_from_google(image_path):
    """Detects labels from Google Cloud Vision."""
    try:
        client = vision.ImageAnnotatorClient() # Initialize client after credentials are confirmed
        with io.open(image_path, 'rb') as image_file:
            content = image_file.read()
        image_vision = vision.Image(content=content) # Renamed to avoid conflict with PIL.Image
        response = client.label_detection(image=image_vision)

        if response.error.message:
            raise Exception(f"Google Vision API error: {response.error.message}")

        labels = response.label_annotations
        google_tags = ", ".join([label.description for label in labels])
        return google_tags if google_tags else "No tags found by Google Vision"
    except Exception as e:
        print(f"Error getting tags from Google Vision for {os.path.basename(image_path)}: {e}")
        return "Google Vision tags failed"

def convert_image_to_base64(image_path):
    """Convert a local image to a base64 string."""
    try:
        with open(image_path, 'rb') as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        return base64_image
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path} for base64 conversion.")
        return None
    except Exception as e:
        print(f"Error converting image {os.path.basename(image_path)} to base64: {e}")
        return None

def get_description_from_astica(image_base64):
    """Uses Astica AI to get a description for a base64-encoded image."""
    astica_api_key = os.environ.get('ASTICA_API_KEY')
    if not astica_api_key:
        print("Error: ASTICA_API_KEY environment variable not set.")
        return "Astica API key not configured"

    api_endpoint = 'https://vision.astica.ai/describe'
    params = {
        'tkn': astica_api_key,
        'modelVersion': '1.0_full', # Consider making this configurable
        'input': image_base64,
        'visionParams': 'describe'
    }

    try:
        response_astica = requests.post(api_endpoint, json=params, timeout=ASTICA_API_TIMEOUT)
        response_astica.raise_for_status() # Raises HTTPError for bad responses (4XX or 5XX)
        astica_result = response_astica.json()

        if astica_result.get('status') == 'error' or 'error' in astica_result:
            error_msg = astica_result.get('error', 'Unknown Astica API error')
            print(f"Astica API returned an error: {error_msg}")
            return f"Astica API error: {error_msg}"

        astica_description = astica_result.get('caption', '').strip()
        return astica_description if astica_description else "No description available from Astica"

    except requests.exceptions.HTTPError as http_err:
        print(f"Astica API HTTP error: {http_err} - Response: {response_astica.text if 'response_astica' in locals() else 'No response object'}")
        return "Astica API request failed (HTTP error)"
    except requests.exceptions.RequestException as req_err: # Catches DNS errors, connection timeouts, etc.
        print(f"Astica API request error: {req_err}")
        return "Astica API request failed (Connection/Request error)"
    except ValueError as json_err: # Catches JSONDecodeError
        print(f"Error decoding Astica API JSON response: {json_err}")
        return "Astica API response JSON decoding failed"
    except Exception as e:
        print(f"An unexpected error occurred with Astica API: {e}")
        return "Astica description processing failed (Unexpected error)"

# --- Main Processing Logic ---
def process_image_metadata(image_path):
    """Generates metadata for a single image."""
    filename = os.path.basename(image_path)
    print(f"\nProcessing image: {filename}...")

    compress_image(image_path) # Compresses in-place

    google_tags = get_image_tags_from_google(image_path)

    astica_description = "Astica processing skipped (image not available for base64)"
    image_base64 = convert_image_to_base64(image_path)
    if image_base64:
        astica_description = get_description_from_astica(image_base64)
    else:
        print(f"Skipping Astica for {filename} due to base64 conversion failure.")


    metadata = {
        'filename': filename,
        'description': astica_description,
        'tags': google_tags
    }
    return metadata

def main():
    """
    Main function to handle image inputs and metadata processing.
    """
    print("Starting Image Metadata Generation Script...")

    if not setup_google_credentials():
        print("Google Cloud credentials setup failed. Exiting, as Google Vision API is essential.")
        return

    if not os.environ.get('ASTICA_API_KEY'):
        print("Warning: ASTICA_API_KEY environment variable not set. Descriptions from Astica will not be generated.")
        # Script can continue, but Astica part will yield errors/skipped messages.

    processed_images_metadata = []
    temp_image_paths_colab = [] # To store paths of files temporarily saved in Colab

    if is_running_in_colab():
        from google.colab import files
        print("\nCOLAB MODE: Please upload images for processing.")
        uploaded_images_content = files.upload() # Dict of {filename: content}
        if not uploaded_images_content:
            print("No images uploaded in Colab. Exiting.")
            return

        current_dir = os.getcwd()
        for name, content in uploaded_images_content.items():
            temp_path = os.path.join(current_dir, name)
            try:
                with open(temp_path, 'wb') as f:
                    f.write(content)
                temp_image_paths_colab.append(temp_path)
                print(f"Temporarily saved uploaded file: {name}")
            except Exception as e:
                print(f"Error saving uploaded Colab file {name} temporarily: {e}")
        
        image_paths_to_process = temp_image_paths_colab

    else: # LOCAL MODE
        image_source_path = input("Enter the path to a directory containing images OR a single image file: ").strip()
        if not os.path.exists(image_source_path):
            print(f"Error: Path '{image_source_path}' does not exist.")
            return
        
        image_paths_to_process = []
        if os.path.isdir(image_source_path):
            print(f"Scanning directory: {image_source_path}")
            for f_name in os.listdir(image_source_path):
                if f_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                    image_paths_to_process.append(os.path.join(image_source_path, f_name))
        elif os.path.isfile(image_source_path):
            if image_source_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                image_paths_to_process.append(image_source_path)
            else:
                print(f"Error: '{image_source_path}' is not a recognized image file type.")
                return
        else:
            print(f"Error: Path '{image_source_path}' is not a valid file or directory.")
            return
        
        if not image_paths_to_process:
            print(f"No compatible image files found at '{image_source_path}'.")
            return

    # Process all identified images
    for image_path in image_paths_to_process:
        metadata = process_image_metadata(image_path)
        processed_images_metadata.append(metadata)
        print(f"Finished processing {metadata['filename']}.")
        print(f"  Description: {metadata['description']}")
        print(f"  Tags: {metadata['tags']}")

    # Write to CSV
    if processed_images_metadata:
        try:
            with open(OUTPUT_CSV_FILENAME, "w", newline='', encoding='utf-8') as csvfile:
                fieldnames = ["Filename", "Description", "Tags"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(processed_images_metadata) # Use writerows for list of dicts
            print(f"\nMetadata successfully saved to {OUTPUT_CSV_FILENAME}")

            if is_running_in_colab():
                from google.colab import files
                print("Attempting to download CSV in Colab...")
                files.download(OUTPUT_CSV_FILENAME)
        except IOError as e:
            print(f"Error writing CSV file '{OUTPUT_CSV_FILENAME}': {e}")
        except Exception as e:
            print(f"An unexpected error occurred during CSV export: {e}")
    else:
        print("\nNo images were processed, so no metadata CSV was generated.")

    # Clean up temporary image files created from Colab uploads
    if is_running_in_colab() and temp_image_paths_colab:
        print("\nCleaning up temporary files from Colab session...")
        for temp_file_path in temp_image_paths_colab:
            if os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                    print(f"Cleaned up: {os.path.basename(temp_file_path)}")
                except Exception as e:
                    print(f"Error cleaning up temporary file {os.path.basename(temp_file_path)}: {e}")
    
    print("\nScript finished.")

if __name__ == "__main__":
    main()
