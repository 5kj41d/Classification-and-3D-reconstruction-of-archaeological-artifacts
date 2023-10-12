'''
Purpose:
Download the images from DIME to external storage for Machine Learning. 
'''
import os
import requests
import csv
import platform
from concurrent.futures import ThreadPoolExecutor

# Check platform
if platform.system() == "Windows":
    external_source_folder = r"E:\Classification-and-3D-reconstruction-of-archaeological-artifacts_DATA\Dime images"
elif platform.system() == "Linux":
    external_source_folder = "/run/media/magnusjsc/T7/Classification-and-3D-reconstruction-of-archaeological-artifacts_DATA/DIME images"

os.makedirs(external_source_folder, exist_ok=True)

# Function to check if an image file already exists
def image_exists(image_filename):
    return os.path.exists(os.path.join(external_source_folder, image_filename))

# Download images
def download_image(image_url, save_path):
    try:
        response = requests.get(image_url)
        if response.status_code == 200:
            with open(save_path, 'wb') as image_file:
                image_file.write(response.content)
                return 1  # Image downloaded successfully
        else:
            print(f"Failed to download image from {image_url}. Status code: {response.status_code}")
            return -1  # Image download failed
    except Exception as e:
        print(f"Error downloading image from {image_url}: {e}")
        return -1  # Image download failed

# Process each datarow
def process_row(row):
    image_url = row['URL']
    image_filename = os.path.basename(image_url)
    save_path_original_images = os.path.join(external_source_folder, image_filename)
    
    # Check if the image file already exists
    if os.path.exists(save_path_original_images):
        return 0  # Image already downloaded
    
    # Download and save the image
    return download_image(image_url, save_path_original_images)

# Resume downloads
def resume_downloads():
    with open(csv_file_path, mode='r', newline='') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=';')
        number_of_images_downloaded = 0

        # Map functions make sure each row is only processed once.
        # Concurrent process for faster execution.
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(process_row, csv_reader))

        # Count successful downloads (where the return value is 1)
        number_of_images_downloaded = results.count(1)

    print(f"Total images downloaded: {number_of_images_downloaded}")

# Call the function to resume downloads
resume_downloads()