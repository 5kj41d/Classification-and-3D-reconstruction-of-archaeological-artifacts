import os
import cv2
import numpy as np

def move_low_variance_images_color(folder_path, target_folder, variance_threshold):
    """
    Moves color images with low variance from a specified folder to another folder.

    Parameters:
    - folder_path: Path to the folder containing the images.
    - target_folder: Path to the folder where low variance images will be moved.
    - variance_threshold: Variance threshold below which images are moved.
    """
    # Create the target folder if it does not exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            if image is None:
                continue

            # Convert image to grayscale because variance in grayscale is a good indicator of overall variance
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            variance = np.var(gray_image)
            print(variance, filename)

            if variance < variance_threshold:
                # Move the image to the target folder
                target_image_path = os.path.join(target_folder, filename)
                os.rename(image_path, target_image_path)
                print(f"Moved {filename} to {target_folder} due to low variance: {variance}")

# Usage
folder_path = '/mnt/c/projs/Classification-and-3D-reconstruction-of-archaeological-artifacts/live_generated_images_gan'
target_folder = '/mnt/c/projs/Classification-and-3D-reconstruction-of-archaeological-artifacts/removedImages'
variance_threshold = 1000  # You can adjust this value based on your needs
move_low_variance_images_color(folder_path, target_folder, variance_threshold)
