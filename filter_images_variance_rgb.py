import os
import cv2
import numpy as np

def delete_low_variance_images_color(folder_path, variance_threshold):
    """
    Deletes color images with low variance from a specified folder.

    Parameters:
    - folder_path: Path to the folder containing the images.
    - variance_threshold: Variance threshold below which images are deleted.
    """
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            if image is None:
                continue

            # Calculate variance for each color channel
            variances = [np.var(image[:, :, i]) for i in range(image.shape[2])]  # image.shape[2] should be 3 (RGB)
            total_variance = sum(variances)  # Summing up the variances of R, G, B channels
            print(total_variance, filename)

            if total_variance < variance_threshold:
                target_image_path = os.path.join(target_folder, filename)
                os.rename(image_path, target_image_path)
                print(f"Moved {filename} to {target_folder} due to low variance: {total_variance}")

# Usage
folder_path = '/mnt/c/projs/Classification-and-3D-reconstruction-of-archaeological-artifacts/live_generated_images_gan'
target_folder = '/mnt/c/projs/Classification-and-3D-reconstruction-of-archaeological-artifacts/removedImages'
variance_threshold = 30  # You can adjust this value based on your needs
delete_low_variance_images_color(folder_path, variance_threshold)
