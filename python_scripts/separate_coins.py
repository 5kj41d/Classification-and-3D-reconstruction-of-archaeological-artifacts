import os
import shutil
import pandas as pd

# Define the paths
csv_file_path = '/mnt/c/projs/Classification-and-3D-reconstruction-of-archaeological-artifacts/dime_data/DIME billeder.csv'
images_folder_path = '/mnt/c/Users/jon/Pictures/downloaded'
output_folder_path = '/mnt/c/Users/jon/Pictures/destination'

# Load the CSV file with error handling
try:
    df = pd.read_csv(csv_file_path, delimiter=';', quotechar='"', on_bad_lines='skip')
except pd.errors.ParserError as e:
    print(f"Error parsing CSV file: {e}")
    exit()

# Ensure the output directory exists
os.makedirs(output_folder_path, exist_ok=True)

# Iterate over the images in the folder
for image_filename in os.listdir(images_folder_path):
    # Check if the image is listed in the CSV file
    matching_rows = df[df['filnavn'] == image_filename]
    if not matching_rows.empty:
        # Get the "thesaurus" value
        thesaurus_value = matching_rows['thesaurus'].values[0]
        # Check if it contains "dime.find.coin"
        if 'dime.find.coin' in thesaurus_value:
            # Copy the image to the output folder
            shutil.copy2(os.path.join(images_folder_path, image_filename), os.path.join(output_folder_path, image_filename))

print("Images copied successfully.")