import os
import shutil

# Paths
source_folder = r"/mnt/c/Users/jon/Pictures/downloaded"
destination_folder = r"/mnt/c/Users/jon/Pictures/rawiron"
txt_file_path = r"/mnt/c/projs/Classification-and-3D-reconstruction-of-archaeological-artifacts/dime_data/iron.txt"

# Create the destination folder if it doesn't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Read the filenames from the txt file
with open(txt_file_path, 'r') as file:
    filenames = [line.strip() for line in file.readlines()]

# Move the files
for filename in filenames:
    source_path = os.path.join(source_folder, filename)
    if os.path.isfile(source_path):
        shutil.move(source_path, destination_folder)
        print(f"Moved: {filename}")
    else:
        print(f"File not found: {filename}")
