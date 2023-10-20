import os 
from PIL import Image 
import csv 

# Get number of elements in a folder using Bash.
# ls -1 | wc -l

def process_images_and_save_to_csv(input_folder, csv_filename):
    image_data = {} 
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg')):
            image_path = os.path.join(input_folder, filename)
            try: 
                with Image.open(image_path) as img: 
                    width, height = img.size 
                    key = (width, height)
                    if key in image_data:
                        # Increment occurence 
                        image_data[key] += 1
                    else: 
                        # Add new occurence entry
                        image_data[key] = 1
            except Exception as e: 
                print(f"Error processing {filename}: {str(e)}")

    # Save image data/dimensions to a CSV file as last step
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write headers to the CSV file
        csv_writer.writerow(['Width', 'Height', 'Occurrences'])
        # Write the image data
        for (width, height), occurrences in image_data.items():
            csv_writer.writerow([width, height, occurrences])

if __name__ == "__main__":
    input_folder = "/run/media/magnusjsc/T7/Classification-and-3D-reconstruction-of-archaeological-artifacts_DATA/DIME images/"
    csv_filename = "processed_images_dimensions.csv"

    process_images_and_save_to_csv(input_folder, csv_filename)

# INPUT - /run/media/magnusjsc/T7/Classification-and-3D-reconstruction-of-archaeological-artifacts_DATA/DIME images/
