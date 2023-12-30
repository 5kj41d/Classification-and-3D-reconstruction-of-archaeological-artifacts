from PIL import Image
import os

def resize_images(input_folder, output_folder, size=(224, 224)):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        print('Output folder do not exist. Creates a new one and copying to that :)')
        os.makedirs(output_folder)
    else: 
        print('The output folder alreadt exists. Starts copying the images to that :)')

    # Get a list of all files in the input folder
    files = os.listdir(input_folder)

    for file in files:
        try:
            # Open the image file
            img_path = os.path.join(input_folder, file)
            img = Image.open(img_path)

            # Resize the image
            img_resized = img.resize(size)

            # Save the resized image to the output folder
            output_path = os.path.join(output_folder, file)
            img_resized.save(output_path)

            print(f"Resized {file} successfully.")
        except Exception as e:
            print(f"Error resizing {file}: {e}")

# Replace 'input_folder_path' and 'output_folder_path' with your folder paths
input_folder_path = '/run/media/magnusjsc/T7/Classification-and-3D-reconstruction-of-archaeological-artifacts_DATA/resized_images_others_256x256/'
output_folder_path = '/run/media/magnusjsc/T7/Classification-and-3D-reconstruction-of-archaeological-artifacts_DATA/other-2023 224x224/other/'

resize_images(input_folder_path, output_folder_path)