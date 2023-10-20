import os
from PIL import Image

# Use below command in the folder of interest to print the number of elements. Bash. 
# ls -1 | wc -l 

def resize_images(input_folder, output_folder, target_size=(256, 256), start_index=0):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = os.listdir(input_folder)
    unidentified_images = []

    for i in range(start_index, len(files)):
        filename = files[i]

        if filename.endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            try:
                with Image.open(input_path) as img:
                    img_resized = img.resize(target_size)
                    img_resized.save(output_path)
            except Exception as e:
                print(f"Error processing {input_path}: {str(e)}")
                unidentified_images.append(filename) 
    
    with open("unidentified_images.txt", "w") as logfile:
        for filename in unidentified_images:
            logfile.write(filename + "\n")

if __name__ == "__main__":
    input_folder = "/run/media/magnusjsc/T7/Classification-and-3D-reconstruction-of-archaeological-artifacts_DATA/others/"
    output_folder = "/run/media/magnusjsc/T7/Classification-and-3D-reconstruction-of-archaeological-artifacts_DATA/resized_images_others_256x256/"

    start_index = 135392

    resize_images(input_folder, output_folder, target_size=(256, 256), start_index=start_index)

# Coins:
# INPUT - /run/media/magnusjsc/T7/Classification-and-3D-reconstruction-of-archaeological-artifacts_DATA/coin/
# OUTPUT - /run/media/magnusjsc/T7/Classification-and-3D-reconstruction-of-archaeological-artifacts_DATA/resized_images_coin_256x256/

# Others: 
# INPUT - /run/media/magnusjsc/T7/Classification-and-3D-reconstruction-of-archaeological-artifacts_DATA/others/
# OUTPUT - /run/media/magnusjsc/T7/Classification-and-3D-reconstruction-of-archaeological-artifacts_DATA/resized_images_others_256x256/