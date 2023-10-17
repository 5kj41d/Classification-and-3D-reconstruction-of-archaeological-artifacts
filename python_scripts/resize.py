import os
from PIL import Image

def resize_images(input_folder, output_folder, target_size=(256, 256)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            with Image.open(input_path) as img:
                img_resized = img.resize(target_size)
                img_resized.save(output_path)

if __name__ == "__main__":
    input_folder = ""
    output_folder = ""

    resize_images(input_folder, output_folder, target_size=(256, 256))
