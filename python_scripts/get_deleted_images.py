import os
import shutil
import hashlib
from multiprocessing import Pool, cpu_count

def calculate_hash(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return file_path, hash_md5.hexdigest()

def get_image_files(folder):
    image_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                image_files.append(os.path.join(root, file))
    return image_files

def get_image_hashes(image_files):
    with Pool(cpu_count()) as pool:
        image_hashes = pool.map(calculate_hash, image_files)
    return dict(image_hashes)

def copy_unique_images(large_folder, small_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    print("Reading small folder...")
    small_folder_files = get_image_files(small_folder)
    print("Hashing small folder images...")
    small_folder_hashes = get_image_hashes(small_folder_files)

    print("Reading large folder...")
    large_folder_files = get_image_files(large_folder)
    print("Hashing large folder images...")
    large_folder_hashes = get_image_hashes(large_folder_files)

    for file_path, image_hash in large_folder_hashes.items():
        if image_hash not in small_folder_hashes.values():
            shutil.copy(file_path, output_folder)
            print(f'Copied: {file_path} to {output_folder}')

large_folder = '/mnt/c/Users/jon/Pictures/destination'
small_folder = '/mnt/c/Users/jon/Pictures/sorted'
output_folder = '/mnt/c/Users/jon/Pictures/deleted'

copy_unique_images(large_folder, small_folder, output_folder)
