import os
import sys
from PIL import Image
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import platform
import threading

# Define paths for both Windows and Linux
external_hard_disk_path_coin = ""
external_hard_disk_path_others = ""
# Sorted by dimension folders - Post
external_destination_source_folder_good_coin = ""
external_destination_source_folder_bad_coin = "" 
external_destination_source_folder_good_others = ""
external_destination_source_folder_bad_others = "" 

# Check platform
if platform.system() == "Windows":
    # Pre-
    external_hard_disk_path_coin = r"E:\Classification-and-3D-reconstruction-of-archaeological-artifacts_DATA\coin"
    external_hard_disk_path_others = r"E:\Classification-and-3D-reconstruction-of-archaeological-artifacts_DATA\others"
    external_source_folder = r"E:\Classification-and-3D-reconstruction-of-archaeological-artifacts_DATA"
elif platform.system() == "Linux":
    # Pre-
    external_hard_disk_path_coin = "/run/media/magnusjsc/T7/Classification-and-3D-reconstruction-of-archaeological-artifacts_DATA/coin/"
    external_hard_disk_path_others = "/run/media/magnusjsc/T7/Classification-and-3D-reconstruction-of-archaeological-artifacts_DATA/others/"
    external_source_folder = "/run/media/magnusjsc/T7/Classification-and-3D-reconstruction-of-archaeological-artifacts_DATA/"

# File extensions to look for (tuple)
extensions_to_look_for = ('.jpg')

# Maximum dimensions for images
min_width = 100  # Minimum width
max_width = 800  # Maximum width
min_height = 100  # Minimum height
max_height = 600  # Maximum height

# Initialize counters
DATA_SLICE_COINS = 0
DATA_SLICE_OTHERS = 0
TOTAL_DATASET_SIZE = 0

# Define the CSV file path for image sizes
csv_file_path = "../image_sizes.csv"
DATA_SIZES = []

# Batch size
BATCH_SIZE = 50

# Threading 
NUM_THREADS = 100
thread_pool = ThreadPoolExecutor(max_workers=NUM_THREADS)  # Adjust max_workers as needed
# Lock mechanism
lock = threading.Lock()

class ThreadedProcess: 
    def __init__(self, data_slice, thread_id, path):
        self.data_slice = data_slice
        self.path = path
        # ID of the thread
        self.thread_id = thread_id 
    
    def run(self):
        object_type = ""
        global processed_images

        if self.path == file_list_coins:
            self.object_type = 'COIN' 
        else: 
            self.object_type = 'OTHERS'
        
        # Prepare thread pirce of data
        start_index = self.thread_id * self.data_slice
        end_index = min((self.thread_id + 1) * self.data_slice, len(self.data_slice))
        batch = [] 

        for index in range(start_index, end_index):
            filename = self.data_slice[index]
            try:
                source_path = os.path.join(external_source_folder, filename)
                image = Image.open(source_path)
                batch.append((filename, image))
            except Exception as e:
                print(f'Error opening {filename}: {str(e)}')
            if len(batch) == BATCH_SIZE:
                copy_images(batch, object_type, source_path)
            batch = [] 
        # After the loop, process any remaining images in the batch
        if batch:
            copy_images(batch)
            

def copy_images(list_of_images, object_type, source_path):
    for filename, image in list_of_images:
        # Use shutil.copy2 to copy the file
        try:
            destination_path = sort_by_type_and_status(image ,object_type)
            shutil.copy2(source_path, destination_path)
            # Update the number of processed images using the lock - Can lead to worse performance due to contention
            with lock:
                processed_images += 1  # Increment the count of processed images
                print(f'Processed images: {processed_images}. Remaining images: {TOTAL_DATASET_SIZE - processed_images}', flush=True, end='\r')
                # Update the processed_images.txt file
                with open(processed_images_file, 'w') as file:
                    file.write(str(processed_images))
        except FileExistsError:
            print(f"File at {destination_path} already exists. Skipping copy.", end='\r', flush=True)
        except Exception as e:
            print(f'Error copying {filename}: {str(e)}')

# Sort the images by wanted dimensions and check valid dimensions - Good and bad
def sort_by_type_and_status(image, object_type):
    dimension_status = check_dimension(image)  # Get the dimension status (GOOD or BAD)
    
    if object_type == 'COIN':
        if dimension_status == 'GOOD':
            destination_folder = external_destination_source_folder_good_coin
        else:
            destination_folder = external_destination_source_folder_bad_coin
    elif object_type == 'OTHERS':
        if dimension_status == 'GOOD':
            destination_folder = external_destination_source_folder_good_others
        else:
            destination_folder = external_destination_source_folder_bad_others
    else: 
        print("Invalid type or status")
        return

    return destination_folder

def check_dimension(image):
    width, height = image.size
    save_image_sizes(width, height)
    if width > max_width or width < min_width:
        return 'BAD'
    elif height > max_height or height < min_height:
        return 'BAD'
    else: 
        return 'GOOD'

# Save all image sizes 
def save_image_sizes(width, height):
    # Check if the CSV file already exists; if not, create it
    if not os.path.isfile(csv_file_path):
        # Create the CSV file with headers
        with open(csv_file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Width", "Height", "Occurrences"])
    # Check if the dimension already exists in the CSV
    dimension_exists = False
    for row in data:
        if len(row) == 3 and row[0] == str(width) and row[1] == str(height):
            # If it exists, increment the occurrence count
            row[2] = str(int(row[2]) + 1)
            dimension_exists = True
    # If not found, add a new entry to the CSV with an initial count of 1
    if not dimension_exists:
        data.append([image_id, str(width), str(height), "1"])
    # Save the updated data to the CSV file
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)

def main(): 
    # Check if a file with the count exists, and read it if found
    global processed_images_file
    processed_images_file = 'processed_images_dimensions.txt'
    if os.path.exists(processed_images_file):
        with open(processed_images_file, 'r') as file:
            processed_images = int(file.read())
    # Init dest. folders
    destination_dir_coin_bad = os.path.join(external_hard_disk_path_coin, f'bad_{max_width}_{max_height}')
    destination_dir_coin_good = os.path.join(external_hard_disk_path_coin, f'good_{max_width}_{max_height}')
    destination_dir_others_good = os.path.join(external_hard_disk_path_others, f'good_{max_width}_{max_height}')
    destination_dir_others_bad = os.path.join(external_hard_disk_path_others, f'bad_{max_width}_{max_height}')

    # Ensure the destination directories exist. Create them if they dont
    os.makedirs(destination_dir_coin_bad, exist_ok=True)
    os.makedirs(destination_dir_coin_good, exist_ok=True)
    os.makedirs(destination_dir_others_good, exist_ok=True)
    os.makedirs(destination_dir_others_bad, exist_ok=True)

    # Actual search path used by threads to get images - Check existence
    global file_list_coins 
    global file_list_others
    if os.path.exists(external_source_folder):
        print(f"YAS!: Directory found! - {external_source_folder}")
        file_list_coins = os.path.join(external_source_folder, 'coin')
    else:
        print(f"Error: Directory not found - {external_source_folder}")
        sys.exit(1)

    if os.path.exists(external_source_folder):
        file_list_others = os.path.join(external_source_folder, 'others')
    else:
        print(f"Error: Directory not found - {external_source_folder}")
        sys.exit(1)
    
    TOTAL_DATASET_SIZE = len(file_list_coins) + len(file_list_others)
    print(f'Total size of source dataset reading from: {TOTAL_DATASET_SIZE}')
    
    # Divide number of threads between two folders - Biggest get more i.e. others - Floor divisor
    NUM_THREADS_COINS = NUM_THREADS // 3 
    NUM_THREADS_OTHERS = NUM_THREADS - NUM_THREADS_COINS
    # Get data and image sizes - Floor divisor
    DATA_SLICE_COINS = len(file_list_coins) // NUM_THREADS_COINS
    DATA_SLICE_OTHERS = len(file_list_others) // NUM_THREADS_OTHERS
    print(f'Data slice for coins: {DATA_SLICE_COINS}')
    print(f'Data slice for others: {DATA_SLICE_OTHERS}')

    futures = []
    # Start processing in separate threads
    for i in range(NUM_THREADS_COINS):
        # Start - end slice
        data_slice = file_list_coins[i * DATA_SLICE_COINS : (i + 1) * DATA_SLICE_COINS]
        # Give data slice and thread number to ThreadProcess class
        thread = ThreadedProcess(data_slice, i, file_list_coins)
        # Start thread(s)
        futures.append(thread_pool.submit(thread.run))
    for j in range(NUM_THREADS_OTHERS):
        # Start - end slice
        data_slice = file_list_others[i * DATA_SLICE_OTHERS : (j + 1) * DATA_SLICE_OTHERS]
        # Give data slice and thread number to ThreadProcess class
        thread = ThreadedProcess(data_slice, i, file_list_others)
        # Start thread(s)
        futures.append(thread_pool.submit(thread.run))

    # Use thread_pool._threads to get the actual number of active threads
    active_threads = len(thread_pool._threads)
    print(f'Number of active threads: {active_threads}')
        
        # Wait for all threads to complete
    for future in as_completed(futures):
        future.result()

    # Shutdown the ThreadPoolExecutor to release resources
    thread_pool.shutdown()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Program interrupted. Terminating threads...")
        # Shutdown the ThreadPoolExecutor on program interruption
        thread_pool.shutdown(wait=False)
    except InterruptedError: 
        print("Program interrupted. Terminating threads...")
        # Shutdown the ThreadPoolExecutor on program interruption
        thread_pool.shutdown(wait=False)