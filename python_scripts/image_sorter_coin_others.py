import os
import shutil
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import platform
import threading

'''
Important note: Play around with the amount of workers and the size of each batch. Depends on your system.
This can decrease the time taken for this script to run. 
'''

# Define paths:
# NOTE: Change these for 'your' machine.
external_hard_disk_path_windows = r"E:\Classification-and-3D-reconstruction-of-archaeological-artifacts_DATA\DIME images"
external_destination_source_folder_windows = r"E:\Classification-and-3D-reconstruction-of-archaeological-artifacts_DATA"
external_hard_disk_path_linux = "/run/media/magnusjsc/T7/Classification-and-3D-reconstruction-of-archaeological-artifacts_DATA/DIME images/"
external_destination_source_folder_linux = "/run/media/magnusjsc/T7/Classification-and-3D-reconstruction-of-archaeological-artifacts_DATA/"
csv_file_path = r"..\DIME billeder.csv"

# File extensions to look for (tuple)
extensions_to_look_for = ('.jpg')

# Search thesaurus
thesaurus_label = ['dime.find.coin']
# Define a batch size for copying images
BATCH_SIZE = 100
# Create a ThreadPoolExecutor to manage threads
thread_pool = ThreadPoolExecutor(max_workers=500)  # Adjust max_workers as needed
# Lock mechanism
lock = threading.Lock()
# Global variable for processed images
processed_images = 0
total_images = 0

# Thread class for copy images concurrently 
class ThreadedCopy:
    def __init__(self, filename_destination_pairs, source_directory):
        self.filename_destination_pairs = filename_destination_pairs
        self.source_directory = source_directory

    def run(self):
        batch = []
        for filename, destination_dir in self.filename_destination_pairs:
            source_path = os.path.join(self.source_directory, filename)
            destination_path = os.path.join(destination_dir, filename)
            batch.append((source_path, destination_path))
        try:
            # Batch copy the images
            for source_path, destination_path in batch:
                shutil.copy2(source_path, destination_path)
            # Update the number of processed images using the lock
            with lock:
                global processed_images
                processed_images += len(batch)
                # Save the value to a file for resuming if stopped mid-through
                with open('../processed_images.txt', 'w') as file:
                    file.write(str(processed_images))
                # Calculate and print the progress
                remaining_images = total_images - processed_images
                print(f'Processed images: {processed_images}, Remaining images: {remaining_images}', flush=True, end='\r')
        except Exception as e:
            print(f'Error copying images: {str(e)}', flush=True, end='\r')


# Process and sort images
def process_and_sort_images(df, source_directory, destination_directory_coin, destination_directory_others):
    try:
        # Check if the source directory exists
        if not os.path.exists(source_directory):
            print(f'Source directory {source_directory} does not exist')
            return
        # Check if the destination folders exist. Else create them
        for destination_dir in (destination_directory_coin, destination_directory_others):
            if not os.path.exists(destination_dir):
                os.makedirs(destination_dir)
        # Create a list of filenames to copy
        filenames_to_copy = []
        destination_dir = None
        # Iterate through and prepare images for copying
        for _, row in df.iterrows():
            filename = row['filnavn']
            thesaurus = row['thesaurus']
            if any(thesaurus.startswith(label) for label in thesaurus_label) and filename in os.listdir(source_directory):
                destination_dir = destination_directory_coin
            else:
                destination_dir = destination_directory_others
            # Append the filename and destination to the batch as a tuple
            filenames_to_copy.append((filename, destination_dir))
            # If the batch size is reached, submit it to the thread pool
            if len(filenames_to_copy) == BATCH_SIZE:
                # New thread is started with the current batch at this point - No thread will have the same batch
                thread_pool.submit(ThreadedCopy(filenames_to_copy, source_directory).run)
                filenames_to_copy = []  # Reset the list for the next batch
                
        # Copy any remaining files in the (last) batch
        if filenames_to_copy:
            thread_pool.submit(ThreadedCopy(filenames_to_copy, source_directory).run)
    except Exception as e:
        print(f'An error occurred: {str(e)}')

def main():
    # Load the CSV file
    df = pd.read_csv(csv_file_path, delimiter=';')
    global total_images 
    total_images = df.shape[0]  # Define total_images

    if not os.path.exists(csv_file_path):
        print(f'CSV file "{csv_file_path}" not found.')
        return

    # Path of images and destination of coin images based on OS.
    if platform.system() == "Windows":
        print('Running windows.')
        source_directory = external_hard_disk_path_windows
        destination_directory_coin = os.path.join(external_destination_source_folder_windows, "coin")
        destination_directory_others = os.path.join(external_destination_source_folder_windows, "others")
    else:
        print('Running Linux.')
        source_directory = external_hard_disk_path_linux
        destination_directory_coin = os.path.join(external_destination_source_folder_linux, "coin")
        destination_directory_others = os.path.join(external_destination_source_folder_linux, "others")

    # Check if there are images that have already been processed
    processed_images_file = '../processed_images.txt'  # Modify the path as needed
    if os.path.exists(processed_images_file):
        with open(processed_images_file, 'r') as file: # r = read
            last_processed_index = int(file.read())
    if last_processed_index > 0:
        # Skip the processed images in the DataFrame
        df = df.iloc[last_processed_index:]
        # Update the manager to the number of already processed images
        global processed_images 
        processed_images = last_processed_index
        print(f'Number of processed images already: {last_processed_index}')
    # Let's go....
    process_and_sort_images(df, source_directory, destination_directory_coin, destination_directory_others)

if __name__ == "__main__":
    main()
