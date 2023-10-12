import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
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

NUM_THREADS = 50
# Create a ThreadPoolExecutor to manage threads
thread_pool = ThreadPoolExecutor(max_workers=NUM_THREADS)  # Adjust max_workers as needed
# Lock mechanism
lock = threading.Lock()


# Thread class for processing data concurrently
class ThreadedProcess:
    def __init__(self, data, thread_id):
        self.data = data
        # ID of the thread
        self.thread_id = thread_id
    
    def run(self):
        # A thread for each slice based on id * DATASET_SLICE. The end index is the start of the other. Also check if it is the last thread
        if self.thread_id == 0 and START_INDEX > 0:
            # Make sure the first thread is starting the correct index if some images have already been processed
            start_index = int(START_INDEX)
            end_index = int(((self.thread_id + 1) * DATASET_SLICE) + START_INDEX) if self.thread_id < NUM_THREADS - 1 else total_images
        else:
            start_index = int(self.thread_id * DATASET_SLICE + START_INDEX)
            end_index = int((self.thread_id + 1) * DATASET_SLICE+ START_INDEX) if self.thread_id < NUM_THREADS - 1 else total_images

        # Process and copy images
        for index in range(start_index, end_index):
            item = self.data.iloc[index]
            # Process the 'item' (your data processing logic goes here)
            filename = item['filnavn']
            thesaurus = item['thesaurus']
            if any(thesaurus.startswith(label) for label in thesaurus_label) and filename in os.listdir(source_directory):
                destination_dir = destination_directory_coin
            else:
                destination_dir = destination_directory_others

            source_path = os.path.join(source_directory, filename)
            destination_path = os.path.join(destination_dir, filename)
            '''
            print(f'SOURCE PATH: {source_path}')
            print(f'DESTINATION PATH {destination_path}')
            '''

            # Use shutil.copy2 to copy the file
            try:
                shutil.copy2(source_path, destination_path)
                # Update the number of processed images using the lock
                with lock:
                    global processed_images
                    processed_images += 1  # Increment the count of processed images
                    print(f'Processed images: {processed_images}. Remaining images: {total_images - processed_images}', flush=True, end='\r')
            except FileExistsError:
                print(f"File at {destination_path} already exists. Skipping copy.", end='\r', flush=True)
            except Exception as e:
                print(f'Error copying {filename}: {str(e)}')

def main():
    # Global and shared path variables
    global source_directory
    global destination_directory_coin
    global destination_directory_others
    global DATASET_SLICE 
    global TOTAL_IMAGES_TO_COPY 
    global processed_images
    global total_images 
    global START_INDEX 

    processed_images = 0
    
    # Load the CSV file
    df = pd.read_csv(csv_file_path, delimiter=';')
    total_images = df.shape[0]  # Define total_images

    if not os.path.exists(csv_file_path):
        print(f'CSV file "{csv_file_path}" not found.')
        return
    # Check if there are images that have already been processed
    processed_images_file = '../processed_images.txt'  # Modify the path as needed
    if os.path.exists(processed_images_file):
        with open(processed_images_file, 'r') as file: # r = read
            processed_images = int(file.read())
    if processed_images > 0:
        TOTAL_IMAGES_TO_COPY = total_images - processed_images
        START_INDEX = processed_images
    else: 
        TOTAL_IMAGES_TO_COPY = total_images
        START_INDEX = 0

    # Update the DATASET_SLICE variable to contain only the neccesary part of un-processed data, if the case
    DATASET_SLICE = TOTAL_IMAGES_TO_COPY / NUM_THREADS
    print(f'Total images left to copy: {TOTAL_IMAGES_TO_COPY}')
    print(f'Data slice for each thread: {DATASET_SLICE}')

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

    futures = []
    # Split the data and create threads for concurrent processing
    for i in range(0,NUM_THREADS):
        thread = ThreadedProcess(df, i)
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
        # Update processed_images.txt with the last processed image index
        print(f'Writing the processed image count {processed_images} to file for future processing.')
        with open('../processed_images.txt', 'w') as file:
            file.write(str(processed_images))
    finally:
        # Update processed_images.txt with the last processed image index
        print(f'Writing the processed image count {processed_images} to file for future processing.')
        with open('../processed_images.txt', 'w') as file:
            file.write(str(processed_images))

