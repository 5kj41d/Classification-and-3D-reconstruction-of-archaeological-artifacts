import os
import shutil
import random

import torch
import torch.utils
from torch.utils.data import DataLoader, Subset
import torch.utils.data
from torchvision.transforms import v2 


def show_dataset_item(dataset, index):
    image, label = dataset[index]
    print(f"Index: {index}")
    print(f"Label: {label}")
    print(f"Image Size: {image.size()}")  # Since image is a tensor, size() will give the dimensions
    # To display the image, convert the tensor to a PIL image
    img = v2.ToPILImage()(image)
    img.show()

transforms = v2.Compose([
    v2.Resize(size=(64, 64)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# show_dataset_item(dataset, 0)


def copy_random_files(target_dir, dest_dir, num_files):
    """Copies num_files random files from target_dir to dest_dir"""
    files = os.listdir(target_dir)
    if num_files > len(files):
        raise ValueError("Not enough files in the target directory")

    random_files = random.sample(files, num_files)
    for file in random_files:
        shutil.copy2(os.path.join(target_dir, file), dest_dir)

def move_random_files(target_dir, dest_dir, num_files):
    """Moves num_files random files from target_dir to dest_dir"""
    files = os.listdir(target_dir)
    if num_files > len(files):
        raise ValueError("Not enough files in the target directory")

    random_files = random.sample(files, num_files)
    for file in random_files:
        shutil.move(os.path.join(target_dir, file), dest_dir)


#move_random_files(r"C:\Users\Mate\Projects\Classifiers\data\viking\viking", r"C:\Users\Mate\Projects\Classifiers\data\viking\test_1", 1000)
#move_random_files(r"C:\Users\Mate\Projects\Classifiers\data\viking\viking", r"C:\Users\Mate\Projects\Classifiers\data\viking\test_2", 1000)
#move_random_files(r"C:\Users\Mate\Projects\Classifiers\data\viking\viking", r"C:\Users\Mate\Projects\Classifiers\data\viking\test_3", 1000)

#move_random_files(r"C:\Users\Mate\Projects\Classifiers\data\others", r"C:\Users\Mate\Projects\Classifiers\data\split-others\test_1", 1000)
#move_random_files(r"C:\Users\Mate\Projects\Classifiers\data\others", r"C:\Users\Mate\Projects\Classifiers\data\split-others\test_2", 1000)
#move_random_files(r"C:\Users\Mate\Projects\Classifiers\data\others", r"C:\Users\Mate\Projects\Classifiers\data\split-others\34954others2", 34954)

#copy_random_files(r"/home/student.aau.dk/ra86nk/data/medieval/medieval", r"/home/student.aau.dk/ra86nk/data/van_plus_filt/", 34954)
copy_random_files(r"/home/student.aau.dk/ra86nk/data/34954others", r"/home/student.aau.dk/ra86nk/data/69908others/", 34954)
