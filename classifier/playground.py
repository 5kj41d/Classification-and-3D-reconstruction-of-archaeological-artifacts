import os
import shutil
import random

import torch
import torch.utils
from torch.utils.data import DataLoader
import torch.utils.data
from torchvision.transforms import v2 

from utils.iron_and_others_dataset import IronDataset, GeneratedDataset


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

#coins = r"C:\Users\Mate\Projects\Classifiers\data\augmented_iron\augmented_images"
coins = r"C:\Users\Mate\Projects\Classifiers\data\iron\iron"
others = r"C:\Users\Mate\Projects\Classifiers\data\others"
dataset = IronDataset(coin_dir=coins, 
                    others_dir=others, 
                    transform=transforms)

train_set, test_set = torch.utils.data.random_split(dataset, [0.8, 0.2])

train_loader = DataLoader(train_set, batch_size=1, shuffle=True, drop_last=True, num_workers=0)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, drop_last=True, num_workers=0)

gen_coins = r"C:\Users\Mate\Projects\Classifiers\data\generated_iron\full_iron"
gen_dataset = GeneratedDataset(
    coin_dir=coins,
    others_dir=others,
    gen_dir=gen_coins,
    transform=transforms,
    gen_image_num=1479)

print(gen_dataset.__len__())
'''print(dataset.__len__())
print(train_loader.__len__())
print(test_loader.__len__())'''

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

#move_random_files(coins, r"C:\Users\Mate\Projects\Classifiers\data\test_data\coin", 300)
#move_random_files(others, r"C:\Users\Mate\Projects\Classifiers\data\test_data\other", 300)



