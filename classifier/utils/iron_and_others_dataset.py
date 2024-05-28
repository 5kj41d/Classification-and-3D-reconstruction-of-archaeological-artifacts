import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random

class IronDataset(Dataset):
    def __init__(self, coin_dir, others_dir, transform=None):
        # Provide path to coins and others
        # The dataset should grab the same amount of images from both, based on the lower amount

        self.coin_dir = coin_dir
        self.others_dir = others_dir
        self.transform = transform
        
        # Load images from source 1 and assign label 0
        self.coin_images = [(os.path.join(coin_dir, img), 0) for img in os.listdir(coin_dir)]
        
        # Load images from source 2 and assign label 1
        self.other_images = [(os.path.join(others_dir, img), 1) for img in os.listdir(others_dir)]
        
        # Ensure equal number of images from both sources
        min_len = min(len(self.coin_images), len(self.other_images))
        self.coin_images = random.sample(self.coin_images, min_len)
        self.other_images = random.sample(self.other_images, min_len)
        
        # Combine the images and labels
        self.all_images = self.coin_images + self.other_images

        # Optionally shuffle the combined dataset
        random.shuffle(self.all_images)

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        img_path, label = self.all_images[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
class GeneratedDataset(Dataset):
    def __init__(self, coin_dir, gen_dir, others_dir, transform=None, gen_image_num=1479):
        # Provide path to coins, generated images, and others

        self.coin_dir = coin_dir
        self.gen_dir = gen_dir
        self.others_dir = others_dir
        self.transform = transform
        self.gen_image_num = gen_image_num
        
        # Load images from source 1 and assign label 0
        self.coin_images = [(os.path.join(coin_dir, img), 0) for img in os.listdir(coin_dir)]

        # Load images from source 2 and assign label 0
        self.gen_images = [(os.path.join(gen_dir, img), 0) for img in os.listdir(gen_dir)]

        # Load images from source 3 and assign label 1
        self.other_images = [(os.path.join(others_dir, img), 1) for img in os.listdir(others_dir)]
        
        # Take every coin image, specified num of generated images, their sum as other images
        coin_len = len(self.coin_images)
        other_len = coin_len + gen_image_num

        self.coin_images = random.sample(self.coin_images, coin_len)
        self.gen_images = random.sample(self.gen_images, gen_image_num)
        self.other_images = random.sample(self.other_images, other_len)
        
        # Combine the images and labels
        self.all_images = self.coin_images + self.gen_images + self.other_images

        # Optionally shuffle the combined dataset
        random.shuffle(self.all_images)

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        img_path, label = self.all_images[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

'''class IronDataset(Dataset):
    """Dataset class for Iron dataset with images in different subfolders labeled by their names """

    def __init__(self, coin_dir, others_dir, transform=None, num_per_folder=500):
        """
        Args:
            coin_dir (string): Directory with images of iron coins
            others_dir (string): Directory with images of other artifacts
            transform (callable, optional): Optional transform to be applied
                on each image. Defaults to None
            num_per_folder (int, optional): Number of images to take randomly from each folder
                Defaults to 500
        """
        self.coin_dir = coin_dir
        self.others_dir = others_dir
        self.transform = transform
        self.num_per_folder = num_per_folder

        #self.iron_folder = os.path.join(root_dir, coin_subfolder)
        #self.others_folder = os.path.join(root_dir, others_subfolder)
        self.iron_folder = coin_dir
        self.others_folder = others_dir
        self.iron_file_names = [os.path.join(self.iron_folder, f) for f in os.listdir(self.iron_folder)]
        self.others_file_names = [os.path.join(self.others_folder, f) for f in os.listdir(self.others_folder)]
        self.num_iron = len(self.iron_file_names)
        self.num_others = len(self.others_file_names)
        self.num_total = self.num_iron + self.num_others
        self.file_names = self.iron_file_names + self.others_file_names

        if num_per_folder > self.num_iron:
            self.num_iron_take = self.num_iron
        else:
            self.num_iron_take = num_per_folder

        if num_per_folder > self.num_others:
            self.num_others_take = self.num_others
        else:
            self.num_others_take = num_per_folder

        self.iron_take_names = random.sample(self.iron_file_names, self.num_iron_take)
        self.others_take_names = random.sample(self.others_file_names, self.num_others_take)
        self.file_names_take = self.iron_take_names + self.others_take_names

    def __len__(self):
        """Return the number of images in the dataset"""
        return len(self.file_names_take)

    def __getitem__(self, idx):
        """Return a sample from dataset"""
        img = Image.open(self.file_names_take[idx])
        if os.path.dirname(self.file_names_take[idx]) == self.iron_folder:
            target = 0
        elif os.path.dirname(self.file_names_take[idx]) == self.others_folder:
            target = 1

        if self.transform:
            img = self.transform(img)

        return img, target

'''