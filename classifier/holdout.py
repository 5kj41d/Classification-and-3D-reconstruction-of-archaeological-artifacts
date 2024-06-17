import os
import shutil
import json
import numpy as np

import torch
import torch.utils
from torch.utils.data import DataLoader, Subset
import torch.utils.data
from torchvision.transforms import v2 
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from utils.iron_and_others_dataset import IronDataset, GeneratedDataset, RandomIronDataset
from models.lenet_dynamic import LeNet5_Dynamic
import test


def train_func(model, optimizer, train_loader, loss_fn):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

def test_func(model, test_loader, loss_fn):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()

            loss = loss_fn(output, target)

    final_accuracy = correct / len(test_loader.dataset)

    return loss, final_accuracy

def test_cnn(model, test_loader, loss_fn):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    test_loss = 0
    correct = 0
    total_pred = 0
    total_true = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            _, predicted = torch.max(pred, 1)
            correct += (predicted == y).sum().item()
            total_pred += predicted.sum().item()
            total_true += y.sum().item()

    accuracy = correct / len(test_loader.dataset)
    precision = correct / total_pred
    recall = correct / total_true
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    result = {
        "accuracy": accuracy,
        "test_loss": test_loss / len(test_loader),
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }

    return result


def write_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


og_only_config = {
    "conv_kernel": 7, 
    "conv_stride": 1, 
    "pool_kernel": 2, 
    "pool_stride": 2, 
    "fc1_size": 512, 
    "fc2_size": 256, 
    "batch_size": 64, #64
    "epochs": 1, # 150
    "lr": 0.0003855676609555027, 
    "path": r"C:\Users\Mate\Projects\Classifiers\saved\og"
}

#coins = r"C:\Users\Mate\Projects\Classifiers\data\augmented_iron\augmented_images"
#others = r"C:\Users\Mate\Projects\Classifiers\data\split-others\3558others"
#viking = r"C:\Users\Mate\Downloads\viking\3558"

coins = r"C:\Users\Mate\Projects\Classifiers\data\iron\iron"
others = r"C:\Users\Mate\Projects\Classifiers\data\split-others\1779others"
transforms = v2.Compose([
    v2.Resize(size=(64, 64)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
dataset = IronDataset(coin_dir=coins, others_dir=others, transform=transforms)
train_set, test_set = torch.utils.data.random_split(dataset, [0.8, 0.2])

train_loader = DataLoader(
    dataset=train_set, 
    batch_size=og_only_config["batch_size"],
    num_workers=0,
    shuffle=True,
    )

test_loader = DataLoader(
    dataset=test_set, 
    batch_size=og_only_config["batch_size"],
    num_workers=0,
    )


# Define model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = test.load_model(og_only_config) # Load og model

model = LeNet5_Dynamic(
conv_kernel=og_only_config["conv_kernel"], 
conv_stride=og_only_config["conv_stride"], 
pool_kernel=og_only_config["pool_kernel"], 
pool_stride=og_only_config["pool_stride"], 
fc1_size=og_only_config["fc1_size"], 
fc2_size=og_only_config["fc2_size"])

model.to(device)

# Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=og_only_config["lr"])

# Run the training and get accuracy and loss from test function
for epoch in range(og_only_config["epochs"]):
    train_func(model, optimizer, train_loader, loss_fn)

final_loss, final_accuracy = test_func(model, test_loader, loss_fn)
print(f"Loss: {final_loss:.4f}, Accuracy: {final_accuracy:.4f}")
result = test_cnn(model, test_loader, loss_fn)
write_json(result, os.path.join(og_only_config["path"], "result99.json"))
print("-------")