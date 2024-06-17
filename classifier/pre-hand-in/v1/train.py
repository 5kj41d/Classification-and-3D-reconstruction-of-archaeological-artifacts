import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 
from utils.iron_and_others_dataset import IronDataset
from models.iron_resnet import IronResNet
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

# Define transformations
transforms = v2.Compose([
    v2.Resize(size=(64, 64)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# Create dataset with transformations
coin_subfolder = r".\data\iron"
others_subfolder = r".\data\others"
number_of_images = 100

dataset = IronDataset(coin_dir=coin_subfolder, 
                      others_dir=others_subfolder, 
                      transform=transforms,
                      num_per_folder=number_of_images)



# Define data loader
train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
test_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)


# Define model
model = IronResNet(num_classes=1)

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

# Define loss function and optimizer
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training loop

def f1_score_tensor(pred, y):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    y = y.type(torch.float)

    precision = torch.sum(pred * y, dim=0) / (torch.sum(pred, dim=0) + 1e-10)
    recall = torch.sum(pred * y, dim=0) / (torch.sum(y, dim=0) + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

    return f1


def train_loop(num_epochs):
    """
    Simple CNN training loop

    :param num_epochs: Number of epochs to train the model
    """

    total_f1 = 0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        total_loss = 0
        correct = 0
        for batch, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            pred = pred.squeeze(1)  # Squeeze the dimension to match the target size
            y = y.type(torch.float)
            loss = loss_fn(pred, y)
            loss.backward()
            correct += (pred.view(-1) == y).type(torch.float).sum().item()

            total_loss += loss.item()

        accuracy = correct / len(train_loader.dataset)

        f1_score = f1_score_tensor(pred, y)
        total_f1 += f1_score.item()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  accuracy: {(100*accuracy):>0.1f}%  f1 score: {(100*f1_score):>0.1f}%  [{current:>5d}/{len(train_loader.dataset):>5d}]")

        log_path = "train_log.csv"
        with open(log_path, "a") as f:
            f.write(f"{epoch+1},{total_loss/(len(train_loader.dataset)//32)},{accuracy:.4f},{f1_score:.4f}\n")

        save_path = "model_epoch{}.pth".format(epoch+1)
        torch.save(model.state_dict(), save_path)
        print("Model saved to {}".format(save_path))



# Test set accuracy
def test_loop(dataloader, model, loss_fn):
    """
    Evaluate model accuracy on test set
    """
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            pred = pred.squeeze(1)
            y = y.type(torch.float)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.view(-1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# Display image and label.
train_features, train_labels = next(iter(train_loader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
image_np = img.numpy().transpose(1, 2, 0)
plt.imshow(image_np)
plt.show()
print(f"Label: {label}")

# Main 
#train_loop(3)
#test_loop(test_loader, model, loss_fn)