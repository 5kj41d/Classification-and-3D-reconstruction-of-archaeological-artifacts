import os
import tempfile
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 
import torch.nn as nn
import torch.optim as optim

from ray import tune, train
from ray.tune.schedulers import ASHAScheduler
from ray.train import Checkpoint

from utils.iron_and_others_dataset import IronDataset
from models.lenet_dynamic import LeNet5_Dynamic

import json
import matplotlib.pyplot as plt
import numpy as np


# Define training function
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


# Define evaluation function
def test_func(model, test_leader, loss_fn):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0
    val_loss = 0
    val_steps = 0
    with torch.no_grad():
        for batch, (data, target) in enumerate(test_leader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            loss = loss_fn(output, target)
            val_loss += loss.cpu().numpy()
            val_steps += 1

    final_loss = val_loss / val_steps
    final_accuracy = correct / total

    return final_loss, final_accuracy


# Define Tuner setup for training the model
def tune_trainer(config):

    # Set up transformations, dataset, and data loader
    transforms = v2.Compose([
        v2.Resize(size=(64, 64)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    coins = r"/home/student.aau.dk/ra86nk/data/medieval/medieval"
    others = r"/home/student.aau.dk/ra86nk/data/34954others"
    dataset = IronDataset(coin_dir=coins, 
                        others_dir=others, 
                        transform=transforms)

    train_set, test_set = torch.utils.data.random_split(dataset, [0.8, 0.2])

    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=config["batch_size"], shuffle=False, num_workers=0)

    # Define model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = LeNet5()
    model = LeNet5_Dynamic(
        conv_kernel=config["conv_kernel"], 
        conv_stride=config["conv_stride"], 
        pool_kernel=config["pool_kernel"], 
        pool_stride=config["pool_stride"], 
        fc1_size=config["fc1_size"], 
        fc2_size=config["fc2_size"])
    model.to(device)

    # Define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    # Run the training and get accuracy and loss from test function
    for epoch in range(config["epochs"]):
        train_func(model, optimizer, train_loader, loss_fn)
        loss, accuracy = test_func(model, test_loader, loss_fn)

    # Save model checkpoint
    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
        checkpoint = None
        if (epoch + 1) % 5 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(temp_checkpoint_dir, "model.pth")
            )
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

        # Send current training result back to Tune
        train.report({"loss": loss, "accuracy": accuracy}, checkpoint=checkpoint)

def main():
    # Define search space for tuning
    search_space = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.grid_search([8, 16, 32, 64]),
        "epochs": 10,
        "conv_kernel": tune.grid_search([3, 5, 7]),
        "conv_stride": tune.choice([1, 2, 3]),
        "pool_kernel": tune.choice([2, 3]),
        "pool_stride": tune.choice([1, 2]),
        "fc1_size": tune.choice([128, 256, 512]),
        "fc2_size": tune.choice([64, 128, 256]),
    }


    # Run tuning
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(tune_trainer),
            resources={"cpu": 4, "gpu": 1},
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            num_samples=5,
            scheduler=ASHAScheduler( max_t=10, grace_period=1, reduction_factor=2),
        ),
        run_config=train.RunConfig(
            name="medieval_vanilla",
            storage_path=r"/home/student.aau.dk/ra86nk/ray-results",
        ),
        param_space=search_space,
    )

    results = tuner.fit()

    best_loss = results.get_best_result("loss", "min")
    best_acc = results.get_best_result("accuracy", "max")

    with open(r"/home/student.aau.dk/ra86nk/ray-results/medieval_vanilla/best_loss.json", "w", encoding='utf-8') as f:
        json.dump(best_loss.metrics, f, ensure_ascii=False, indent=4)
    with open(r"/home/student.aau.dk/ra86nk/ray-results/medieval_vanilla/best_accuracy.json", "w", encoding='utf-8') as f:
        json.dump(best_acc.metrics, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()