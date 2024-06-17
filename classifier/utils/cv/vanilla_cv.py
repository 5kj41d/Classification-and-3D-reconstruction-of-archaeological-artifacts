import os
import json
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.utils
from torch.utils.data import DataLoader, Subset
import torch.utils.data
from torchvision.transforms import v2 
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import StratifiedKFold
import sklearn.metrics as metrics

from utils.iron_and_others_dataset import IronDataset, GeneratedDataset, RandomIronDataset
from models.lenet_dynamic import LeNet5_Dynamic


def train_func(model, optimizer, train_loader, loss_fn):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    epoch_loss = 0.0
    total_samples = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        # Accumulate the loss and sample count
        epoch_loss += loss.item() * data.size(0)
        total_samples += data.size(0)

    avg_epoch_loss = epoch_loss / total_samples

    return avg_epoch_loss

def best_test(model, test_loader, fold, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            _, predicted = torch.max(pred.data, 1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred, zero_division=1)
    recall = metrics.recall_score(y_true, y_pred, zero_division=1)
    f1_score = metrics.f1_score(y_true, y_pred, zero_division=1)
    roc_auc = metrics.roc_auc_score(y_true, y_pred)

    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels=["coins", "others"])
    disp.plot()
    path = config["path"]
    filename = f"confusion_matrix{fold + 1}.png"
    plt.savefig(os.path.join(path, filename))
    
    result = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "roc_auc": roc_auc,
    }

    return result

def write_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def main():

    og_only_config = {
        "conv_kernel": 7, 
        "conv_stride": 1, 
        "pool_kernel": 2, 
        "pool_stride": 2, 
        "fc1_size": 512, 
        "fc2_size": 256, 
        "batch_size": 64, #64
        "epochs": 150, # 150
        "lr": 0.0003855676609555027, 
        "path": r"C:\Users\Mate\Projects\Classifiers\saved\og"
    }

    #coins = r"C:\Users\Mate\Projects\Classifiers\data\augmented_iron\augmented_images"
    viking = r"C:\Users\Mate\Downloads\viking\3558"
    #others = r"C:\Users\Mate\Projects\Classifiers\data\split-others\3558others"

    coins = r"C:\Users\Mate\Projects\Classifiers\data\iron\iron"
    others = r"C:\Users\Mate\Projects\Classifiers\data\split-others\1779others"

    transforms = v2.Compose([
        v2.Resize(size=(64, 64)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    fold_data = IronDataset(coin_dir=coins, others_dir=others, transform=transforms)

    labels = [label for _, label in fold_data.all_images]

    skf = StratifiedKFold(n_splits=3, shuffle=True)

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"Fold {fold + 1}")
        print("Train indices length:", len(train_idx))
        print("Test indices length:", len(test_idx))
        print("Train labels distribution:", np.bincount([labels[i] for i in train_idx]))
        print("Test labels distribution:", np.bincount([labels[i] for i in test_idx]))

        train_loader = DataLoader(
            dataset=fold_data, 
            batch_size=og_only_config["batch_size"],
            num_workers=2,
            sampler=torch.utils.data.SubsetRandomSampler(train_idx)
            )
        
        test_loader = DataLoader(
            dataset=fold_data, 
            batch_size=og_only_config["batch_size"],
            num_workers=0,
            sampler=torch.utils.data.SubsetRandomSampler(test_idx)
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

        epoch_losses = {}

        # Run the training 
        for epoch in range(og_only_config["epochs"]):
            loss = train_func(model, optimizer, train_loader, loss_fn)
            epoch_losses[f"epoch_{epoch + 1}"] = loss
        
        # Run the test
        print("Training done, beginning test...")
        result = best_test(model, test_loader, fold, og_only_config)
        acc = result["accuracy"]
        print(f"Accuracy: {acc:.4f}")

        # Save results and losses
        write_json(result, os.path.join(og_only_config["path"], f"result{fold + 1}.json"))
        write_json(epoch_losses, os.path.join(og_only_config["path"], f"losses_fold{fold + 1}.json"))

        # Save model
        print("Fold evaluated, saving model...")
        model_path = os.path.join(og_only_config["path"], f"model_fold_{fold + 1}.pt")
        torch.save(model.state_dict(), model_path)
        print("Model saved.")
        print("-------")

if __name__ == '__main__':
    main()