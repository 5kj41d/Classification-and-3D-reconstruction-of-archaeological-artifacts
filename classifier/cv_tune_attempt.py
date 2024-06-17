import os
import json
import tempfile
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

from ray import tune, train, air
from ray.tune.schedulers import ASHAScheduler
from ray.train import Checkpoint
from ray.air import session

from utils.iron_and_others_dataset import IronDataset, GeneratedDataset, RandomIronDataset
from models.lenet_dynamic import LeNet5_Dynamic




def train_function(model, optimizer, train_loader, loss_fn):
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

def eval_function(model, test_loader, fold, config):
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

    '''confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels=["coins", "others"])
    disp.plot()
    path = config["path"]
    filename = f"confusion_matrix{fold + 1}.png"
    plt.savefig(os.path.join(path, filename))'''
    
    result = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "roc_auc": roc_auc,
    }

    return result

def tune_function(config):
    pass
    # Set up transformations, dataset
    transforms = v2.Compose([
        v2.Resize(size=(64, 64)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = IronDataset(
        coin_dir=r"C:\Users\Mate\Projects\Classifiers\data\viking\test_1",
        others_dir=r"C:\Users\Mate\Projects\Classifiers\data\split-others\test_1",
        transform=transforms
    )

    # Initialize lists to store cross-validation results
    cv_metrics = {
        "loss": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1_score": [],
        "roc_auc": []
    }

    # Initialize stratified K-Fold Cross-Validation
    labels = [label for _, label in dataset.all_images]
    skf = StratifiedKFold(n_splits=5, shuffle=True)

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        train_set = Subset(dataset, train_idx)
        eval_set = Subset(dataset, val_idx)

        train_loader = DataLoader(
            train_set,
            batch_size=config["batch_size"],
            num_workers=0,
        )

        eval_loader = DataLoader(
            eval_set,
            batch_size=config["batch_size"],
            num_workers=0,
        )

        # Define model and device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LeNet5_Dynamic(
            conv_kernel=config["conv_kernel"], 
            conv_stride=config["conv_stride"], 
            pool_kernel=config["pool_kernel"], 
            pool_stride=config["pool_stride"], 
            fc1_size=config["fc1_size"], 
            fc2_size=config["fc2_size"]        
        )
        model.to(device)

        # Define loss function and optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config["lr"])

        # Lists to calculate avg loss
        loss_list = []
        avg_fold_loss = []

        # Train model on training set
        for epoch in range(config["epochs"]):
            loss_per_epoch = train_function(model=model, train_loader=train_loader, loss_fn=loss_fn, optimizer=optimizer)
            loss_list.append(loss_per_epoch)

        # Evaluate model on validation set
        eval_results = eval_function(model=model, test_loader=eval_loader, fold=fold, config=config)

        # Collect cross-validation results and training loss
        for key, value in eval_results.items():
            cv_metrics[key].append(value)

        avg_fold_loss = sum(loss_list) / len(loss_list)
        cv_metrics["loss"].append(avg_fold_loss)

        # Save model checkpoint after each fold
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            torch.save(model.state_dict(), os.path.join(temp_checkpoint_dir, "model.pth"))
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            #train.report({"fold": fold, "checkpoint": checkpoint})

    # Compute average metrics over all folds
    avg_metrics = {key: sum(values) / len(values) for key, values in cv_metrics.items()}

    # Report average metrics to Tune
    train.report(avg_metrics)

def main():
    pass
    # Define search space for hyperparameters
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

    # Set up and configure the tuner
    scheduler = ASHAScheduler(
    metric="accuracy",
    mode="max",
    max_t=10,
    grace_period=1,
    reduction_factor=2)

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(tune_function),
            resources={"cpu": 1, "gpu": 0},
        ),
        tune_config=tune.TuneConfig(
            num_samples=1,
            scheduler=scheduler,
        ),
        run_config=train.RunConfig(
            name="medieval_vanilla",
            storage_path=r"C:\Users\Mate\Projects\Classifiers\ray_results",
        ),
        param_space=search_space,
    )

    # Run the tuner to perform hyperparameter optimization
    results = tuner.fit()

    # Retrieve the best results
    best_loss = results.get_best_result("loss", "min")
    best_acc = results.get_best_result("accuracy", "max")

    # Save the best results to files
    with open(r"C:\Users\Mate\Projects\Classifiers\ray_results\medieval_vanilla\best_loss.json", "w", encoding='utf-8') as f:
        json.dump(best_loss.metrics, f, ensure_ascii=False, indent=4)
    with open(r"C:\Users\Mate\Projects\Classifiers\ray_results\medieval_vanilla\best_accuracy.json", "w", encoding='utf-8') as f:
        json.dump(best_acc.metrics, f, ensure_ascii=False, indent=4)    

if __name__ == "__main__":
    main()