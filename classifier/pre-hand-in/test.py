import os
import json
import torch
import torch.nn.functional as F
from torchvision.transforms import v2 
from utils.iron_and_others_dataset import IronDataset
from torch.utils.data import DataLoader
from models.lenet_dynamic import LeNet5_Dynamic

def test_cnn(model, test_loader):

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
            test_loss += F.cross_entropy(pred, y).item()
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

def load_model(config):
    model = LeNet5_Dynamic(
    conv_kernel=config["conv_kernel"], 
    conv_stride=config["conv_stride"], 
    pool_kernel=config["pool_kernel"], 
    pool_stride=config["pool_stride"], 
    fc1_size=config["fc1_size"], 
    fc2_size=config["fc2_size"])

    model.load_state_dict(torch.load(os.path.join(config["path"], "model.pth")))

    return model


def main():

    transforms = v2.Compose([
    v2.Resize(size=(64, 64)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = IronDataset(coin_dir=r"C:\Users\Mate\Projects\Classifiers\data\test_data\coin", 
                        others_dir=r"C:\Users\Mate\Projects\Classifiers\data\test_data\other", 
                        transform=transforms)

    test_loader = DataLoader(dataset, batch_size=1, drop_last=True, shuffle=False, num_workers=0)

    # Initialize config for models
    og_only_config = {
        "conv_kernel": 7, 
        "conv_stride": 1, 
        "pool_kernel": 2, 
        "pool_stride": 2, 
        "fc1_size": 512, 
        "fc2_size": 256,
        "path": r"C:\Users\Mate\Projects\Classifiers\saved\og"
    }

    bootstrap_config = {
        "conv_kernel": 5, 
        "conv_stride": 1, 
        "pool_kernel": 3, 
        "pool_stride": 1, 
        "fc1_size": 128, 
        "fc2_size": 64,
        "path": r"C:\Users\Mate\Projects\Classifiers\saved\bootstrap"
    }

    augmented_config = {
        "conv_kernel": 5, 
        "conv_stride": 1, 
        "pool_kernel": 2, 
        "pool_stride": 2, 
        "fc1_size": 512, 
        "fc2_size": 256,
        "path": r"C:\Users\Mate\Projects\Classifiers\saved\augmented"
    }

    # Load pytorch model
    model = load_model(augmented_config)

    # Run testing
    for i in range(3):
        result = test_cnn(model, test_loader)
        write_json(result, os.path.join(augmented_config["path"], f"result{i}.json"))

if __name__ == "__main__":
    main()