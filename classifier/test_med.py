import os
import json
import torch
import torch.nn.functional as F
from torchvision.transforms import v2 
from utils.iron_and_others_dataset import IronDataset, GeneratedDataset, RandomIronDataset
from torch.utils.data import DataLoader
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from models.lenet_dynamic import LeNet5_Dynamic
from torchinfo import summary

def test_cnn(model, test_loader, fold, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []
    y_pred_probs = []

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            prob = torch.nn.functional.softmax(pred, dim=1)
            _, predicted = torch.max(pred.data, 1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_pred_probs.extend(prob[:, 1].cpu().numpy())

    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred, zero_division=1, pos_label=1)
    recall = metrics.recall_score(y_true, y_pred, zero_division=1, pos_label=1)
    f1_score = metrics.f1_score(y_true, y_pred, zero_division=1, pos_label=1)
    roc_auc = metrics.roc_auc_score(y_true, y_pred_probs)

    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels=["coins", "others"])
    disp.plot()
    path = config["path"]
    filename = f"confusion_matrix{fold}.png"
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

def load_model(config, filename):
    model = LeNet5_Dynamic(
    conv_kernel=config["conv_kernel"], 
    conv_stride=config["conv_stride"], 
    pool_kernel=config["pool_kernel"], 
    pool_stride=config["pool_stride"], 
    fc1_size=config["fc1_size"], 
    fc2_size=config["fc2_size"])

    model.load_state_dict(torch.load(os.path.join(config["path"], filename)))

    return model


def main():

    transforms = v2.Compose([
    v2.Resize(size=(64, 64)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    coins = r"C:\Users\Mate\Projects\Classifiers\data\medieval\test_set"
    others_1 = r"C:\Users\Mate\Projects\Classifiers\data\split-others\10kothers"

    dataset = IronDataset(coin_dir=coins, 
                        others_dir=others_1, 
                        transform=transforms)


    # Initialize config for models
    vanilla = {
        "conv_kernel": 5, 
        "conv_stride": 1, 
        "pool_kernel": 3, 
        "pool_stride": 2, 
        "fc1_size": 256, 
        "fc2_size": 64,
        "batch_size": 16, 
        "epochs": 50, 
        "lr": 0.0001, 
        "path": r"C:\Users\Mate\Projects\Classifiers\saved\tests\med_filt"
    }

    current_config = vanilla


    # Create dataloader
    test_loader = DataLoader(dataset, batch_size=16, drop_last=True, shuffle=False, num_workers=0)


    for i in range(1, 6):
        # Load pytorch model
        model = load_model(current_config, f"model_fold_{i}.pt")

        # Run testing
        result = test_cnn(model, test_loader, i, current_config)
        write_json(result, os.path.join(current_config["path"], f"result{i}.json"))
        model_stats = summary(
            model, 
            input_size=(1, 3, 64, 64), 
            col_names=[
                "input_size",
                "output_size",
                "num_params",
                "params_percent",
                "kernel_size",
                "mult_adds",
                "trainable",
            ],
            verbose=0,
            row_settings=["var_names"],
            )

        summary_str = str(model_stats)
        with open(os.path.join(current_config["path"], f"summary{i}.txt"), "w", encoding="utf-8") as f:
            f.write(summary_str)

if __name__ == "__main__":
    main()