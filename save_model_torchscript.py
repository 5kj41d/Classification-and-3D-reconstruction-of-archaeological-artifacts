import torch
import os
import glob
import torch.nn as nn

"""
Author: Magnus 
Version 1.1
Goal: Generic modulation is next step -> Can load and handle an arbitrary model and path is next goal
"""

# Add the checkpoint path and filename of the model you want to load in and save as a TorchScript for deployment
source_checkpoint_dir = "./checkpoints"
source_file_name = "dcgan_medieval_checkpoint_TRUE.pt"
target_filename = "dcgan_medieval_script.pt"
target_dir = "./torch_script_db"

n_channels = 3

class G(nn.Module):
    def __init__(self, nz, ngf):
        super(G, self).__init__()

        latent_size = nz
        ngf = ngf

        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_size, ngf * 8, kernel_size = 4, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(ngf, n_channels, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.main(x)
        return x

# Setup model and optimizer - NOTE: Make sure the parameters fit the original model
generator = G(nz = 100, ngf = 35)

def load_checkpoint(checkpoint_path, g_model):
    checkpoint = torch.load(checkpoint_path)

    if isinstance(g_model, torch.nn.DataParallel):
        g_model.module.load_state_dict(checkpoint['generator_state_dict'])
    else:
        g_model.load_state_dict(checkpoint['generator_state_dict'])

    return g_model

def find_latest_checkpoint():
    checkpoints = glob.glob(os.path.join(source_checkpoint_dir, source_file_name))
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        return latest_checkpoint
    else:
        return None

# Load and save model
latest_checkpoint = find_latest_checkpoint()
if latest_checkpoint:
    print("Checkpoint was found!")
    generator = load_checkpoint(latest_checkpoint, generator)

    os.makedirs(target_dir, exist_ok=True)
    target_file_path = os.path.join(target_dir, target_filename)

    model_scripted = torch.jit.script(generator)
    model_scripted.save(target_file_path)
    
    print(f"Model saved to {target_file_path}")

    # NOTE: When loading the script remember to use model.eval()
else:
    print("No checkpoint found.")
