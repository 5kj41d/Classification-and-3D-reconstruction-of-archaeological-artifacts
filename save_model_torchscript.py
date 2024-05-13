import torch
import os
import glob
import torch.nn as nn 


"""
Author: Magnus 
Version 1.0
Goal: Generic modulation is next step -> Can load and handle an arbritrary model and path is next goal
"""

# Add the checkpoint path and filename on the model you want to load in and save as a Torchscript for deployment
source_checkpoint_dir = "./checkpoints" 
source_file_name = "gan_checkpoint.pt"
target_filename = "gan_1st_script" 
target_dir = "./torch_script_db" 

n_channels = 3 

class G(nn.Module):
    def __init__(
        self,
        nz, 
        ngf
    ): 
        super(G, self).__init__()

        latent_size = nz
        ngf = ngf
        
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(latent_size, ngf * 8, kernel_size = 4, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace = True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace = True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace = True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace = True),
            # state size. ``(nc) x 64 x 64``
            # Adding one more layer to upscale from 64x64 to 128x128
            nn.ConvTranspose2d(ngf, n_channels, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.Tanh()
            # state size. ``(nc) x 128 x 128``
        )        

    def forward(self, x):
        x = self.main(x) # Output size = 64x64 
        return x 
    
    
# Setup model and optimizer
generator = G(ngf = 32, nz = 100)


def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    generator_state_dict = checkpoint['generator_state_dict']
    
    # Remove "module." prefix from keys if present - They are created due to Data Parallel class
    generator_state_dict = {k.replace('module.', ''): v for k, v in generator_state_dict.items()}

    return generator_state_dict


def find_latest_checkpoint():
    checkpoints = glob.glob(os.path.join(source_checkpoint_dir, source_file_name))
    if checkpoints: 
        # Use the OS metadata to recognize the latest file changed/added
        latest_checkpoint = max(checkpoints, key = os.path.getctime)
        return latest_checkpoint
    else: 
        return None
    
    

# Load and save model
latest_checkpoint = find_latest_checkpoint() 
if latest_checkpoint: 
    print("Checkpoint was found!")
    generator_state_dict_load = load_checkpoint(latest_checkpoint)
    generator.load_state_dict(generator_state_dict_load)
    
    os.makedirs(target_dir, exist_ok=True)
    target_file_path = os.path.join(target_dir, target_filename)
    
    model_scripted = torch.jit.script(generator)
    model_scripted.save(target_file_path)
    
    # NOTE: When loading the script remember using the model.eval() 
    
    
    
    