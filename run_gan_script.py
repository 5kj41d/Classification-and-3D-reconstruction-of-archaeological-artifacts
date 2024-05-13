import torch
from torchvision.utils import save_image
import os

subfolder= "torch_script_db"
model_file_name = "gan_1st_script" 
model_path = os.path.join(subfolder, model_file_name) 

model_scripted = torch.jit.load(model_path)
model_scripted.eval()

for name, param in model_scripted.named_parameters():
    print(f"Parameter name: {name}, Size: {param.size()}")

output_folder = "./live_generated_images_gan"
os.makedirs(output_folder, exist_ok = True)

num_images = 100 
for i in range(num_images):
    noise = torch.randn(1, 100, 1, 1) 
    generated_image = model_scripted(noise)
    save_image(generated_image, os.path.join(output_folder, f"generated_image_{i+1}.png"))

print(f"{num_images} images generated and saved to {output_folder}.")



