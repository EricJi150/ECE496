import numpy as np
import wandb
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader



class concat_fft:
  def __call__(self, image):
    grayimage = transforms.Grayscale(num_output_channels=1)(image)
    fft = np.fft.fftshift(np.fft.fft2(grayimage.numpy()))
    magnitude = torch.from_numpy(np.abs(fft)).float()
    tensor = torch.cat((image,magnitude), dim = 0)
    return tensor
  
def main():
    wandb.login(key="76c1f7f13f849593c4dc0d5de21f718b76155fea")
    wandb.init(project='2D-FACT-Values')

    transform = transforms.Compose([
            transforms.ToTensor(),
            concat_fft(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406, 0],
                                 std= [0.229, 0.224, 0.225, 1]),
        ])

    train_dataset = ImageFolder(root='../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/train/LDM', transform = transform)

    channel_sums = np.zeros(4)
    channel_mins = np.ones(4) * float('inf')
    channel_maxs = np.ones(4) * float('-inf')

    count = 0
    num_pixels = 0

    for train_X, train_Y in train_dataset:
        if count == 100:
            break
        count += 1
        
        num_pixels += train_X.size(1) * train_X.size(2)

        for channel in range(4):
            data = train_X[channel, :, :]
            channel_sums[channel] += data.sum().item()
            channel_mins[channel] = min(channel_mins[channel], data.min().item())
            channel_maxs[channel] = max(channel_maxs[channel], data.max().item())
    
    
    channels=["Red", "Green", "Blue", "Magnitude"]
    table = wandb.Table(columns=["Channel", "Min", "Max", "Avg"])
    
    for i, channel in enumerate(channels):
        table.add_data(channel, channel_mins[i], channel_maxs[i], channel_sums[i]/num_pixels)

    wandb.log({"Channel Statistics": table})

    return

import numpy as np
from PIL import Image
def min_max():
    print("Checking min and max values of an image")
    # Load an image
    image_path = '../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/train/LDM/1_fake/sample_038999.png'
    image = Image.open(image_path)
    image_data = np.array(image)

    # Get the minimum and maximum values
    min_val = np.min(image_data)
    max_val = np.max(image_data)

    # Output the min and max values
    print(f"Minimum pixel value: {min_val}")
    print(f"Maximum pixel value: {max_val}")
    return

import numpy as np
from PIL import Image

class grayscale:
  def __call__(self, image):
    grayimage = transforms.Grayscale(num_output_channels=1)(image)
    return grayimage

def fft_range():
    print("Checking min and max values of an image before fft after grayscale")

    transform = transforms.Compose([
            transforms.ToTensor(),
            grayscale()
        ])

    # Load an image
    image_path = '../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/train/LDM/1_fake/sample_038999.png'
    image = Image.open(image_path)
    tensor = transform(image)
    image_data = np.array(tensor)

    # Get the minimum and maximum values
    min_val = np.min(image_data)
    max_val = np.max(image_data)

    # Output the min and max values
    print(f"Minimum pixel value: {min_val}")
    print(f"Maximum pixel value: {max_val}")
    return

if __name__ == "__main__":
    fft_range()