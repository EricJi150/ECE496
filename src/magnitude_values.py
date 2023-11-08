import numpy as np
import wandb
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

wandb.login(key="76c1f7f13f849593c4dc0d5de21f718b76155fea")
wandb.init(project='2D-FACT-Values')

class concat_fft:
  def __call__(self, image):
    grayimage = transforms.Grayscale(num_output_channels=1)(image)
    fft = np.fft.fftshift(np.fft.fft2(grayimage.numpy()))
    magnitude = torch.from_numpy(np.abs(fft)).float()
    tensor = torch.cat((image,magnitude), dim = 0)
    return tensor
  
def main():
    transform = transforms.Compose([
            transforms.ToTensor(),
            concat_fft(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406, 0],
                                 std= [0.229, 0.224, 0.225, 1]),
        ])

    train_dataset = ImageFolder(root='../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/train/LDM', transform = transform)

    channel_sums = torch.zeros(4)
    channel_mins = torch.ones(4) * float('inf')
    channel_maxs = torch.ones(4) * float('-inf')

    num_pixels = 0

    for train_X, train_Y in train_dataset:

        num_pixels += train_X.size(1) * train_X.size(2)

        for channel in range(4):
            data = train_X[channel, :, :]
            channel_sums[channel] += data.sum().item()
            channel_mins[channel] = min(channel_mins[channel], data.min().item())
            channel_maxs[channel] = max(channel_maxs[channel], data.max().item())

    for channel in range(4):
        wandb.log({'Epoch': channel, 'Average': channel_sums[channel]/num_pixels})
        wandb.log({'Epoch': channel, 'Minimum': channel_mins[channel]})
        wandb.log({'Epoch': channel, 'Maximum': channel_maxs[channel]})
        
    return

if __name__ == "__main__":
    main()