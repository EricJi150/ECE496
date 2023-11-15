import wandb
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder

'''
Finding range of values between all channels after transforms
'''
class concat_fft:
  def __call__(self, image):
    # grayimage = transforms.Grayscale(num_output_channels=1)(image)
    # fft = np.fft.fftshift(np.fft.fft2(grayimage.numpy()))
    # magnitude = torch.from_numpy(np.abs(fft)).float()
    # tensor = torch.cat((image,magnitude), dim = 0)
    # return tensor

    grayimage = transforms.Grayscale(num_output_channels=1)(image)
    fft = np.fft.fftshift(np.fft.fft2(grayimage.numpy()))
    magnitude = torch.from_numpy(np.abs(fft)).float()
    phase = torch.from_numpy(np.angle(fft)).float()
    tensor1 = torch.cat((image,magnitude), dim = 0)
    tensor2 = torch.cat((tensor1,phase))

    return tensor2
  
def main():
    wandb.login(key="76c1f7f13f849593c4dc0d5de21f718b76155fea")
    wandb.init(project='2D-FACT-Values')

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std= [0.229, 0.224, 0.225]),
            concat_fft(),
        ])

    train_dataset = ImageFolder(root='../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/train/LDM', transform = transform)

    channel_sums = np.zeros(5)
    channel_mins = np.ones(5) * float('inf')
    channel_maxs = np.ones(5) * float('-inf')

    count = 0
    num_pixels = 0

    for train_X, train_Y in train_dataset:
        if count == 100:
            break
        count += 1
        
        num_pixels += train_X.size(1) * train_X.size(2)

        for channel in range(5):
            data = train_X[channel, :, :]
            channel_sums[channel] += data.sum().item()
            channel_mins[channel] = min(channel_mins[channel], data.min().item())
            channel_maxs[channel] = max(channel_maxs[channel], data.max().item())
    
    
    channels=["Red", "Green", "Blue", "Magnitude", "Phase"]
    table = wandb.Table(columns=["Channel", "Min", "Max", "Avg"])
    
    for i, channel in enumerate(channels):
        table.add_data(channel, channel_mins[i], channel_maxs[i], channel_sums[i]/num_pixels)

    wandb.log({"Channel Statistics": table})
    return


'''
Finding range of values before any transform
'''
def min_max():
    # Load an image
    image_path = '../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/train/LDM/1_fake/sample_038999.png'
    image = Image.open(image_path)
    image_data = np.array(image)

    min_val = np.min(image_data)
    max_val = np.max(image_data)


    print(f"Minimum pixel value: {min_val}")
    print(f"Maximum pixel value: {max_val}")
    return


'''
Finding range of values after grayscaling before taking FFT
'''
class grayscale:
  def __call__(self, image):
    grayimage = transforms.Grayscale(num_output_channels=1)(image)
    return grayimage

def fft_range():
    transform = transforms.Compose([
            transforms.ToTensor(),
            grayscale()
        ])

    image_path = '../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/train/LDM/1_fake/sample_038999.png'
    image = Image.open(image_path)
    tensor = transform(image)
    image_data = np.array(tensor)

    min_val = np.min(image_data)
    max_val = np.max(image_data)

    print(f"Minimum pixel value: {min_val}")
    print(f"Maximum pixel value: {max_val}")
    return

if __name__ == "__main__":
    fft_range()
