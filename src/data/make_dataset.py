import numpy as np
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
  
def import_data():
    transform = transforms.Compose([
            transforms.ToTensor(),
            concat_fft(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406, 0],
                                 std= [0.229, 0.224, 0.225, 1]),
        ])

    train_dataset = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/train/ProGAN', transform = transform)
    val_dataset = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/val/ProGAN', transform = transform)
    test_dataset = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/test/ProGAN', transform = transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, num_workers=6)
    val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False, num_workers=6)
    test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False, num_workers=6)

    return train_loader, val_loader, test_loader
