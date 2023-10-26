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

def import_testsets():
    transform = transforms.Compose([
            transforms.ToTensor(),
            concat_fft(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406, 0],
                                 std= [0.229, 0.224, 0.225, 1]),
        ])

    test_dataset0 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/test/ADM', transform = transform)
    test_dataset1 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/test/DDPM', transform = transform)
    test_dataset2 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/test/Diff-ProjectedGAN', transform = transform)
    test_dataset3 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/test/Diff-StyleGAN2', transform = transform)
    test_dataset4 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/test/IDDPM', transform = transform)
    test_dataset5 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/test/LDM', transform = transform)
    test_dataset6 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/test/PNDM', transform = transform)
    test_dataset7 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/test/ProGAN', transform = transform)
    test_dataset8 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/test/ProjectedGAN', transform = transform)
    test_dataset9 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/test/StyleGAN', transform = transform)

    test_loader = np.empty(10)
    test_loader[0] = DataLoader(dataset=test_dataset0, batch_size=16, shuffle=False, num_workers=6)
    test_loader[1] = DataLoader(dataset=test_dataset1, batch_size=16, shuffle=False, num_workers=6)
    test_loader[2] = DataLoader(dataset=test_dataset2, batch_size=16, shuffle=False, num_workers=6)
    test_loader[3] = DataLoader(dataset=test_dataset3 batch_size=16, shuffle=False, num_workers=6)
    test_loader[4] = DataLoader(dataset=test_dataset4, batch_size=16, shuffle=False, num_workers=6)
    test_loader[5] = DataLoader(dataset=test_dataset5, batch_size=16, shuffle=False, num_workers=6)
    test_loader[6] = DataLoader(dataset=test_dataset6, batch_size=16, shuffle=False, num_workers=6)
    test_loader[7] = DataLoader(dataset=test_dataset7, batch_size=16, shuffle=False, num_workers=6)
    test_loader[8] = DataLoader(dataset=test_dataset8, batch_size=16, shuffle=False, num_workers=6)
    test_loader[9] = DataLoader(dataset=test_dataset9, batch_size=16, shuffle=False, num_workers=6)

    return test_loader
