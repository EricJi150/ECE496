import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset, ConcatDataset

#Concatanation Transformation
class concat_fft:
  def __call__(self, image):
    grayimage = transforms.Grayscale(num_output_channels=1)(image)
    fft = np.fft.fftshift(np.fft.fft2(grayimage.numpy()))
    magnitude = np.log(1+torch.from_numpy(np.abs(fft)).float())
    phase = torch.from_numpy(np.angle(fft)).float()/np.pi
    tensor1 = torch.cat((image,magnitude))  #4 channel
    tensor2 = torch.cat((tensor1,phase))    #5 channel
    tensor3 = torch.cat((magnitude, phase)) #2 channel

    return tensor3
#Binary Classifier Training Dataset
def import_data(dataset):
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std= [0.229, 0.224, 0.225]),
            concat_fft(),    
        ])

    train_dataset = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/train/'+dataset, transform = transform)
    val_dataset = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/val/'+dataset, transform = transform)
    test_dataset = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/test/'+dataset, transform = transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, num_workers=6)
    val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False, num_workers=6)
    test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False, num_workers=6)

    return train_loader, val_loader, test_loader

#Binary Classifier Evaluation Dataset
def import_testsets():
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std= [0.229, 0.224, 0.225]),
            concat_fft(),    
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

    test_loader0 = DataLoader(dataset=test_dataset0, batch_size=16, shuffle=False, num_workers=6)
    test_loader1 = DataLoader(dataset=test_dataset1, batch_size=16, shuffle=False, num_workers=6)
    test_loader2 = DataLoader(dataset=test_dataset2, batch_size=16, shuffle=False, num_workers=6)
    test_loader3 = DataLoader(dataset=test_dataset3, batch_size=16, shuffle=False, num_workers=6)
    test_loader4 = DataLoader(dataset=test_dataset4, batch_size=16, shuffle=False, num_workers=6)
    test_loader5 = DataLoader(dataset=test_dataset5, batch_size=16, shuffle=False, num_workers=6)
    test_loader6 = DataLoader(dataset=test_dataset6, batch_size=16, shuffle=False, num_workers=6)
    test_loader7 = DataLoader(dataset=test_dataset7, batch_size=16, shuffle=False, num_workers=6)
    test_loader8 = DataLoader(dataset=test_dataset8, batch_size=16, shuffle=False, num_workers=6)
    test_loader9 = DataLoader(dataset=test_dataset9, batch_size=16, shuffle=False, num_workers=6)

    test_loader = [test_loader0, test_loader1, test_loader2, test_loader3, test_loader4, test_loader5, test_loader6, test_loader7, test_loader8, test_loader9]

    return test_loader

#Custom Dataset for Pytorch Dataloader
class customDataset(Dataset):
        def __init__(self, data, offset):
            self.data = data
            self.label_offset = offset

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            data, label = self.data[idx]
            if label == 0:
                return data, 10
            else:
                return data, self.label_offset
        
#Multiclass Classifier Training Dataset
def import_train_multi():
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std= [0.229, 0.224, 0.225]),
            concat_fft(),    
        ])

    train_dataset0 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/train/ADM', transform = transform)
    train_dataset1 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/train/DDPM', transform = transform)
    train_dataset2 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/train/Diff-ProjectedGAN', transform = transform)
    train_dataset3 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/train/Diff-StyleGAN2', transform = transform)
    train_dataset4 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/train/IDDPM', transform = transform)
    train_dataset5 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/train/LDM', transform = transform)
    train_dataset6 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/train/PNDM', transform = transform)
    train_dataset7 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/train/ProGAN', transform = transform)
    train_dataset8 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/train/ProjectedGAN', transform = transform)
    train_dataset9 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/train/StyleGAN', transform = transform)

    val_dataset0 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/val/ADM', transform = transform)
    val_dataset1 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/val/DDPM', transform = transform)
    val_dataset2 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/val/Diff-ProjectedGAN', transform = transform)
    val_dataset3 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/val/Diff-StyleGAN2', transform = transform)
    val_dataset4 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/val/IDDPM', transform = transform)
    val_dataset5 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/val/LDM', transform = transform)
    val_dataset6 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/val/PNDM', transform = transform)
    val_dataset7 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/val/ProGAN', transform = transform)
    val_dataset8 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/val/ProjectedGAN', transform = transform)
    val_dataset9 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/val/StyleGAN', transform = transform)

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

    train_dataset0 = customDataset(train_dataset0, 0)
    train_dataset1 = customDataset(train_dataset1, 1)
    train_dataset2 = customDataset(train_dataset2, 2)
    train_dataset3 = customDataset(train_dataset3, 3)
    train_dataset4 = customDataset(train_dataset4, 4)
    train_dataset5 = customDataset(train_dataset5, 5)
    train_dataset6 = customDataset(train_dataset6, 6)
    train_dataset7 = customDataset(train_dataset7, 7)
    train_dataset8 = customDataset(train_dataset8, 8)
    train_dataset9 = customDataset(train_dataset9, 9)

    val_dataset0 = customDataset(val_dataset0, 0)
    val_dataset1 = customDataset(val_dataset1, 1)
    val_dataset2 = customDataset(val_dataset2, 2)
    val_dataset3 = customDataset(val_dataset3, 3)
    val_dataset4 = customDataset(val_dataset4, 4)
    val_dataset5 = customDataset(val_dataset5, 5)
    val_dataset6 = customDataset(val_dataset6, 6)
    val_dataset7 = customDataset(val_dataset7, 7)
    val_dataset8 = customDataset(val_dataset8, 8)
    val_dataset9 = customDataset(val_dataset9, 9)

    test_dataset0 = customDataset(test_dataset0, 0)
    test_dataset1 = customDataset(test_dataset1, 1)
    test_dataset2 = customDataset(test_dataset2, 2)
    test_dataset3 = customDataset(test_dataset3, 3)
    test_dataset4 = customDataset(test_dataset4, 4)
    test_dataset5 = customDataset(test_dataset5, 5)
    test_dataset6 = customDataset(test_dataset6, 6)
    test_dataset7 = customDataset(test_dataset7, 7)
    test_dataset8 = customDataset(test_dataset8, 8)
    test_dataset9 = customDataset(test_dataset9, 9)

    train_dataset = ConcatDataset([train_dataset0, train_dataset1, train_dataset2, train_dataset3, train_dataset4, train_dataset5, train_dataset6, train_dataset7, train_dataset8, train_dataset9])
    val_dataset = ConcatDataset([val_dataset0, val_dataset1, val_dataset2, val_dataset3, val_dataset4, val_dataset5, val_dataset6, val_dataset7, val_dataset8, val_dataset9])
    test_dataset = ConcatDataset([test_dataset0, test_dataset1, test_dataset2, test_dataset3, test_dataset4, test_dataset5, test_dataset6, test_dataset7, test_dataset8, test_dataset9])

    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, num_workers=6)
    val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False, num_workers=6)
    test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False, num_workers=6)

    return train_loader, val_loader, test_loader
