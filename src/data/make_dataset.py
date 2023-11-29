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
    tensor1 = torch.cat((image,magnitude), dim = 0)
    tensor2 = torch.cat((tensor1,phase))

    return tensor2

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
def import_data_multi():
#     transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std= [0.229, 0.224, 0.225]),
#             concat_fft(),    
#         ])

#     test_dataset0 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/test/ADM', transform = transform)
#     test_dataset1 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/test/DDPM', transform = transform)
#     test_dataset2 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/test/Diff-ProjectedGAN', transform = transform)
#     test_dataset3 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/test/Diff-StyleGAN2', transform = transform)
#     test_dataset4 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/test/IDDPM', transform = transform)
#     test_dataset5 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/test/LDM', transform = transform)
#     test_dataset6 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/test/PNDM', transform = transform)
#     test_dataset7 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/test/ProGAN', transform = transform)
#     test_dataset8 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/test/ProjectedGAN', transform = transform)
#     test_dataset9 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/test/StyleGAN', transform = transform)

#     test_loader0 = DataLoader(dataset=test_dataset0, batch_size=16, shuffle=False, num_workers=6)
#     test_loader1 = DataLoader(dataset=test_dataset1, batch_size=16, shuffle=False, num_workers=6)
#     test_loader2 = DataLoader(dataset=test_dataset2, batch_size=16, shuffle=False, num_workers=6)
#     test_loader3 = DataLoader(dataset=test_dataset3, batch_size=16, shuffle=False, num_workers=6)
#     test_loader4 = DataLoader(dataset=test_dataset4, batch_size=16, shuffle=False, num_workers=6)
#     test_loader5 = DataLoader(dataset=test_dataset5, batch_size=16, shuffle=False, num_workers=6)
#     test_loader6 = DataLoader(dataset=test_dataset6, batch_size=16, shuffle=False, num_workers=6)
#     test_loader7 = DataLoader(dataset=test_dataset7, batch_size=16, shuffle=False, num_workers=6)
#     test_loader8 = DataLoader(dataset=test_dataset8, batch_size=16, shuffle=False, num_workers=6)
#     test_loader9 = DataLoader(dataset=test_dataset9, batch_size=16, shuffle=False, num_workers=6)

#     test_loader = [test_loader0, test_loader1, test_loader2, test_loader3, test_loader4, test_loader5, test_loader6, test_loader7, test_loader8, test_loader9]

#     return test_loader

# #Custom Dataset for Pytorch Dataloader
# class customDataset(Dataset):
#         def __init__(self, data, offset):
#             self.data = data
#             self.label_offset = offset

#         def __len__(self):
#             return len(self.data)

#         def __getitem__(self, idx):
#             data, label = self.data[idx]
#             return data, label + self.label_offset
        
# #Multiclass Classifier Training Dataset
# def import_train_multi():
#     transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std= [0.229, 0.224, 0.225]),
#             concat_fft(),    
#         ])

#     train_dataset0 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/train/ADM/1_fake', transform = transform)
#     train_dataset1 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/train/DDPM/1_fake', transform = transform)
#     train_dataset2 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/train/Diff-ProjectedGAN/1_fake', transform = transform)
#     train_dataset3 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/train/Diff-StyleGAN2/1_fake', transform = transform)
#     train_dataset4 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/train/IDDPM/1_fake', transform = transform)
#     train_dataset5 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/train/LDM/1_fake', transform = transform)
#     train_dataset6 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/train/PNDM/1_fake', transform = transform)
#     train_dataset7 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/train/ProGAN/1_fake', transform = transform)
#     train_dataset8 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/train/ProjectedGAN/1_fake', transform = transform)
#     train_dataset9 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/train/StyleGAN/1_fake', transform = transform)
#     train_dataset10 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/train/ADM/0_real', transform = transform)
#     train_dataset11 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/train/DDPM/0_real', transform = transform)
#     train_dataset12 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/train/Diff-ProjectedGAN/0_real', transform = transform)
#     train_dataset13 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/train/Diff-StyleGAN2/0_real', transform = transform)
#     train_dataset14 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/train/IDDPM/0_real', transform = transform)
#     train_dataset15 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/train/LDM/0_real', transform = transform)
#     train_dataset16 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/train/PNDM/0_real', transform = transform)
#     train_dataset17 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/train/ProGAN/0_real', transform = transform)
#     train_dataset18 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/train/ProjectedGAN/0_real', transform = transform)
#     train_dataset19 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/train/StyleGAN/0_real', transform = transform)

#     val_dataset0 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/val/ADM/1_fake', transform = transform)
#     val_dataset1 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/val/DDPM/1_fake', transform = transform)
#     val_dataset2 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/val/Diff-ProjectedGAN/1_fake', transform = transform)
#     val_dataset3 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/val/Diff-StyleGAN2/1_fake', transform = transform)
#     val_dataset4 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/val/IDDPM/1_fake', transform = transform)
#     val_dataset5 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/val/LDM/1_fake', transform = transform)
#     val_dataset6 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/val/PNDM/1_fake', transform = transform)
#     val_dataset7 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/val/ProGAN/1_fake', transform = transform)
#     val_dataset8 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/val/ProjectedGAN/1_fake', transform = transform)
#     val_dataset9 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/val/StyleGAN/1_fake', transform = transform)
#     val_dataset10 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/val/ADM/0_real', transform = transform)
#     val_dataset11 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/val/DDPM/0_real', transform = transform)
#     val_dataset12 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/val/Diff-ProjectedGAN/0_real', transform = transform)
#     val_dataset13 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/val/Diff-StyleGAN2/0_real', transform = transform)
#     val_dataset14 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/val/IDDPM/0_real', transform = transform)
#     val_dataset15 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/val/LDM/0_real', transform = transform)
#     val_dataset16 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/val/PNDM/0_real', transform = transform)
#     val_dataset17 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/val/ProGAN/0_real', transform = transform)
#     val_dataset18 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/val/ProjectedGAN/0_real', transform = transform)
#     val_dataset19 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/val/StyleGAN/0_real', transform = transform)

#     test_dataset0 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/test/ADM/1_fake', transform = transform)
#     test_dataset1 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/test/DDPM/1_fake', transform = transform)
#     test_dataset2 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/test/Diff-ProjectedGAN/1_fake', transform = transform)
#     test_dataset3 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/test/Diff-StyleGAN2/1_fake', transform = transform)
#     test_dataset4 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/test/IDDPM/1_fake', transform = transform)
#     test_dataset5 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/test/LDM/1_fake', transform = transform)
#     test_dataset6 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/test/PNDM/1_fake', transform = transform)
#     test_dataset7 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/test/ProGAN/1_fake', transform = transform)
#     test_dataset8 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/test/ProjectedGAN/1_fake', transform = transform)
#     test_dataset9 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/test/StyleGAN/1_fake', transform = transform)
#     test_dataset10 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/test/ADM/0_real', transform = transform)
#     test_dataset11 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/test/DDPM/0_real', transform = transform)
#     test_dataset12 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/test/Diff-ProjectedGAN/0_real', transform = transform)
#     test_dataset13 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/test/Diff-StyleGAN2/0_real', transform = transform)
#     test_dataset14 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/test/IDDPM/0_real', transform = transform)
#     test_dataset15 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/test/LDM/0_real', transform = transform)
#     test_dataset16 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/test/PNDM/0_real', transform = transform)
#     test_dataset17 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/test/ProGAN/0_real', transform = transform)
#     test_dataset18 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/test/ProjectedGAN/0_real', transform = transform)
#     test_dataset19 = ImageFolder(root='../../../../../../shared/rsaas/common/diffusion_model_deepfakes_lsun_bedrooms/diffusion_model_deepfakes_lsun_bedroom/test/StyleGAN/0_real', transform = transform)

#     train_dataset0 = customDataset(train_dataset0, 0)
#     train_dataset1 = customDataset(train_dataset1, 1)
#     train_dataset2 = customDataset(train_dataset2, 2)
#     train_dataset3 = customDataset(train_dataset3, 3)
#     train_dataset4 = customDataset(train_dataset4, 4)
#     train_dataset5 = customDataset(train_dataset5, 5)
#     train_dataset6 = customDataset(train_dataset6, 6)
#     train_dataset7 = customDataset(train_dataset7, 7)
#     train_dataset8 = customDataset(train_dataset8, 8)
#     train_dataset9 = customDataset(train_dataset9, 9)
#     train_dataset10 = customDataset(train_dataset10, 10)
#     train_dataset11 = customDataset(train_dataset11, 10)
#     train_dataset12 = customDataset(train_dataset12, 10)
#     train_dataset13 = customDataset(train_dataset13, 10)
#     train_dataset14 = customDataset(train_dataset14, 10)
#     train_dataset15 = customDataset(train_dataset15, 10)
#     train_dataset16 = customDataset(train_dataset16, 10)
#     train_dataset17 = customDataset(train_dataset17, 10)
#     train_dataset18 = customDataset(train_dataset18, 10)
#     train_dataset19 = customDataset(train_dataset19, 10)

#     val_dataset0 = customDataset(val_dataset0, 0)
#     val_dataset1 = customDataset(val_dataset1, 1)
#     val_dataset2 = customDataset(val_dataset2, 2)
#     val_dataset3 = customDataset(val_dataset3, 3)
#     val_dataset4 = customDataset(val_dataset4, 4)
#     val_dataset5 = customDataset(val_dataset5, 5)
#     val_dataset6 = customDataset(val_dataset6, 6)
#     val_dataset7 = customDataset(val_dataset7, 7)
#     val_dataset8 = customDataset(val_dataset8, 8)
#     val_dataset9 = customDataset(val_dataset9, 9)
#     val_dataset10 = customDataset(val_dataset10, 10)
#     val_dataset11 = customDataset(val_dataset11, 10)
#     val_dataset12 = customDataset(val_dataset12, 10)
#     val_dataset13 = customDataset(val_dataset13, 10)
#     val_dataset14 = customDataset(val_dataset14, 10)
#     val_dataset15 = customDataset(val_dataset15, 10)
#     val_dataset16 = customDataset(val_dataset16, 10)
#     val_dataset17 = customDataset(val_dataset17, 10)
#     val_dataset18 = customDataset(val_dataset18, 10)
#     val_dataset19 = customDataset(val_dataset19, 10)

#     test_dataset0 = customDataset(test_dataset0, 0)
#     test_dataset1 = customDataset(test_dataset1, 1)
#     test_dataset2 = customDataset(test_dataset2, 2)
#     test_dataset3 = customDataset(test_dataset3, 3)
#     test_dataset4 = customDataset(test_dataset4, 4)
#     test_dataset5 = customDataset(test_dataset5, 5)
#     test_dataset6 = customDataset(test_dataset6, 6)
#     test_dataset7 = customDataset(test_dataset7, 7)
#     test_dataset8 = customDataset(test_dataset8, 8)
#     test_dataset9 = customDataset(test_dataset9, 9)
#     test_dataset10 = customDataset(test_dataset10, 10)
#     test_dataset11 = customDataset(test_dataset11, 10)
#     test_dataset12 = customDataset(test_dataset12, 10)
#     test_dataset13 = customDataset(test_dataset13, 10)
#     test_dataset14 = customDataset(test_dataset14, 10)
#     test_dataset15 = customDataset(test_dataset15, 10)
#     test_dataset16 = customDataset(test_dataset16, 10)
#     test_dataset17 = customDataset(test_dataset17, 10)
#     test_dataset18 = customDataset(test_dataset18, 10)
#     test_dataset19 = customDataset(test_dataset19, 10)

#     train_dataset = ConcatDataset([train_dataset0, train_dataset1, train_dataset2, train_dataset3, train_dataset4, train_dataset5, train_dataset6, train_dataset7, train_dataset8, train_dataset9, train_dataset10, train_dataset11, train_dataset12, train_dataset13, train_dataset14, train_dataset15, train_dataset16, train_dataset17,train_dataset18, train_dataset19])
#     val_dataset = ConcatDataset([val_dataset0, val_dataset1, val_dataset2, val_dataset3, val_dataset4, val_dataset5, val_dataset6, val_dataset7, val_dataset8, val_dataset9, val_dataset10, val_dataset11, val_dataset12, val_dataset13, val_dataset14, val_dataset15, val_dataset16, val_dataset17, val_dataset18, val_dataset19])
#     test_dataset = ConcatDataset([test_dataset0, test_dataset1, test_dataset2, test_dataset3, test_dataset4, test_dataset5, test_dataset6, test_dataset7, test_dataset8, test_dataset9, test_dataset10, test_dataset11, test_dataset12, test_dataset13, test_dataset14, test_dataset15, test_dataset16, test_dataset17, test_dataset18, test_dataset19])

#     train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, num_workers=6)
#     val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=True, num_workers=6)
#     test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=True, num_workers=6)

#     return train_loader, val_loader, test_loader

    return 1, 2, 3