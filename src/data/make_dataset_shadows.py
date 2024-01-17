import glob
import numpy as np
import torch
from PIL import Image 
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from pandas.core.common import flatten

#Concatanation Transformation
class concat_fft:
  def __call__(self, image):
    grayimage = transforms.Grayscale(num_output_channels=1)(image)
    fft = np.fft.fftshift(np.fft.fft2(grayimage.numpy()))
    magnitude = np.log(1+torch.from_numpy(np.abs(fft)).float())
    phase = torch.from_numpy(np.angle(fft)).float()/np.pi
    tensor = torch.cat((magnitude, phase)) #2 channel

    return tensor

class DatasetWithFilepaths(Dataset):
    def __init__(self, image_paths, transform = None):
        self.image_paths = image_paths
        self.transform = transforms.Compose(transform)
    
    def __len__(self):
        return len(self.image_paths)
    
    # modify get_item based on transform and cv2 = double check perspective_fields architecture
    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = self.transform(Image.open(image_filepath).convert('RGB'))
        label = image_filepath.split("/")[-2]
        class_to_idx = {'gen':0,'real':1}
        label = class_to_idx[label]
        return image_filepath, image, label
    
#Binary Classification for indoor data from Ayush's dataset
def import_outdoor_data():
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std= [0.229, 0.224, 0.225]),
            concat_fft(),    
        ])
    
    test_image_paths0 = []
    test_image_paths1 = []
    test_image_paths2 = []
    test_image_paths3 = []
    test_image_paths = []

    test_data_path0 = '/data/amitabh3/bdd_prequalified/test/gen'
    test_data_path1 = '/data/amitabh3/mapv_prequalified/test/gen'
    test_data_path2 = '/data/amitabh3/bdd_prequalified/test/real'
    test_data_path3 = '/data/amitabh3/mapv_prequalified/test/real'

    for data_path in glob.glob(test_data_path0 + '/*'):
        test_image_paths0.append(glob.glob(data_path + '/*'))
    test_image_paths0 = list(flatten(test_image_paths0))

    for data_path in glob.glob(test_data_path1 + '/*'):
       test_image_paths1.append(glob.glob(data_path + '/*'))
    test_image_paths1 = list(flatten(test_image_paths1))
    for data_path in glob.glob(test_data_path2 + '/*'):
        test_image_paths2.append(glob.glob(data_path + '/*'))
    test_image_paths2 = list(flatten(test_image_paths2))

    for data_path in glob.glob(test_data_path3 + '/*'):
       test_image_paths3.append(glob.glob(data_path + '/*'))
    test_image_paths3= list(flatten(test_image_paths3))

    test_image_paths = test_image_paths0 + test_data_path1 + test_data_path2 + test_data_path3


    print(len(test_image_paths), "image in the test dataset")

   
    test_dataset = DatasetWithFilepaths(test_image_paths, transform=transform)

    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, num_workers=6)
    
    return test_loader
