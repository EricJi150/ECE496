import glob
import random
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
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    # modify get_item based on transform and cv2 = double check perspective_fields architecture
    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = self.transform(Image.open(image_filepath).convert('RGB'))
        label = image_filepath.split("/")[-2]
        # class_to_idx = {'gen':1,'real':0}
        class_to_idx = {'gen':0,'real':1}
        label = class_to_idx[label]
        return image_filepath, image, label
    
#Binary Classification for indoor data from Ayush's dataset
def import_data():
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std= [0.229, 0.224, 0.225]),
            concat_fft(),    
        ])
    
    bdd_train_path = "/data/amitabh3/bdd_prequalified/train"
    bdd_val_path = "/data/amitabh3/bdd_prequalified/val"
    bdd_test_path = "/data/amitabh3/bdd_prequalified/test"

    mapv_train_path = "/data/amitabh3/mapv_prequalified/train"
    mapv_val_path = "/data/amitabh3/mapv_prequalified/val"
    mapv_test_path = "/data/amitabh3/mapv_prequalified/test"

    bedroom_path = '/data/amitabh3/bedroom_193k_prequalified/test'
    diningroom_path = '/data/amitabh3/dining_room_prequalified/test'
    kitchen_path = '/data/amitabh3/kitchen_prequalified/test'
    livingroom_path = '/data/amitabh3/living_187k_prequalified/test'

    dalle_path = '/data/amitabh3/dalle_unconfident_gen_images/test'
    deepfloyd_indoor_path = '/data/amitabh3/deepfloyd_unconfident_gen_images/test'
    firefly_indoor_path = '/data/amitabh3/firefly_unconfident_gen_images_indoor/test'
    kadinsky_indoor_path = '/data/amitabh3/kadinsky_unconfident_gen_images/test'

    deepfloyd_outdoor_path = '/data/amitabh3/deepfloyd_unconfident_gen_images_outdoor/test'
    firefly_outdoor_path = '/data/amitabh3/firefly_unconfident_gen_images_outdoor/test'
    kadinsky_outdoor_path = '/data/amitabh3/kadinsky_unconfident_gen_images_outdoor/test'


    mode = 'streets'

    bdd_train_image_paths = []
    mapv_train_image_paths = []

    bdd_val_image_paths = []
    mapv_val_image_paths = []

    bdd_test_image_paths = []
    mapv_test_image_paths = []

    bedroom_image_paths = []
    diningroom_image_paths = []
    kitchen_image_paths = []
    livingroom_image_paths = []

    dalle_path_image_paths = []
    deepfloyd_indoor_image_paths = []
    firefly_indoor_image_paths = []
    kadinsky_indoor_image_paths = []

    deepfloyd_outdoor_image_paths = []
    firefly_outdoor_image_paths = []
    kadinsky_outdoor_image_paths = []

    classes = []


    for data_path in glob.glob(bdd_train_path + '/*'):
        classes.append(data_path.split('/')[-1])
        if mode == 'streets':
            bdd_train_image_paths.append(glob.glob(data_path + '/*'))


    for data_path in glob.glob(mapv_train_path + '/*'):
        if mode == 'streets':
            mapv_train_image_paths.append(glob.glob(data_path + '/*'))

            
    train_image_paths = bdd_train_image_paths + mapv_train_image_paths
    train_image_paths = list(flatten(train_image_paths))
    random.shuffle(train_image_paths)

    for data_path in glob.glob(bdd_val_path + '/*'):
        classes.append(data_path.split('/')[-1])
        if mode == 'streets':
            bdd_val_image_paths.append(glob.glob(data_path + '/*'))


    for data_path in glob.glob(mapv_val_path + '/*'):
        if mode == 'streets':
            mapv_val_image_paths.append(glob.glob(data_path + '/*'))

    val_image_paths = bdd_val_image_paths + mapv_val_image_paths
    val_image_paths = list(flatten(val_image_paths))
    random.shuffle(val_image_paths)


    for data_path in glob.glob(firefly_indoor_path + '/*'):
        classes.append(data_path.split('/')[-1])
        if mode == 'streets':
            firefly_indoor_image_paths.append(glob.glob(data_path + '/*'))

    # for data_path in glob.glob(mapv_test_path + '/*'):
    #     if mode == 'streets':
    #         mapv_test_image_paths.append(glob.glob(data_path + '/*'))
    
    # for data_path in glob.glob(kitchen_path + '/*'):
    #     classes.append(data_path.split('/')[-1])
    #     if mode == 'streets':
    #         kitchen_image_paths.append(glob.glob(data_path + '/*'))


    # for data_path in glob.glob(livingroom_path + '/*'):
    #     if mode == 'streets':
    #         livingroom_image_paths.append(glob.glob(data_path + '/*'))

    test_image_paths = firefly_indoor_image_paths
    test_image_paths = list(flatten(test_image_paths))


    print("Train size: {}\nValid size: {}\nOriginal Test Size : {}".format(len(train_image_paths), len(val_image_paths), len(test_image_paths)))

   
    test_dataset = DatasetWithFilepaths(test_image_paths, transform=transform)

    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, num_workers=6)
    
    return test_loader
