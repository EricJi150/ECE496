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
        class_to_idx = {'gen':0,'real':1}
        label = class_to_idx[label]
        return image_filepath, image, label
    
def import_indoor_data():
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std= [0.229, 0.224, 0.225]),
            concat_fft(),    
        ])

    #indoor dataset
    bedroom_train_path = '/data/amitabh3/bedroom_193k_prequalified/train'
    diningroom_train_path = '/data/amitabh3/dining_room_prequalified/train'
    kitchen_train_path = '/data/amitabh3/kitchen_prequalified/train'
    livingroom_train_path = '/data/amitabh3/living_187k_prequalified/train'

    bedroom_val_path = '/data/amitabh3/bedroom_193k_prequalified/val'
    diningroom_val_path = '/data/amitabh3/dining_room_prequalified/val'
    kitchen_val_path = '/data/amitabh3/kitchen_prequalified/val'
    livingroom_val_path = '/data/amitabh3/living_187k_prequalified/val'

    bedroom_test_path = '/data/amitabh3/bedroom_193k_prequalified/test'
    diningroom_test_path = '/data/amitabh3/dining_room_prequalified/test'
    kitchen_test_path = '/data/amitabh3/kitchen_prequalified/test'
    livingroom_test_path = '/data/amitabh3/living_187k_prequalified/test'

    bedroom_train_image_paths = []
    diningroom_train_image_paths = []
    kitchen_train_image_paths = []
    livingroom_train_image_paths = []

    bedroom_val_image_paths = []
    diningroom_val_image_paths = []
    kitchen_val_image_paths = []
    livingroom_val_image_paths = []

    bedroom_test_image_paths = []
    diningroom_test_image_paths = []
    kitchen_test_image_paths = []
    livingroom_test_image_paths = []

    for data_path in glob.glob(bedroom_train_path + '/*'):
        bedroom_train_image_paths.append(glob.glob(data_path + '/*'))
    for data_path in glob.glob(diningroom_train_path + '/*'):
        diningroom_train_image_paths.append(glob.glob(data_path + '/*'))
    for data_path in glob.glob(kitchen_train_path + '/*'):
        kitchen_train_image_paths.append(glob.glob(data_path + '/*'))
    for data_path in glob.glob(livingroom_train_path + '/*'):
        livingroom_train_image_paths.append(glob.glob(data_path + '/*'))
    train_image_paths = bedroom_train_image_paths + diningroom_train_image_paths + kitchen_train_image_paths + livingroom_train_image_paths
    train_image_paths = list(flatten(train_image_paths))

    for data_path in glob.glob(bedroom_val_path + '/*'):
        bedroom_val_image_paths.append(glob.glob(data_path + '/*'))
    for data_path in glob.glob(diningroom_val_path + '/*'):
        diningroom_val_image_paths.append(glob.glob(data_path + '/*'))
    for data_path in glob.glob(kitchen_val_path + '/*'):
        kitchen_val_image_paths.append(glob.glob(data_path + '/*'))
    for data_path in glob.glob(livingroom_val_path + '/*'):
        livingroom_val_image_paths.append(glob.glob(data_path + '/*'))
    val_image_paths = bedroom_val_image_paths + diningroom_val_image_paths + kitchen_val_image_paths + livingroom_val_image_paths
    val_image_paths = list(flatten(val_image_paths))

    for data_path in glob.glob(bedroom_test_path + '/*'):
        bedroom_test_image_paths.append(glob.glob(data_path + '/*'))
    for data_path in glob.glob(diningroom_test_path + '/*'):
        diningroom_test_image_paths.append(glob.glob(data_path + '/*'))
    for data_path in glob.glob(kitchen_test_path + '/*'):
        kitchen_test_image_paths.append(glob.glob(data_path + '/*'))
    for data_path in glob.glob(livingroom_test_path + '/*'):
        livingroom_test_image_paths.append(glob.glob(data_path + '/*'))
    test_image_paths = bedroom_test_image_paths + diningroom_test_image_paths + kitchen_test_image_paths + livingroom_test_image_paths
    test_image_paths = list(flatten(test_image_paths))

    print("Train size: {}\nValid size: {}\nTest Size : {}".format(len(train_image_paths), len(val_image_paths), len(test_image_paths)))

    train_dataset = DatasetWithFilepaths(train_image_paths, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=6)
    val_dataset = DatasetWithFilepaths(val_image_paths, transform=transform)
    val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False, num_workers=6)
    test_dataset = DatasetWithFilepaths(test_image_paths, transform=transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, num_workers=6)
    
    return train_loader, val_loader, test_loader

def import_outdoor_data():
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std= [0.229, 0.224, 0.225]),
            concat_fft(),    
        ])

    #outdoor dataset
    bdd_train_path = "/data/amitabh3/bdd_prequalified/train"
    bdd_val_path = "/data/amitabh3/bdd_prequalified/val"
    bdd_test_path = "/data/amitabh3/bdd_prequalified/test"

    mapv_train_path = "/data/amitabh3/mapv_prequalified/train"
    mapv_val_path = "/data/amitabh3/mapv_prequalified/val"
    mapv_test_path = "/data/amitabh3/mapv_prequalified/test"

    bdd_train_image_paths = []
    mapv_train_image_paths = []

    bdd_val_image_paths = []
    mapv_val_image_paths = []

    bdd_test_image_paths = []
    mapv_test_image_paths = []

    for data_path in glob.glob(bdd_train_path + '/*'):
        bdd_train_image_paths.append(glob.glob(data_path + '/*'))
    for data_path in glob.glob(mapv_train_path + '/*'):
       mapv_train_image_paths.append(glob.glob(data_path + '/*'))
    
    train_image_paths = bdd_train_image_paths + mapv_train_image_paths
    train_image_paths = list(flatten(train_image_paths))

    for data_path in glob.glob(bdd_val_path + '/*'):
        bdd_val_image_paths.append(glob.glob(data_path + '/*'))
    for data_path in glob.glob(mapv_val_path + '/*'):
        mapv_val_image_paths.append(glob.glob(data_path + '/*'))

    val_image_paths = bdd_val_image_paths + mapv_val_image_paths
    val_image_paths = list(flatten(val_image_paths))

    for data_path in glob.glob(bdd_test_path + '/*'):
        bdd_test_image_paths.append(glob.glob(data_path + '/*'))
    for data_path in glob.glob(mapv_test_path + '/*'):
        mapv_test_image_paths.append(glob.glob(data_path + '/*'))

    test_image_paths = bdd_test_image_paths + mapv_test_image_paths
    test_image_paths = list(flatten(test_image_paths))

    print("Train size: {}\nValid size: {}\nTest Size : {}".format(len(train_image_paths), len(val_image_paths), len(test_image_paths)))
    
    train_dataset = DatasetWithFilepaths(train_image_paths, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=6)
    val_dataset = DatasetWithFilepaths(val_image_paths, transform=transform)
    val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False, num_workers=6)
    test_dataset = DatasetWithFilepaths(test_image_paths, transform=transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, num_workers=6)
    
    return train_loader, val_loader, test_loader

def import_test_data():
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std= [0.229, 0.224, 0.225]),
            concat_fft(),    
        ])

    #test dataset
    dalle_path = '/data/amitabh3/dalle_unconfident_gen_images/test'
    deepfloyd_indoor_path = '/data/amitabh3/deepfloyd_unconfident_gen_images/test'
    firefly_indoor_path = '/data/amitabh3/firefly_unconfident_gen_images_indoor/test'
    kadinsky_indoor_path = '/data/amitabh3/kadinsky_unconfident_gen_images/test'

    deepfloyd_outdoor_path = '/data/amitabh3/deepfloyd_unconfident_gen_images_outdoor/test'
    firefly_outdoor_path = '/data/amitabh3/firefly_unconfident_gen_images_outdoor/test'
    kadinsky_outdoor_path = '/data/amitabh3/kadinsky_unconfident_gen_images_outdoor/test'

    #set the path
    test_path = kadinsky_outdoor_path

    test_image_paths = []

    for data_path in glob.glob(test_path + '/*'):
        test_image_paths.append(glob.glob(data_path + '/*'))
    test_image_paths = list(flatten(test_image_paths))

    print("Test Size : {}".format(len(test_image_paths)))

    test_dataset = DatasetWithFilepaths(test_image_paths, transform=transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, num_workers=6)

    return test_loader

def import_kandinsky_indoor_large_data():
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std= [0.229, 0.224, 0.225]),
            concat_fft(),    
        ])

    #test dataset
    train_path = '/data/amitabh3/kandinsky_indoor_large/test'
    val_path = '/data/amitabh3/kandinsky_indoor_large/test'
    test_path = '/data/amitabh3/kandinsky_indoor_large/test'

    train_image_paths = []
    val_image_paths = []
    test_image_paths = []

    for data_path in glob.glob(train_path + '/*'):
        train_image_paths.append(glob.glob(data_path + '/*'))
    train_image_paths = list(flatten(train_image_paths))

    for data_path in glob.glob(val_path + '/*'):
        val_image_paths.append(glob.glob(data_path + '/*'))
    val_image_paths = list(flatten(val_image_paths))

    for data_path in glob.glob(test_path + '/*'):
        test_image_paths.append(glob.glob(data_path + '/*'))
    test_image_paths = list(flatten(test_image_paths))

    print("Train Size : {}".format(len(train_image_paths)))
    print("Val Size : {}".format(len(val_image_paths)))
    print("Test Size : {}".format(len(test_image_paths)))

    train_dataset = DatasetWithFilepaths(train_image_paths, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=False, num_workers=6)

    val_dataset = DatasetWithFilepaths(val_image_paths, transform=transform)
    val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False, num_workers=6)

    test_dataset = DatasetWithFilepaths(test_image_paths, transform=transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, num_workers=6)

    return train_loader, val_loader, test_loader

def import_deepfloyd_indoor_large_data():
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std= [0.229, 0.224, 0.225]),
            concat_fft(),    
        ])

    #test dataset
    train_path = '/data/amitabh3/deepfloyd_indoor_large/test'
    val_path = '/data/amitabh3/deepfloyd_indoor_large/test'
    test_path = '/data/amitabh3/deepfloyd_indoor_large/test'

    train_image_paths = []
    val_image_paths = []
    test_image_paths = []

    for data_path in glob.glob(train_path + '/*'):
        train_image_paths.append(glob.glob(data_path + '/*'))
    train_image_paths = list(flatten(train_image_paths))

    for data_path in glob.glob(val_path + '/*'):
        val_image_paths.append(glob.glob(data_path + '/*'))
    val_image_paths = list(flatten(val_image_paths))

    for data_path in glob.glob(test_path + '/*'):
        test_image_paths.append(glob.glob(data_path + '/*'))
    test_image_paths = list(flatten(test_image_paths))

    print("Train Size : {}".format(len(train_image_paths)))
    print("Val Size : {}".format(len(val_image_paths)))
    print("Test Size : {}".format(len(test_image_paths)))

    train_dataset = DatasetWithFilepaths(train_image_paths, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=False, num_workers=6)

    val_dataset = DatasetWithFilepaths(val_image_paths, transform=transform)
    val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False, num_workers=6)

    test_dataset = DatasetWithFilepaths(test_image_paths, transform=transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, num_workers=6)

    return train_loader, val_loader, test_loader

