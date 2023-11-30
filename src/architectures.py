import torch
import torchvision
import torch.nn as nn

'''
Modified ResNet-18 to support 4 channel input
'''
class ResNet18_4(nn.Module):
    def __init__(self):
        super(ResNet18_4, self).__init__()
        self.model = torchvision.models.resnet18()
        self.model.conv1 = torch.nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
    
    def forward(self, x):
        x = self.model(x)
        return x
    
'''
Modified ResNet-18 to support 5 channel input
'''
class ResNet18_5(nn.Module):
    def __init__(self):
        super(ResNet18_5, self).__init__()
        self.model = torchvision.models.resnet18()
        self.model.conv1 = torch.nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
    
    def forward(self, x):
        x = self.model(x)
        return x
    
'''
Modified ResNet-18 to support multiclass 4 channel input
'''
class ResNet18_4_Multi(nn.Module):
    def __init__(self):
        super(ResNet18_4_Multi, self).__init__()
        self.model = torchvision.models.resnet18()
        self.model.conv1 = torch.nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 11)
    
    def forward(self, x):
        x = self.model(x)
        return x
    
'''
Modified ResNet-18 to support multiclass 5 channel input
'''
class ResNet18_5_Multi(nn.Module):
    def __init__(self):
        super(ResNet18_5_Multi, self).__init__()
        self.model = torchvision.models.resnet18()
        self.model.conv1 = torch.nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 11)
    
    def forward(self, x):
        x = self.model(x)
        return x