import torch
import torchvision
import torch.nn as nn

'''
Modified ResNet-50 to support multiclass 2 channel input
'''
class ResNet50_2(nn.Module):
    def __init__(self):
        super(ResNet50_2, self).__init__()
        self.model = torchvision.models.resnet50(pretrained = False)
        self.model.conv1 = torch.nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
    
    def forward(self, x):
        x = self.model(x)
        return x
    
'''
Modified ResNet-18 to support multiclass 2 channel input
'''
class ResNet18_2(nn.Module):
    def __init__(self):
        super(ResNet18_2, self).__init__()
        self.model = torchvision.models.resnet18(pretrained = False)
        self.model.conv1 = torch.nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
    
    def forward(self, x):
        x = self.model(x)
        return x

'''
Modified ResNet-18 to support 4 channel input
'''
class ResNet18_4(nn.Module):
    def __init__(self):
        super(ResNet18_4, self).__init__()
        self.model = torchvision.models.resnet18(pretrained = False)
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
        self.model = torchvision.models.resnet18(pretrained = False)
        self.model.conv1 = torch.nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
    
    def forward(self, x):
        x = self.model(x)
        return x
    
'''
Modified ResNet-18 to support multiclass 3 channel input
'''
class ResNet18_3_Multi(nn.Module):
    def __init__(self):
        super(ResNet18_3_Multi, self).__init__()
        self.model = torchvision.models.resnet18(pretrained = False)
        self.model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 11)
    
    def forward(self, x):
        x = self.model(x)
        return x
    
'''
Modified ResNet-18 to support multiclass 4 channel input
'''
class ResNet18_4_Multi(nn.Module):
    def __init__(self):
        super(ResNet18_4_Multi, self).__init__()
        self.model = torchvision.models.resnet18(pretrained = False)
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
        self.model = torchvision.models.resnet18(pretrained = False)
        self.model.conv1 = torch.nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 11)
    
    def forward(self, x):
        x = self.model(x)
        return x
    
