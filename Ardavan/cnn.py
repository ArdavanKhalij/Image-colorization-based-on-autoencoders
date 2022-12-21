############################################################################################
# Libraries
import kornia as kornia
import numpy as np
import torch
import torchvision.datasets
from torchvision import datasets, transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# For plotting
import numpy as np
import matplotlib.pyplot as plt
# For conversion
from skimage.color import lab2rgb, rgb2lab, rgb2gray
from skimage import io
# For everything
import torch
import torch.nn as nn
import torch.nn.functional as F
# For our model
import torchvision.models as models
from torchvision import datasets, transforms
# For utilities
import os, shutil, time
from torch.utils.data import DataLoader

############################################################################################
# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_path = 'C:/Users/xhuli/OneDrive/Desktop/Education/VUB/3rd Semester/Deep Learning/Project/images/'
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # can experiment
    transforms.ToTensor()  # divide the pixel by 255
])

############################################################################################
# Loading the data
train_dataset = torchvision.datasets.ImageFolder(root=train_path, transform=train_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=70, shuffle=True)

############################################################################################
# Convert RDG to LAB space
# assign the L channel to the x vector
x = []
y = []
for idx, (data, target) in enumerate(train_loader):
    for i in range(70):
        lab = kornia.color.rgb_to_lab(data[i])
        x.append(lab[:, :, 0])
        y.append(lab[:, :, 1:] / 128)
    break