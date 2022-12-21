# For plotting
import numpy as np
import matplotlib.pyplot as plt
# For everything
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
# For our model
import torchvision.datasets
import torchvision.models as models
from torchvision import datasets, transforms
# For utilities
import os, shutil, time

# Device Configuration

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_path = 'C:/Users/xhuli/OneDrive/Desktop/Education/VUB/3rd Semester/Deep Learning/Project/images/'

train_transform = transforms.Compose([
    transforms.Resize((256,256)), # can experiment
    transforms.ToTensor() # divide the pixel by 255
])

train_dataset = torchvision.datasets.ImageFolder(root=train_path, transform=train_transform)

def set_device():
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    return torch.device(dev)

def train_nn(model, train_loader, test_loader, criterion, optimizer, n_epochs):


def evaluate_model_on_test_set(model, test_loader):




