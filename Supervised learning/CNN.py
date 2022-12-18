# Libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms

# Check the GPU availability
use_gpu = torch.cuda.is_available()

class CNN(nn):
    def __init__(self, K):
        super(CNN, self).__init__()
        # Define the convolutional layers
        self.conv = nn.Sequential()
        # Define the linear layers
        self.dense = nn.Sequential()
    def forward(self, X):
        out = self.conv(X)
        out = out.view(-1, ?)
        out = self.dense(out)
        return out
    
