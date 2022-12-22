import kornia as kornia
import numpy as np
import torch
from skimage import color
import torchvision.transforms as T
import torchvision.datasets
from skimage.io import imshow
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

# Device Configuration

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_path = 'C:/Users/xhuli/OneDrive/Desktop/Education/VUB/3rd Semester/Deep Learning/Project/images/'

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # can experiment
    transforms.ToTensor(),  # divide the pixel by 255
])
# Loading the data
train_dataset = torchvision.datasets.ImageFolder(root=train_path, transform=train_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=72, shuffle=True)

# Convert RDG to LAB space
# assign the L channel to the x vector
x = []  # array of tensors of  L
y = []  # array of tensors of A and B
for idx, (data, target) in enumerate(train_loader):
        for i in data:
            lab = kornia.color.rgb_to_lab(i/255)
            x.append(lab[0])
            y.append(lab[1:3])
            break




# class CNN(nn.Module):
#     def __init__(self, in_channels=1):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         return x
#
#
# num_epochs = 1
# batch_size = 64
# learning_rate = 0.001
#
# # Initialize the network
# model = CNN().to(device)
#
# # Loss and Optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#
# # Train the Network
#
# for epoch in range(num_epochs):
#     for batch_idx, (data, targets) in enumerate(train_dataset):
#         data = data.to(device= device)
#         targets = targets.to(device=device)
#
#         # forward
#         scores = model(data)
#         loss = criterion(scores, targets)
#
#         # gradient descent
#         optimizer.step()
#
#
# def check_accuracy(loader, model):
#     nr_correct = 0
#     nr_samples = 0
#     model.eval()
#
#     with torch.no_grad():
#         for x, y in loader:
#             x = x.to(device=device)
#             y = y.to(device=device)
#
#             scores = model(x)
#             _, predictions = scores.max(1)
#             nr_correct += (predictions == y).sum()
#             nr_samples += predictions.size(0)
#
#         print(f'Got {nr_correct}/{nr_samples} with accuracy {float(nr_correct)/float(nr_samples)*100:.2f}')
#
#     model.train()
#
# check_accuracy(train_dataset, model)
