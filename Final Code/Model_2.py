###########################################################################
# Data of the problem
size = 32
batchSize = 345
train_path = 'C:/Users/xhuli/OneDrive/Desktop/Education/VUB/3rd Semester/Deep Learning/Project/images/'
epoch_num = 20
###########################################################################


###########################################################################
# Libraries
print("=> Load Libraries")
import kornia as kornia
import numpy as np
import plotly.express as px
import torch
from kornia.color import lab_to_rgb, rgb_to_linear_rgb, rgb_to_xyz, xyz_to_rgb, linear_rgb_to_rgb
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
###########################################################################


###########################################################################
# RGB to LAB
def rgb_to_lab(image: torch.Tensor) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    # Convert from sRGB to Linear RGB
    lin_rgb = rgb_to_linear_rgb(image)

    xyz_im: torch.Tensor = rgb_to_xyz(lin_rgb)

    # normalize for D65 white point
    xyz_ref_white = torch.tensor([0.95047, 1.0, 1.08883], device=xyz_im.device, dtype=xyz_im.dtype)[..., :, None, None]
    xyz_normalized = torch.div(xyz_im, xyz_ref_white)

    threshold = 0.008856
    power = torch.pow(xyz_normalized.clamp(min=threshold), 1 / 3.0)
    scale = 7.787 * xyz_normalized + 4.0 / 29.0
    xyz_int = torch.where(xyz_normalized > threshold, power, scale)

    x: torch.Tensor = xyz_int[..., 0, :, :]
    y: torch.Tensor = xyz_int[..., 1, :, :]
    z: torch.Tensor = xyz_int[..., 2, :, :]

    L: torch.Tensor = (116.0 * y) - 16.0
    a: torch.Tensor = 500.0 * (x - y)
    _b: torch.Tensor = 200.0 * (y - z)

    out: torch.Tensor = torch.stack([L, a, _b], dim=-3)

    return out
###########################################################################


###########################################################################
# LAB to RGB
def lab_to_rgb(L: torch.Tensor, a: torch.Tensor, _b: torch.Tensor, clip: bool = True) -> torch.Tensor:
    # if not isinstance(image, torch.Tensor):
    #     raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")
    #
    # if len(image.shape) < 3 or image.shape[-3] != 3:
    #     raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    # L: torch.Tensor = image[..., 0, :, :]
    # a: torch.Tensor = image[..., 1, :, :]
    # _b: torch.Tensor = image[..., 2, :, :]

    fy = (L + 16.0) / 116.0
    fx = (a / 500.0) + fy
    fz = fy - (_b / 200.0)

    # if color data out of range: Z < 0
    fz = fz.clamp(min=0.0)

    fxyz = torch.stack([fx, fy, fz], dim=-3)

    # Convert from Lab to XYZ
    power = torch.pow(fxyz, 3.0)
    scale = (fxyz - 4.0 / 29.0) / 7.787
    xyz = torch.where(fxyz > 0.2068966, power, scale)

    # For D65 white point
    xyz_ref_white = torch.tensor([0.95047, 1.0, 1.08883], device=xyz.device, dtype=xyz.dtype)[..., :, None, None]
    xyz_im = xyz * xyz_ref_white

    rgbs_im: torch.Tensor = xyz_to_rgb(xyz_im)

    # https://github.com/richzhang/colorization-pytorch/blob/66a1cb2e5258f7c8f374f582acc8b1ef99c13c27/util/util.py#L107
    #     rgbs_im = torch.where(rgbs_im < 0, torch.zeros_like(rgbs_im), rgbs_im)

    # Convert from RGB Linear to sRGB
    rgb_im = linear_rgb_to_rgb(rgbs_im)

    # Clip to 0,1 https://www.w3.org/Graphics/Color/srgb
    if clip:
        rgb_im = torch.clamp(rgb_im, min=0.0, max=1.0)

    return rgb_im
###########################################################################


###########################################################################
# Saving model
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> saving checkpoint")
    torch.save(state, filename)
###########################################################################


###########################################################################
# Device
print("=> Add device")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
###########################################################################


###########################################################################
# Transform data
train_transform = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor(),
])
###########################################################################


###########################################################################
# Load train data
print("=> Load data")
train_dataset = torchvision.datasets.ImageFolder(root=train_path, transform=train_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
###########################################################################


###########################################################################
# Model
print("=> Load model")
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.t_conv1 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)
        self.t_conv2 = nn.ConvTranspose2d(256, 64, 3, stride=2, padding=1, output_padding=1)
        self.t_conv3 = nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1)
        self.t_conv4 = nn.ConvTranspose2d(192, 15, 3, stride=1, padding=1)
        self.dropout = nn.Dropout(0.2)
        self.converge = nn.Conv2d(16, 2, 3, stride=1, padding=1)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x3))
        xd = F.relu(self.t_conv1(x4))
        xd = torch.cat((xd, x3), dim=1)
        xd = self.dropout(xd)
        xd = F.relu(self.t_conv2(xd))
        xd = torch.cat((xd, x2), dim=1)
        xd = self.dropout(xd)
        xd = F.relu(self.t_conv3(xd))
        xd = torch.cat((xd, x1), dim=1)
        xd = self.dropout(xd)
        xd = F.relu(self.t_conv4(xd))
        xd = torch.cat((xd, x), dim=1)
        x_out = F.relu(self.converge(xd))
        return x_out

###########################################################################


###########################################################################
# Optimizer and loss function
model = CNN().to(device)
print("=> Load optimizer")
optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
print("=> Load loss_func")
loss_func = torch.nn.MSELoss()
###########################################################################


###########################################################################
# Train
losses = []
predictions = []
tempPred2 = []
batch_acc_list = []
LAB = []
LABForAllEpochs = []
print("=> Start training")
for epoch in range(epoch_num):
    tempPred = []
    LAB1 = []
    Counter2 = 0
    for idx, (data, target) in enumerate(train_loader):
        Counter = 0
        Counter2 = Counter2 + 1
        for i in data:
            Counter = Counter + 1
            Lab = rgb_to_lab(i / (size-1))
            LAB1.append(Lab)
            x = Lab[0]
            x2 = np.array(x)
            L = torch.tensor(torch.tensor(np.array([x2])).unsqueeze(0))
            AB = Lab[1:3]
            # forward
            scores = model(L)
            tempPred.append(scores)
            loss = loss_func(scores, AB.unsqueeze(0))
            losses.append(loss.item())
            # backward
            optimizer.zero_grad()
            loss.backward()
            # gradient descent step
            optimizer.step()
            # if Counter%10 == 0:
            print(f'Epoch: {epoch} \t data in batch {Counter2}: {Counter}\t  loss: {loss.item()}')
        LAB.append(LAB1)
        LAB1 = []
        tempPred2.append(tempPred)
        print("---------------------------")
        batch_acc = sum(losses)/len(losses)
        batch_acc_list.append(batch_acc)
        epoch_acc = sum(batch_acc_list)/len(batch_acc_list)
        losses = []
    predictions.append(tempPred2)
    tempPred2 = []
    LABForAllEpochs.append(LAB)
    LAB = []

    checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
    save_checkpoint(checkpoint)
###########################################################################
