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


# TODO: Helper Functions

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


def lab_to_rgb(L: torch.Tensor, a: torch.Tensor, _b: torch.Tensor, clip: bool = True) -> torch.Tensor:
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


class AverageMeter(object):
    '''A handy class from the PyTorch ImageNet tutorial'''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# TODO:  Device Configuration
size = 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_path = 'C:/Users/xhuli/OneDrive/Desktop/Education/VUB/3rd Semester/Deep Learning/Project/images/'

train_transform = transforms.Compose([
    transforms.Resize((size, size)),  # can experiment
    transforms.ToTensor(),  # divide the pixel by 255
])

# TODO: Split the dataset
# Loading the data
train_dataset = torchvision.datasets.ImageFolder(root=train_path, transform=train_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=72, shuffle=True)

# TODO: Convert RDG to LAB space
X = []  # array of tensors of  L
y = []  # array of tensors of A and B
LAB = []
z = 0
for idx, (data, target) in enumerate(train_loader):
    for i in data:
        z = z + 1
        Lab = rgb_to_lab(i / (size - 1))
        LAB.append(Lab)
        X.append(Lab[0])
        y.append(Lab[1:3])

x = [] # convert to np.array and add the batch size
for i in X:
    w = np.array(i)
    x.append(torch.tensor(np.array([w])).unsqueeze(0))

for i in range(len(x)): # putting the batch size
    y[i] = y[i].unsqueeze(0)


# print(x[0].shape, " ", y[0].shape)


# TODO: Put a better model, use resnet18
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


# Initialize the network
model = CNN().to(device)
# print("Model Shape= ", model(x[0]).shape)
# print("Y shape=  ", y[0].shape)
# print("Size 0 of X", x[0].size(0))

optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
loss_func = torch.nn.MSELoss()
losses = AverageMeter()

predictions = []

for epoch in range(5):
    tempPred = []
    for i in range(len(x)):
        # forward
        scores = model(x[i])
        tempPred.append(scores)
        loss = loss_func(scores, y[i])
        losses.update(loss.item(), x[i].size(0))
        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent step
        optimizer.step()

        if i % 25 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(train_loader), loss=losses))

    predictions.append(tempPred)

#  TODO: *************************** check this ***************************
# showing the prediction of picture 71
y1 = y[71][0:1]
y2 = y[71][1:2]
output = lab_to_rgb(torch.squeeze(x[71]) * size, torch.squeeze(predictions[4][71][0:1]) * size, torch.squeeze(predictions[4][71][1:2]) * size)
transform = T.ToPILImage()
img = transform(torch.squeeze(output))
img.show()
