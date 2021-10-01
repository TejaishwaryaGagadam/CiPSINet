import torch.nn.functional as F
import torch.optim as optim
import torch, torchvision
import numpy as np
from PIL import Image
import torch.nn as nn
from torch.autograd import Variable
from torchvision.transforms import Resize, ToTensor, CenterCrop
import torchvision.models as models
import matplotlib.pylab as plt
import json
import random
from torchvision.datasets.utils import download_url
import torch.utils.data as Data
import inspect
import sys, os, gc, math
import time

from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler

class FM_CispiNet(object):
  """docstring for FM_CispiNet."""
  def __init__(self, model):
    super(FM_CispiNet, self).__init__()
    self.model = model
    self.outputs = []

  def hook(self, module, input, output):
      self.outputs[0] = output

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    self.model.cipsi_context.mp8.register_forward_hook(self.hook)
    y_pred = self.model(x)
    return self.outputs[0]

class DecisionBlockNet(nn.Module):
  def __init__(self, in_size):
    super(DecisionBlockNet, self).__init__()

    self.conv0, self.conv1, self.mp2, self.conv2a, self.conv2b, self.conv2c = define_dec_block(in_size)
    self.mp1 = nn.MaxPool2d(2, stride=2)

  def forward(self, x):
    x = self.conv0(x)
    x = self.conv1(x)
    x = self.mp1(x)
    m = self.mp2(x)
    a = self.conv2a(x)
    b = self.conv2b(x)
    c = self.conv2c(x)

    x = torch.cat([m, a, b, c], dim=1)
    # x = torch.cat([b, c], dim=1)

    return x

class WideSinglePathCiPSiNet(nn.Module):
  def __init__(self, use_grad):
    super(WideSinglePathCiPSiNet, self).__init__()
    self.use_grad = use_grad

    self.cipsi_context = WideCiPSiNet(1)

    self.fc1 = define_fc_layers(512 * 4 * 4 * 1)

  def forward(self, y):

    omega = self.cipsi_context(y)

    return self.fc1(omega)

class WideSinglePathCiPSiNetClasses(nn.Module):
  def __init__(self, classes):
    super(WideSinglePathCiPSiNetClasses, self).__init__()
    self.use_grad = False

    self.cipsi_context = WideCiPSiNet(1)

    self.fc1 = define_fc_layers(512 * 4 * 4 * 1, classes)

  def forward(self, y):

    if self.use_grad:
      z = sobel_conv(y)

      y = torch.cat([y, z], dim=1)

    omega = self.cipsi_context(y)

    return self.fc1(omega)

class WrapWideCiPSiNet(nn.Module):
  def __init__(self, model):
    super(WrapWideCiPSiNet, self).__init__()

    self.model = model

  def forward(self, x):

    x = self.model.conv1(x)

    x = self.model.mp1_5(x)

    a = self.model.mp2(x)
    b = self.model.conv2a(x)
    c = self.model.conv2b(x)
    d = self.model.conv2c(x)

    a = self.model.conv3a(a)
    b = self.model.conv3b(b)
    c = self.model.conv3c(c)
    d = self.model.conv3d(d)

    x = torch.cat([a, b, c, d], dim=1)
    x = self.model.conv7(x)
    x = self.model.mp4(x)

    x = self.model.conv8(x)

    return x

class WideCiPSiNet(nn.Module):
  def __init__(self, in_layers):
    super(WideCiPSiNet, self).__init__()

    self.conv1, self.mp2, self.conv2a, self.conv2b, self.conv2c = define_init_block(in_layers, 96)

    self.mp1_5 = nn.MaxPool2d(2, stride=2)

    self.conv3a = DecisionBlockNet(96)
    self.conv3b = DecisionBlockNet(128)
    self.conv3c = DecisionBlockNet(108)
    self.conv3d = DecisionBlockNet(108)

    self.conv7 = nn.Sequential(
      nn.Conv2d(1472, 1024, kernel_size=1, stride=1),
      nn.BatchNorm2d(1024),
      nn.ReLU(inplace=True)
    )

    self.mp4 = nn.MaxPool2d(2, stride=2)

    self.conv8 = nn.Sequential(
      nn.Conv2d(1024, 512, kernel_size=1, stride=1),
      nn.BatchNorm2d(512),
      nn.ReLU(inplace=True)
    )
    self.mp8 = nn.MaxPool2d(4, stride=4)


  def forward(self, x):

    x = self.conv1(x)

    x = self.mp1_5(x)

    a = self.mp2(x)
    b = self.conv2a(x)
    c = self.conv2b(x)
    d = self.conv2c(x)

    a = self.conv3a(a)
    b = self.conv3b(b)
    c = self.conv3c(c)
    d = self.conv3d(d)

    x = torch.cat([a, b, c, d], dim=1)
    x = self.conv7(x)
    x = self.mp4(x)

    x = self.conv8(x)
    x = self.mp8(x)

    x = torch.flatten(x, start_dim=1)

    return x


def define_init_block(layer_count, second_count=64):
  dropout = 0.2
  conv1 = nn.Sequential(
    #128
    nn.Conv2d(layer_count, second_count, kernel_size = 3, stride=1, padding=3, dilation=3),
    nn.BatchNorm2d(second_count),
    nn.ReLU(inplace=True)
    #54
  )
  conv2a = nn.Sequential(
    #128
    nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    #54
  )
  dropout = 0.15
  conv2b = nn.Sequential(
    #128
    nn.Conv2d(second_count, 128, kernel_size = 1, stride=1),
    nn.BatchNorm2d(128),
    nn.ReLU(inplace=True)
    #54
  )
  conv2c = nn.Sequential(
    #128
    nn.Conv2d(second_count, 108, kernel_size = 3, stride=1, padding=2, dilation=2),
    nn.BatchNorm2d(108),
    nn.ReLU(inplace=True)
    #54
  )
  conv2d = nn.Sequential(
    #128
    nn.Conv2d(second_count, 108, kernel_size = 3, stride=1, padding=1),
    nn.BatchNorm2d(108),
    nn.ReLU(inplace=True)
    #54
  )

  return conv1, conv2a, conv2b, conv2c, conv2d

def define_dec_block(layer_count):
  dropout = 0.15
  conv0 = nn.Sequential(
    #128
    nn.Conv2d(layer_count, 64, kernel_size = 3, stride=1, dilation=2, padding=2),
    nn.BatchNorm2d(64),
    nn.ReLU(inplace=True)
    #54
  )
  conv1 = nn.Sequential(
    #128
    nn.Conv2d(64, 512, kernel_size = 1, stride=1),
    nn.BatchNorm2d(512),
    nn.ReLU(inplace=True)
    #54
  )
  conv2a = nn.Sequential(
    #128
    nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
    nn.Conv2d(512, 108, kernel_size = 1, stride=1),
    nn.BatchNorm2d(108),
    nn.ReLU(inplace=True),
    nn.Conv2d(108, 92, kernel_size = 3, stride=1, padding=1),
    nn.BatchNorm2d(92),
    nn.ReLU(inplace=True),
    #54
  )
  conv2b = nn.Sequential(
    #128
    nn.Conv2d(512, 64, kernel_size = 1, stride=1),
    nn.BatchNorm2d(64),
    nn.ReLU(inplace=True),
    nn.Conv2d(64, 92, kernel_size = 3, stride=1, padding=1),
    nn.BatchNorm2d(92),
    nn.ReLU(inplace=True),
    #54
  )
  conv2c = nn.Sequential(
    #128
    nn.Conv2d(512, 48, kernel_size = 1, stride=1),
    nn.BatchNorm2d(48),
    nn.ReLU(inplace=True),
    nn.Conv2d(48, 68, kernel_size = 5, stride=1, padding=2),
    nn.BatchNorm2d(68),
    nn.ReLU(inplace=True),
    nn.Conv2d(68, 92, kernel_size = 3, stride=1, padding=1),
    nn.BatchNorm2d(92),
    nn.ReLU(inplace=True),
    #54
  )
  conv2d = nn.Sequential(
    #128
    nn.Conv2d(512, 48, kernel_size = 1, stride=1),
    nn.BatchNorm2d(48),
    nn.ReLU(inplace=True),
    nn.Conv2d(48, 108, kernel_size = 3, stride=1, padding=2, dilation=2),
    nn.BatchNorm2d(108),
    nn.ReLU(inplace=True),
    nn.Conv2d(108, 92, kernel_size = 3, stride=1, padding=1),
    nn.BatchNorm2d(92),
    nn.ReLU(inplace=True),
    #54
  )

  return conv0, conv1, conv2a, conv2b, conv2c, conv2d

def define_fc_layers(start_layers, num_classes=2):
  dropout = 0.25
  net = nn.Sequential(
      nn.Linear(start_layers, 2048),
      nn.ReLU(True),
      nn.Dropout(p=dropout),
      nn.Linear(2048, 512),
      nn.ReLU(True),
      nn.Linear(512, num_classes),
  )

  return net


def load_pt_model(path):
  model = torch.load(path)
  model.eval()
  return model
