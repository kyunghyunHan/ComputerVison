## 모델 불러오기

import os,sys
import glob
import csv
import cv2
import tqdm
import numpy as np
import matplotlib.pyplot as pyplot
from PIL import Image
from typing import Tuple,List,Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import  torchvision
from torchvision import transforms,models


## 데이터전처리

train_trsnform = transforms.Compose([])


## 배치사이즈

batch_size = 8
var_size= 0.2


## 커스텀 데이터셋 클래스
class CUSTOMDataset(Dataset):
    def __init__():


    def __len__():


    def __getitem__():