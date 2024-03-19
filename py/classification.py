import os
import sys
import glob
import csv
import cv2
import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision
from torchvision import transforms, models

# 재현성을 위한 랜덤시드 고정
random_seed = 2023
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

# 데이터 전처리
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 배치 사이즈와 train:validation 비율 정의
batch_size = 256
val_size = 0.2

# torchvision에서 제공하는 CIFAR10 학습 데이터셋 다운로드
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                              download=True, transform=train_transform)
val_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=val_transform)

# Train 데이터에서 일정 비율 vaildation data 분리
num_train = len(train_dataset)
indices = list(range(num_train))
split = int(np.floor(val_size * num_train))
train_idx, val_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)

# 데이터로더 정의
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           sampler=train_sampler, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                         sampler=val_sampler, num_workers=0)

# torchvision에서 제공하는 CIFAR10 테스트 데이터셋 다운로드
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=val_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                          shuffle=False, num_workers=0)

# 클래스 정의
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 데이터셋 확인
print(train_dataset)

# 이미지 데이터 시각화
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# 학습 이미지 얻기
dataiter = iter(train_loader)
images, labels = next(dataiter)
# 이미지 출력
imshow(torchvision.utils.make_grid(images))
# 라벨 프린트
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))