# 재현을 위한 랜덤 시드 고정
import random
import numpy as np
import torch
from ultralytics import YOLO
from glob import glob



seed = 2023
deterministic = True

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
if deterministic:
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False



np.random.seed(724)
# dir_main = "./yolo_data/"
# filenames_image = glob(f"{dir_main}/train/images/*.jpeg")
# filenames_label = [filename.replace('images', 'labels').replace('jpeg', 'txt') for filename in filenames_image]


# print(filenames_label)


# classes = ["dog", "cat"]

# import yaml

# data = {
#     "path": './american/', # dataset root dir
#     "train" : 'train/images',
#     "val" : 'valid/images',
#     "names" : {0 : 'dog', 1 : 'cat'}}


# with open('./tld.yaml', 'w') as f :
#     yaml.dump(data, f)

# check written file
# with open('./tld.yaml', 'r') as f :
#     lines = yaml.safe_load(f)
#     print(lines)

# Load a pretrained YOLO model (recommended for training)
model = YOLO('./yolov8n.pt')

# Train the model using the 'indoor.yaml' dataset for 10 epochs
# model.train(data='./tld.yaml' , epochs=20)

# Evaluate the model's performance on the validation set
results = model.val()