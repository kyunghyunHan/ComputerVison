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
dir_main = "./yolo_data/"
filenames_image = glob(f"{dir_main}/train/images/*.jpg")
filenames_label = [filename.replace('images', 'labels').replace('jpg', 'txt') for filename in filenames_image]


print(filenames_label)


classes = ["dog", "cat"]


model = YOLO('./yolov8n.pt')
model.train(data='./tld.yaml' , epochs=20)


# results = model.val()