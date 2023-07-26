#!/usr/bin/python3
import torch
from torchvision import models
from torchvision import transforms
from PIL import Image
import sys
import numpy as np

from torchsummary import summary as summary

model = models.resnet18(pretrained=True)
model.eval()

input_image = Image.open("cat.jpg")
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
sys.stdout = open('input_cat1.txt','w')
a = input_tensor.tolist()
for i in range(224):
    for j in range(224):
        print(a[0][i][j])
        print(a[1][i][j])
        print(a[2][i][j])
sys.stdout.close()


