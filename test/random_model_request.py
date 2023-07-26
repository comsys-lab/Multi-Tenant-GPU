#!/usr/bin/python3
from numpy import random

for i in range(2000):
    with open("queue_model.txt","a")  as f:                 
        selected_model = random.choice(['alexnet','resnet18','vgg16'])
        if(selected_model == 'alexnet'):
            f.write('alexnet\n')
        elif(selected_model == 'resnet18'):
            f.write('resnet18\n')
        elif(selected_model == 'vgg16'):
            f.write('vgg16\n')
