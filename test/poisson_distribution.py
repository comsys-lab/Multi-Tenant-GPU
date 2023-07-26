#!/usr/bin/python3

import statistics as st
import numpy as np
from numpy import random

# # Each Inference Time (without multi-tenancy)
# # Average: 0.063786296
# interval_alexnet = 0.015063778
# interval_resnet18 = 0.016955667
# interval_vgg16 = 0.159339444

# Each Inference Time (with multi-tenancy & interval = 0)
# Average: 0.094693455
#interval_alexnet = 0.041749805
#interval_resnet18 = 0.045952712
#interval_vgg16 = 0.196377848

# Each Inference Time (with multi-tenancy & interval != 0)
# Average: 0.076218889
interval_alexnet = 0.024401499
interval_resnet18 = 0.02661417
interval_vgg16 = 0.177641

interval_list = [interval_alexnet, interval_resnet18, interval_vgg16]

minimum_inf = min(interval_list)             #minimum inference time
maximum_inf = max(interval_list)            #maximum inference time
average_inf = st.mean(interval_list)
median_inf = np.median(interval_list)

_lambda = average_inf

mod_lambda = _lambda * 1000000                 

poisson_list = random.poisson(mod_lambda, 2000)       
poisson_list = poisson_list / 1000000  

# with open("without_fusion_interval.txt","a")  as f: 
#     for i in range(2000):
#         f.write(f"{(poisson_list[i]):.6f}\n")

# with open("with_fusion_interval.txt","a")  as f: 
#     for i in range(2000):
#         f.write(f"{(poisson_list[i]):.6f}\n")