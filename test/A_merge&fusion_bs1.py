#!/usr/bin/python3

# Before run this code, 
# Execute '/usr/local/cuda/bin/nvcc -Xcompiler -fPIC -shared -o Merge&Fusion_kernels.so Merge&Fusion_kernels.cu' first.

from concurrent.futures import process
from urllib.parse import uses_query
import time, os
import sys
from math import factorial, exp
import multiprocessing
from multiprocessing import Process, Queue, Value
import numpy as np
import statistics as st
from numpy import random
import ctypes
from ctypes import *

def Alex_Res_inference(stopping, starttime_list, endtime_list):
    dll = ctypes.CDLL('./Merge&Fusion_kernels.so',mode=ctypes.RTLD_GLOBAL)
    Alex_Res_host2gpu = dll.Alex_Res_host2gpu
    Alex_Res_host2gpu.argtypes = [POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float))]

    Alex_Res_inference = dll.Alex_Res_inference
    Alex_Res_inference.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                c_int,c_int]
                    
    Alex_Res_cudafree = dll.Alex_Res_cudafree
    Alex_Res_cudafree.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float)]

    
    A_L1_N = POINTER(c_float)()
    A_L2_N = POINTER(c_float)()
    A_L3_N = POINTER(c_float)()
    A_L4_N = POINTER(c_float)()
    A_L5_N = POINTER(c_float)()
    A_L6_N = POINTER(c_float)()
    A_L7_N = POINTER(c_float)()
    A_L8_N = POINTER(c_float)()

    A_L1_b = POINTER(c_float)()
    A_L2_b = POINTER(c_float)()
    A_L3_b = POINTER(c_float)()
    A_L4_b = POINTER(c_float)()
    A_L5_b = POINTER(c_float)()
    A_L6_b = POINTER(c_float)()
    A_L7_b = POINTER(c_float)()
    A_L8_b = POINTER(c_float)()

    A_L1_w = POINTER(c_float)()
    A_L2_w = POINTER(c_float)()
    A_L3_w = POINTER(c_float)()
    A_L4_w = POINTER(c_float)()
    A_L5_w = POINTER(c_float)()
    A_L6_w = POINTER(c_float)()
    A_L7_w = POINTER(c_float)()
    A_L8_w = POINTER(c_float)()

    A_L1_pool = POINTER(c_float)()
    A_L2_pool = POINTER(c_float)()
    A_L5_pool = POINTER(c_float)()
    A_L1_norm = POINTER(c_float)()
    A_L2_norm = POINTER(c_float)()
    A_Result_N = POINTER(c_float)()

    R_L1_N = POINTER(c_float)()
    R_L2_N = POINTER(c_float)()
    R_L3_N = POINTER(c_float)()
    R_L4_N = POINTER(c_float)()
    R_L5_N = POINTER(c_float)()
    R_L6_N = POINTER(c_float)()
    R_L7_N = POINTER(c_float)()
    R_L8_N = POINTER(c_float)()
    R_L9_N = POINTER(c_float)()
    R_L10_N = POINTER(c_float)()
    R_L11_N = POINTER(c_float)()
    R_L12_N = POINTER(c_float)()
    R_L13_N = POINTER(c_float)()
    R_L14_N = POINTER(c_float)()
    R_L15_N = POINTER(c_float)()
    R_L16_N = POINTER(c_float)()
    R_L17_N = POINTER(c_float)()
    R_L18_N = POINTER(c_float)()

    R_L1_w = POINTER(c_float)()
    R_L2_w = POINTER(c_float)()
    R_L3_w = POINTER(c_float)()
    R_L4_w = POINTER(c_float)()
    R_L5_w = POINTER(c_float)()
    R_L6_w = POINTER(c_float)()
    R_L7_w = POINTER(c_float)()
    R_L8_w = POINTER(c_float)()
    R_L9_w = POINTER(c_float)()
    R_L10_w = POINTER(c_float)()
    R_L11_w = POINTER(c_float)()
    R_L12_w = POINTER(c_float)()
    R_L13_w = POINTER(c_float)()
    R_L14_w = POINTER(c_float)()
    R_L15_w = POINTER(c_float)()
    R_L16_w = POINTER(c_float)()
    R_L17_w = POINTER(c_float)()
    R_B3_w = POINTER(c_float)()
    R_B4_w = POINTER(c_float)()
    R_B5_w = POINTER(c_float)()

    R_L1_G = POINTER(c_float)()
    R_L2_G = POINTER(c_float)()
    R_L3_G = POINTER(c_float)()
    R_L4_G = POINTER(c_float)()
    R_L5_G = POINTER(c_float)()
    R_L6_G = POINTER(c_float)()
    R_L7_G = POINTER(c_float)()
    R_L8_G = POINTER(c_float)()
    R_L9_G = POINTER(c_float)()
    R_L10_G = POINTER(c_float)()
    R_L11_G = POINTER(c_float)()
    R_L12_G = POINTER(c_float)()
    R_L13_G = POINTER(c_float)()
    R_L14_G = POINTER(c_float)()
    R_L15_G = POINTER(c_float)()
    R_L16_G = POINTER(c_float)()
    R_L17_G = POINTER(c_float)()
    R_B3_G = POINTER(c_float)()
    R_B4_G = POINTER(c_float)()
    R_B5_G = POINTER(c_float)()

    R_L1_B = POINTER(c_float)()
    R_L2_B = POINTER(c_float)()
    R_L3_B = POINTER(c_float)()
    R_L4_B = POINTER(c_float)()
    R_L5_B = POINTER(c_float)()
    R_L6_B = POINTER(c_float)()
    R_L7_B = POINTER(c_float)()
    R_L8_B = POINTER(c_float)()
    R_L9_B = POINTER(c_float)()
    R_L10_B = POINTER(c_float)()
    R_L11_B = POINTER(c_float)()
    R_L12_B = POINTER(c_float)()
    R_L13_B = POINTER(c_float)()
    R_L14_B = POINTER(c_float)()
    R_L15_B = POINTER(c_float)()
    R_L16_B = POINTER(c_float)()
    R_L17_B = POINTER(c_float)()
    R_B3_B = POINTER(c_float)()
    R_B4_B = POINTER(c_float)()
    R_B5_B = POINTER(c_float)()

    R_L1_M = POINTER(c_float)()
    R_L2_M = POINTER(c_float)()
    R_L3_M = POINTER(c_float)()
    R_L4_M = POINTER(c_float)()
    R_L5_M = POINTER(c_float)()
    R_L6_M = POINTER(c_float)()
    R_L7_M = POINTER(c_float)()
    R_L8_M = POINTER(c_float)()
    R_L9_M = POINTER(c_float)()
    R_L10_M = POINTER(c_float)()
    R_L11_M = POINTER(c_float)()
    R_L12_M = POINTER(c_float)()
    R_L13_M = POINTER(c_float)()
    R_L14_M = POINTER(c_float)()
    R_L15_M = POINTER(c_float)()
    R_L16_M = POINTER(c_float)()
    R_L17_M = POINTER(c_float)()
    R_B3_M = POINTER(c_float)()
    R_B4_M = POINTER(c_float)()
    R_B5_M = POINTER(c_float)()

    R_L1_V = POINTER(c_float)()
    R_L2_V = POINTER(c_float)()
    R_L3_V = POINTER(c_float)()
    R_L4_V = POINTER(c_float)()
    R_L5_V = POINTER(c_float)()
    R_L6_V = POINTER(c_float)()
    R_L7_V = POINTER(c_float)()
    R_L8_V = POINTER(c_float)()
    R_L9_V = POINTER(c_float)()
    R_L10_V = POINTER(c_float)()
    R_L11_V = POINTER(c_float)()
    R_L12_V = POINTER(c_float)()
    R_L13_V = POINTER(c_float)()
    R_L14_V = POINTER(c_float)()
    R_L15_V = POINTER(c_float)()
    R_L16_V = POINTER(c_float)()
    R_L17_V = POINTER(c_float)()
    R_B3_V = POINTER(c_float)()
    R_B4_V = POINTER(c_float)()
    R_B5_V = POINTER(c_float)()

    R_FC_b = POINTER(c_float)()
    R_FC_w = POINTER(c_float)()

    R_L3_basic = POINTER(c_float)()
    R_L5_basic = POINTER(c_float)()
    R_L7_basic = POINTER(c_float)()
    R_L9_basic = POINTER(c_float)()
    R_L11_basic = POINTER(c_float)()
    R_L13_basic = POINTER(c_float)()
    R_L15_basic = POINTER(c_float)()
    R_L17_basic = POINTER(c_float)()
    R_B3_basic = POINTER(c_float)()
    R_B4_basic = POINTER(c_float)()
    R_B5_basic = POINTER(c_float)()
    R_L1_bn = POINTER(c_float)()
    R_L2_bn = POINTER(c_float)()
    R_L3_bn = POINTER(c_float)()
    R_L4_bn = POINTER(c_float)()
    R_L5_bn = POINTER(c_float)()
    R_L6_bn = POINTER(c_float)()
    R_L7_bn = POINTER(c_float)()
    R_L8_bn = POINTER(c_float)()
    R_L9_bn = POINTER(c_float)()
    R_L10_bn = POINTER(c_float)()
    R_L11_bn = POINTER(c_float)()
    R_L12_bn = POINTER(c_float)()
    R_L13_bn = POINTER(c_float)()
    R_L14_bn = POINTER(c_float)()
    R_L15_bn = POINTER(c_float)()
    R_L16_bn = POINTER(c_float)()
    R_L17_bn = POINTER(c_float)()
    R_B3_bn = POINTER(c_float)()
    R_B4_bn = POINTER(c_float)()
    R_B5_bn = POINTER(c_float)()
    R_L1_pool = POINTER(c_float)()
    R_FC_N = POINTER(c_float)()
    R_Result_N = POINTER(c_float)()

    Alex_Res_host2gpu(A_L1_N,A_L2_N,A_L3_N,A_L4_N,
                    A_L5_N,A_L6_N,A_L7_N,A_L8_N,
                    A_L1_b,A_L2_b,A_L3_b,A_L4_b,
                    A_L5_b,A_L6_b,A_L7_b,A_L8_b,
                    A_L1_w,A_L2_w,A_L3_w,A_L4_w,
                    A_L5_w,A_L6_w,A_L7_w,A_L8_w,
                    A_L1_pool,A_L2_pool,A_L5_pool,
                    A_L1_norm,A_L2_norm,A_Result_N,
                    R_L1_N,R_L2_N,R_L3_N,R_L4_N,
                    R_L5_N,R_L6_N,R_L7_N,R_L8_N,
                    R_L9_N,R_L10_N,R_L11_N,R_L12_N,
                    R_L13_N,R_L14_N,R_L15_N,R_L16_N,
                    R_L17_N,R_L18_N,
                    R_L1_w,R_L2_w,R_L3_w,R_L4_w,
                    R_L5_w,R_L6_w,R_L7_w,R_L8_w,
                    R_L9_w,R_L10_w,R_L11_w,R_L12_w,
                    R_L13_w,R_L14_w,R_L15_w,R_L16_w,
                    R_L17_w,R_B3_w,R_B4_w,R_B5_w,
                    R_L1_G,R_L2_G,R_L3_G,R_L4_G,
                    R_L5_G,R_L6_G,R_L7_G,R_L8_G,
                    R_L9_G,R_L10_G,R_L11_G,R_L12_G,
                    R_L13_G,R_L14_G,R_L15_G,R_L16_G,
                    R_L17_G,R_B3_G,R_B4_G,R_B5_G,
                    R_L1_B,R_L2_B,R_L3_B,R_L4_B,
                    R_L5_B,R_L6_B,R_L7_B,R_L8_B,
                    R_L9_B,R_L10_B,R_L11_B,R_L12_B,
                    R_L13_B,R_L14_B,R_L15_B,R_L16_B,
                    R_L17_B,R_B3_B,R_B4_B,R_B5_B,
                    R_L1_M,R_L2_M,R_L3_M,R_L4_M,
                    R_L5_M,R_L6_M,R_L7_M,R_L8_M,
                    R_L9_M,R_L10_M,R_L11_M,R_L12_M,
                    R_L13_M,R_L14_M,R_L15_M,R_L16_M,
                    R_L17_M,R_B3_M,R_B4_M,R_B5_M,
                    R_L1_V,R_L2_V,R_L3_V,R_L4_V,
                    R_L5_V,R_L6_V,R_L7_V,R_L8_V,
                    R_L9_V,R_L10_V,R_L11_V,R_L12_V,
                    R_L13_V,R_L14_V,R_L15_V,R_L16_V,
                    R_L17_V,R_B3_V,R_B4_V,R_B5_V,
                    R_FC_b,R_FC_w,
                    R_L3_basic,R_L5_basic,R_L7_basic,R_L9_basic,
                    R_L11_basic,R_L13_basic,R_L15_basic,R_L17_basic,
                    R_B3_basic,R_B4_basic,R_B5_basic,
                    R_L1_bn,R_L2_bn,R_L3_bn,R_L4_bn,
                    R_L5_bn,R_L6_bn,R_L7_bn,R_L8_bn,
                    R_L9_bn,R_L10_bn,R_L11_bn,R_L12_bn,
                    R_L13_bn,R_L14_bn,R_L15_bn,R_L16_bn,
                    R_L17_bn,R_B3_bn,R_B4_bn,R_B5_bn,
                    R_L1_pool,R_FC_N,R_Result_N)

    # warm up
    for i in range(10):
        Alex_Res_inference(A_L1_N,A_L2_N,A_L3_N,A_L4_N,
                            A_L5_N,A_L6_N,A_L7_N,A_L8_N,
                            A_L1_b,A_L2_b,A_L3_b,A_L4_b,
                            A_L5_b,A_L6_b,A_L7_b,A_L8_b,
                            A_L1_w,A_L2_w,A_L3_w,A_L4_w,
                            A_L5_w,A_L6_w,A_L7_w,A_L8_w,
                            A_L1_pool,A_L2_pool,A_L5_pool,
                            A_L1_norm,A_L2_norm,A_Result_N,
                            R_L1_N,R_L2_N,R_L3_N,R_L4_N,
                            R_L5_N,R_L6_N,R_L7_N,R_L8_N,
                            R_L9_N,R_L10_N,R_L11_N,R_L12_N,
                            R_L13_N,R_L14_N,R_L15_N,R_L16_N,
                            R_L17_N,R_L18_N,
                            R_L1_w,R_L2_w,R_L3_w,R_L4_w,
                            R_L5_w,R_L6_w,R_L7_w,R_L8_w,
                            R_L9_w,R_L10_w,R_L11_w,R_L12_w,
                            R_L13_w,R_L14_w,R_L15_w,R_L16_w,
                            R_L17_w,R_B3_w,R_B4_w,R_B5_w,
                            R_L1_G,R_L2_G,R_L3_G,R_L4_G,
                            R_L5_G,R_L6_G,R_L7_G,R_L8_G,
                            R_L9_G,R_L10_G,R_L11_G,R_L12_G,
                            R_L13_G,R_L14_G,R_L15_G,R_L16_G,
                            R_L17_G,R_B3_G,R_B4_G,R_B5_G,
                            R_L1_B,R_L2_B,R_L3_B,R_L4_B,
                            R_L5_B,R_L6_B,R_L7_B,R_L8_B,
                            R_L9_B,R_L10_B,R_L11_B,R_L12_B,
                            R_L13_B,R_L14_B,R_L15_B,R_L16_B,
                            R_L17_B,R_B3_B,R_B4_B,R_B5_B,
                            R_L1_M,R_L2_M,R_L3_M,R_L4_M,
                            R_L5_M,R_L6_M,R_L7_M,R_L8_M,
                            R_L9_M,R_L10_M,R_L11_M,R_L12_M,
                            R_L13_M,R_L14_M,R_L15_M,R_L16_M,
                            R_L17_M,R_B3_M,R_B4_M,R_B5_M,
                            R_L1_V,R_L2_V,R_L3_V,R_L4_V,
                            R_L5_V,R_L6_V,R_L7_V,R_L8_V,
                            R_L9_V,R_L10_V,R_L11_V,R_L12_V,
                            R_L13_V,R_L14_V,R_L15_V,R_L16_V,
                            R_L17_V,R_B3_V,R_B4_V,R_B5_V,
                            R_FC_b,R_FC_w,
                            R_L3_basic,R_L5_basic,R_L7_basic,R_L9_basic,
                            R_L11_basic,R_L13_basic,R_L15_basic,R_L17_basic,
                            R_B3_basic,R_B4_basic,R_B5_basic,
                            R_L1_bn,R_L2_bn,R_L3_bn,R_L4_bn,
                            R_L5_bn,R_L6_bn,R_L7_bn,R_L8_bn,
                            R_L9_bn,R_L10_bn,R_L11_bn,R_L12_bn,
                            R_L13_bn,R_L14_bn,R_L15_bn,R_L16_bn,
                            R_L17_bn,R_B3_bn,R_B4_bn,R_B5_bn,
                            R_L1_pool,R_FC_N,R_Result_N,
                            1,1)

    stopping.value = stopping.value + 1

    print("Alex&Res Inference Ready!")
    while(stopping.value < 2):
        continue

    starttime = time.time()
    starttime_list.append(starttime)

    Alex_Res_inference(A_L1_N,A_L2_N,A_L3_N,A_L4_N,
                        A_L5_N,A_L6_N,A_L7_N,A_L8_N,
                        A_L1_b,A_L2_b,A_L3_b,A_L4_b,
                        A_L5_b,A_L6_b,A_L7_b,A_L8_b,
                        A_L1_w,A_L2_w,A_L3_w,A_L4_w,
                        A_L5_w,A_L6_w,A_L7_w,A_L8_w,
                        A_L1_pool,A_L2_pool,A_L5_pool,
                        A_L1_norm,A_L2_norm,A_Result_N,
                        R_L1_N,R_L2_N,R_L3_N,R_L4_N,
                        R_L5_N,R_L6_N,R_L7_N,R_L8_N,
                        R_L9_N,R_L10_N,R_L11_N,R_L12_N,
                        R_L13_N,R_L14_N,R_L15_N,R_L16_N,
                        R_L17_N,R_L18_N,
                        R_L1_w,R_L2_w,R_L3_w,R_L4_w,
                        R_L5_w,R_L6_w,R_L7_w,R_L8_w,
                        R_L9_w,R_L10_w,R_L11_w,R_L12_w,
                        R_L13_w,R_L14_w,R_L15_w,R_L16_w,
                        R_L17_w,R_B3_w,R_B4_w,R_B5_w,
                        R_L1_G,R_L2_G,R_L3_G,R_L4_G,
                        R_L5_G,R_L6_G,R_L7_G,R_L8_G,
                        R_L9_G,R_L10_G,R_L11_G,R_L12_G,
                        R_L13_G,R_L14_G,R_L15_G,R_L16_G,
                        R_L17_G,R_B3_G,R_B4_G,R_B5_G,
                        R_L1_B,R_L2_B,R_L3_B,R_L4_B,
                        R_L5_B,R_L6_B,R_L7_B,R_L8_B,
                        R_L9_B,R_L10_B,R_L11_B,R_L12_B,
                        R_L13_B,R_L14_B,R_L15_B,R_L16_B,
                        R_L17_B,R_B3_B,R_B4_B,R_B5_B,
                        R_L1_M,R_L2_M,R_L3_M,R_L4_M,
                        R_L5_M,R_L6_M,R_L7_M,R_L8_M,
                        R_L9_M,R_L10_M,R_L11_M,R_L12_M,
                        R_L13_M,R_L14_M,R_L15_M,R_L16_M,
                        R_L17_M,R_B3_M,R_B4_M,R_B5_M,
                        R_L1_V,R_L2_V,R_L3_V,R_L4_V,
                        R_L5_V,R_L6_V,R_L7_V,R_L8_V,
                        R_L9_V,R_L10_V,R_L11_V,R_L12_V,
                        R_L13_V,R_L14_V,R_L15_V,R_L16_V,
                        R_L17_V,R_B3_V,R_B4_V,R_B5_V,
                        R_FC_b,R_FC_w,
                        R_L3_basic,R_L5_basic,R_L7_basic,R_L9_basic,
                        R_L11_basic,R_L13_basic,R_L15_basic,R_L17_basic,
                        R_B3_basic,R_B4_basic,R_B5_basic,
                        R_L1_bn,R_L2_bn,R_L3_bn,R_L4_bn,
                        R_L5_bn,R_L6_bn,R_L7_bn,R_L8_bn,
                        R_L9_bn,R_L10_bn,R_L11_bn,R_L12_bn,
                        R_L13_bn,R_L14_bn,R_L15_bn,R_L16_bn,
                        R_L17_bn,R_B3_bn,R_B4_bn,R_B5_bn,
                        R_L1_pool,R_FC_N,R_Result_N,
                        1,1)

    endtime = time.time()
    endtime_list.append(endtime)
    print(f"Alex&Res inference time : {(endtime-starttime):.6f} seconds")

    Alex_Res_cudafree(A_L1_N,A_L2_N,A_L3_N,A_L4_N,
                    A_L5_N,A_L6_N,A_L7_N,A_L8_N,
                    A_L1_b,A_L2_b,A_L3_b,A_L4_b,
                    A_L5_b,A_L6_b,A_L7_b,A_L8_b,
                    A_L1_w,A_L2_w,A_L3_w,A_L4_w,
                    A_L5_w,A_L6_w,A_L7_w,A_L8_w,
                    A_L1_pool,A_L2_pool,A_L5_pool,
                    A_L1_norm,A_L2_norm,A_Result_N,
                    R_L1_N,R_L2_N,R_L3_N,R_L4_N,
                    R_L5_N,R_L6_N,R_L7_N,R_L8_N,
                    R_L9_N,R_L10_N,R_L11_N,R_L12_N,
                    R_L13_N,R_L14_N,R_L15_N,R_L16_N,
                    R_L17_N,R_L18_N,
                    R_L1_w,R_L2_w,R_L3_w,R_L4_w,
                    R_L5_w,R_L6_w,R_L7_w,R_L8_w,
                    R_L9_w,R_L10_w,R_L11_w,R_L12_w,
                    R_L13_w,R_L14_w,R_L15_w,R_L16_w,
                    R_L17_w,R_B3_w,R_B4_w,R_B5_w,
                    R_L1_G,R_L2_G,R_L3_G,R_L4_G,
                    R_L5_G,R_L6_G,R_L7_G,R_L8_G,
                    R_L9_G,R_L10_G,R_L11_G,R_L12_G,
                    R_L13_G,R_L14_G,R_L15_G,R_L16_G,
                    R_L17_G,R_B3_G,R_B4_G,R_B5_G,
                    R_L1_B,R_L2_B,R_L3_B,R_L4_B,
                    R_L5_B,R_L6_B,R_L7_B,R_L8_B,
                    R_L9_B,R_L10_B,R_L11_B,R_L12_B,
                    R_L13_B,R_L14_B,R_L15_B,R_L16_B,
                    R_L17_B,R_B3_B,R_B4_B,R_B5_B,
                    R_L1_M,R_L2_M,R_L3_M,R_L4_M,
                    R_L5_M,R_L6_M,R_L7_M,R_L8_M,
                    R_L9_M,R_L10_M,R_L11_M,R_L12_M,
                    R_L13_M,R_L14_M,R_L15_M,R_L16_M,
                    R_L17_M,R_B3_M,R_B4_M,R_B5_M,
                    R_L1_V,R_L2_V,R_L3_V,R_L4_V,
                    R_L5_V,R_L6_V,R_L7_V,R_L8_V,
                    R_L9_V,R_L10_V,R_L11_V,R_L12_V,
                    R_L13_V,R_L14_V,R_L15_V,R_L16_V,
                    R_L17_V,R_B3_V,R_B4_V,R_B5_V,
                    R_FC_b,R_FC_w,
                    R_L3_basic,R_L5_basic,R_L7_basic,R_L9_basic,
                    R_L11_basic,R_L13_basic,R_L15_basic,R_L17_basic,
                    R_B3_basic,R_B4_basic,R_B5_basic,
                    R_L1_bn,R_L2_bn,R_L3_bn,R_L4_bn,
                    R_L5_bn,R_L6_bn,R_L7_bn,R_L8_bn,
                    R_L9_bn,R_L10_bn,R_L11_bn,R_L12_bn,
                    R_L13_bn,R_L14_bn,R_L15_bn,R_L16_bn,
                    R_L17_bn,R_B3_bn,R_B4_bn,R_B5_bn,
                    R_L1_pool,R_FC_N,R_Result_N)

def Vgg_merge_inference(stopping, starttime_list, endtime_list):
    dll = ctypes.CDLL('./Merge&Fusion_kernels.so',mode=ctypes.RTLD_GLOBAL)
    Vgg_host2gpu = dll.Vgg_host2gpu
    Vgg_host2gpu.argtypes = [POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                            POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                            POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                            POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                            POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                            POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                            POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                            POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                            POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                            POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                            POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                            POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                            POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                            POINTER(POINTER(c_float)),POINTER(POINTER(c_float))]
    
    Vgg_merge_inference = dll.Vgg_merge_inference
    Vgg_merge_inference.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),c_int]

    Vgg_cudafree = dll.Vgg_cudafree
    Vgg_cudafree.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                            POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                            POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                            POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                            POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                            POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                            POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                            POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                            POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                            POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                            POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                            POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                            POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                            POINTER(c_float),POINTER(c_float)]

    V_L1_N = POINTER(c_float)()
    V_L2_N = POINTER(c_float)()
    V_L3_N = POINTER(c_float)()
    V_L4_N = POINTER(c_float)()
    V_L5_N = POINTER(c_float)()
    V_L6_N = POINTER(c_float)()
    V_L7_N = POINTER(c_float)()
    V_L8_N = POINTER(c_float)()
    V_L9_N = POINTER(c_float)()
    V_L10_N = POINTER(c_float)()
    V_L11_N = POINTER(c_float)()
    V_L12_N = POINTER(c_float)()
    V_L13_N = POINTER(c_float)()
    V_L14_N = POINTER(c_float)()
    V_L15_N = POINTER(c_float)()
    V_L16_N = POINTER(c_float)()

    V_L1_b = POINTER(c_float)()
    V_L2_b = POINTER(c_float)()
    V_L3_b = POINTER(c_float)()
    V_L4_b = POINTER(c_float)()
    V_L5_b = POINTER(c_float)()
    V_L6_b = POINTER(c_float)()
    V_L7_b = POINTER(c_float)()
    V_L8_b = POINTER(c_float)()
    V_L9_b = POINTER(c_float)()
    V_L10_b = POINTER(c_float)()
    V_L11_b = POINTER(c_float)()
    V_L12_b = POINTER(c_float)()
    V_L13_b = POINTER(c_float)()
    V_L14_b = POINTER(c_float)()
    V_L15_b = POINTER(c_float)()
    V_L16_b = POINTER(c_float)()

    V_L1_w = POINTER(c_float)()
    V_L2_w = POINTER(c_float)()
    V_L3_w = POINTER(c_float)()
    V_L4_w = POINTER(c_float)()
    V_L5_w = POINTER(c_float)()
    V_L6_w = POINTER(c_float)()
    V_L7_w = POINTER(c_float)()
    V_L8_w = POINTER(c_float)()
    V_L9_w = POINTER(c_float)()
    V_L10_w = POINTER(c_float)()
    V_L11_w = POINTER(c_float)()
    V_L12_w = POINTER(c_float)()
    V_L13_w = POINTER(c_float)()
    V_L14_w = POINTER(c_float)()
    V_L15_w = POINTER(c_float)()
    V_L16_w = POINTER(c_float)()

    V_L2_pool = POINTER(c_float)()
    V_L4_pool = POINTER(c_float)()
    V_L7_pool = POINTER(c_float)()
    V_L10_pool = POINTER(c_float)()
    V_L13_pool = POINTER(c_float)()
    V_Result_N = POINTER(c_float)()

    Vgg_host2gpu(V_L1_N,V_L2_N,V_L3_N,V_L4_N,
                    V_L5_N,V_L6_N,V_L7_N,V_L8_N,
                    V_L9_N,V_L10_N,V_L11_N,V_L12_N,
                    V_L13_N,V_L14_N,V_L15_N,V_L16_N,
                    V_L1_b,V_L2_b,V_L3_b,V_L4_b,
                    V_L5_b,V_L6_b,V_L7_b,V_L8_b,
                    V_L9_b,V_L10_b,V_L11_b,V_L12_b,
                    V_L13_b,V_L14_b,V_L15_b,V_L16_b,
                    V_L1_w,V_L2_w,V_L3_w,V_L4_w,
                    V_L5_w,V_L6_w,V_L7_w,V_L8_w,
                    V_L9_w,V_L10_w,V_L11_w,V_L12_w,
                    V_L13_w,V_L14_w,V_L15_w,V_L16_w,
                    V_L2_pool,V_L4_pool,V_L7_pool,V_L10_pool,
                    V_L13_pool,V_Result_N)

    # warm up
    for i in range(10):
        Vgg_merge_inference(V_L1_N,V_L2_N,V_L3_N,V_L4_N,
                            V_L5_N,V_L6_N,V_L7_N,V_L8_N,
                            V_L9_N,V_L10_N,V_L11_N,V_L12_N,
                            V_L13_N,V_L14_N,V_L15_N,V_L16_N,
                            V_L1_b,V_L2_b,V_L3_b,V_L4_b,
                            V_L5_b,V_L6_b,V_L7_b,V_L8_b,
                            V_L9_b,V_L10_b,V_L11_b,V_L12_b,
                            V_L13_b,V_L14_b,V_L15_b,V_L16_b,
                            V_L1_w,V_L2_w,V_L3_w,V_L4_w,
                            V_L5_w,V_L6_w,V_L7_w,V_L8_w,
                            V_L9_w,V_L10_w,V_L11_w,V_L12_w,
                            V_L13_w,V_L14_w,V_L15_w,V_L16_w,
                            V_L2_pool,V_L4_pool,V_L7_pool,V_L10_pool,
                            V_L13_pool,V_Result_N,1)

    stopping.value = stopping.value + 1

    print("Vgg Inference Ready!")
    while(stopping.value < 2):
        continue

    starttime = time.time()
    starttime_list.append(starttime)

    Vgg_merge_inference(V_L1_N,V_L2_N,V_L3_N,V_L4_N,
                        V_L5_N,V_L6_N,V_L7_N,V_L8_N,
                        V_L9_N,V_L10_N,V_L11_N,V_L12_N,
                        V_L13_N,V_L14_N,V_L15_N,V_L16_N,
                        V_L1_b,V_L2_b,V_L3_b,V_L4_b,
                        V_L5_b,V_L6_b,V_L7_b,V_L8_b,
                        V_L9_b,V_L10_b,V_L11_b,V_L12_b,
                        V_L13_b,V_L14_b,V_L15_b,V_L16_b,
                        V_L1_w,V_L2_w,V_L3_w,V_L4_w,
                        V_L5_w,V_L6_w,V_L7_w,V_L8_w,
                        V_L9_w,V_L10_w,V_L11_w,V_L12_w,
                        V_L13_w,V_L14_w,V_L15_w,V_L16_w,
                        V_L2_pool,V_L4_pool,V_L7_pool,V_L10_pool,
                        V_L13_pool,V_Result_N,1)

    endtime = time.time()
    endtime_list.append(endtime)
    print(f"Vgg16 inference time : {(endtime-starttime):.6f} seconds")

    Vgg_cudafree(V_L1_N,V_L2_N,V_L3_N,V_L4_N,
                V_L5_N,V_L6_N,V_L7_N,V_L8_N,
                V_L9_N,V_L10_N,V_L11_N,V_L12_N,
                V_L13_N,V_L14_N,V_L15_N,V_L16_N,
                V_L1_b,V_L2_b,V_L3_b,V_L4_b,
                V_L5_b,V_L6_b,V_L7_b,V_L8_b,
                V_L9_b,V_L10_b,V_L11_b,V_L12_b,
                V_L13_b,V_L14_b,V_L15_b,V_L16_b,
                V_L1_w,V_L2_w,V_L3_w,V_L4_w,
                V_L5_w,V_L6_w,V_L7_w,V_L8_w,
                V_L9_w,V_L10_w,V_L11_w,V_L12_w,
                V_L13_w,V_L14_w,V_L15_w,V_L16_w,
                V_L2_pool,V_L4_pool,V_L7_pool,V_L10_pool,
                V_L13_pool,V_Result_N)

stopping = Value('i', 0)  #for simultaneous starting of inferences between processes
producer_status = Value('i', 0)     #if request producer is done, turn into 1
finisher = Value('i', 0)     #if request producer is done, turn into 1
vgg_num = Value('i', 0)
window = Value('i', 0)

if __name__ == '__main__':

    user_request = Queue()
    count = Queue()

    manager = multiprocessing.Manager()
    starttime_list = manager.list()
    endtime_list = manager.list()

    children = []
   
    children.append(Process(target = Alex_Res_inference, args=(stopping, starttime_list, endtime_list)))
    children.append(Process(target = Vgg_merge_inference, args=(stopping, starttime_list, endtime_list)))

    for child in children:
        child.start()

    for child in children:
        child.join()

    starttime = min(starttime_list)
    endtime = max(endtime_list)
    total_inference_time = endtime-starttime
    
    print(f"total inferene time : {(total_inference_time):.6f}")

