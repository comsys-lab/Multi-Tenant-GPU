#!/usr/bin/python3

# Before run this code, 
# Execute '/usr/local/cuda/bin/nvcc -Xcompiler -fPIC -shared -o aras_models.so aras_models.cu' first.
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

def alexnet_inference(stopping, starttime_list, endtime_list):
    pid = os.getpid()
    print('Alexnet Inference PID : {0}'.format(pid))

    dll = ctypes.CDLL('./aras_models.so', mode=ctypes.RTLD_GLOBAL)
    host2gpu_alexnet = dll.host2gpu_alexnet
    host2gpu_alexnet.argtypes = [POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                    POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                    POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                    POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                    POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                    POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                    POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                    POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float))]

    alex_first_conv = dll.alex_first_conv
    alex_first_conv.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float)]
    alex_fisrt_norm = dll.alex_fisrt_norm
    alex_fisrt_norm.argtypes = [POINTER(c_float),POINTER(c_float)]
    alex_first_pool = dll.alex_first_pool
    alex_first_pool.argtypes = [POINTER(c_float),POINTER(c_float)]

    alex_second_conv = dll.alex_second_conv
    alex_second_conv.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float)]
    alex_second_norm = dll.alex_second_norm
    alex_second_norm.argtypes = [POINTER(c_float),POINTER(c_float)]
    alex_second_pool = dll.alex_second_pool
    alex_second_pool.argtypes = [POINTER(c_float),POINTER(c_float)]

    alex_third_conv = dll.alex_third_conv
    alex_third_conv.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float)]

    alex_fourth_conv = dll.alex_fourth_conv
    alex_fourth_conv.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float)]

    alex_fifth_conv = dll.alex_fifth_conv
    alex_fifth_conv.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float)]
    alex_fifth_pool = dll.alex_fifth_pool
    alex_fifth_pool.argtypes = [POINTER(c_float),POINTER(c_float)]

    alex_first_fc = dll.alex_first_fc
    alex_first_fc.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float)]

    alex_second_fc = dll.alex_second_fc
    alex_second_fc.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float)]

    alex_third_fc = dll.alex_third_fc
    alex_third_fc.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float)]

    free_alexnet = dll.free_alexnet
    free_alexnet.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                            POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                            POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                            POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                            POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                            POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                            POINTER(c_float),POINTER(c_float),POINTER(c_float),
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

    host2gpu_alexnet(A_L1_N,A_L2_N,A_L3_N,A_L4_N,
                    A_L5_N,A_L6_N,A_L7_N,A_L8_N,
                    A_L1_b,A_L2_b,A_L3_b,A_L4_b,
                    A_L5_b,A_L6_b,A_L7_b,A_L8_b,
                    A_L1_w,A_L2_w,A_L3_w,A_L4_w,
                    A_L5_w,A_L6_w,A_L7_w,A_L8_w,
                    A_L1_pool,A_L2_pool,A_L5_pool,
                    A_L1_norm,A_L2_norm,A_Result_N)

    for i in range(5):
        alex_first_conv(A_L1_b,A_L1_N,A_L1_w,A_L1_norm)
        alex_fisrt_norm(A_L1_norm,A_L1_pool)
        alex_first_pool(A_L1_pool,A_L2_N)
        alex_second_conv(A_L2_b,A_L2_N,A_L2_w,A_L2_norm)
        alex_second_norm(A_L2_norm,A_L2_pool)
        alex_second_pool(A_L2_pool,A_L3_N)
        alex_third_conv(A_L3_b,A_L3_N,A_L3_w,A_L4_N)
        alex_fourth_conv(A_L4_b,A_L4_N,A_L4_w,A_L5_N)
        alex_fifth_conv(A_L5_b,A_L5_N,A_L5_w,A_L5_pool)
        alex_fifth_pool(A_L5_pool,A_L6_N)
        alex_first_fc(A_L6_b,A_L6_N,A_L6_w,A_L7_N)
        alex_second_fc(A_L7_b,A_L7_N,A_L7_w,A_L8_N)
        alex_third_fc(A_L8_b,A_L8_N,A_L8_w,A_Result_N)
    
    stopping.value = stopping.value + 1

    print("Alexnet Ready!")
    while(stopping.value < 3):
        continue
        
    starttime = time.time()
    starttime_list.append(starttime)

    alex_first_conv(A_L1_b,A_L1_N,A_L1_w,A_L1_norm)
    alex_fisrt_norm(A_L1_norm,A_L1_pool)
    alex_first_pool(A_L1_pool,A_L2_N)
    alex_second_conv(A_L2_b,A_L2_N,A_L2_w,A_L2_norm)
    alex_second_norm(A_L2_norm,A_L2_pool)
    alex_second_pool(A_L2_pool,A_L3_N)
    alex_third_conv(A_L3_b,A_L3_N,A_L3_w,A_L4_N)
    alex_fourth_conv(A_L4_b,A_L4_N,A_L4_w,A_L5_N)
    alex_fifth_conv(A_L5_b,A_L5_N,A_L5_w,A_L5_pool)
    alex_fifth_pool(A_L5_pool,A_L6_N)
    alex_first_fc(A_L6_b,A_L6_N,A_L6_w,A_L7_N)
    alex_second_fc(A_L7_b,A_L7_N,A_L7_w,A_L8_N)
    alex_third_fc(A_L8_b,A_L8_N,A_L8_w,A_Result_N)

    endtime = time.time()
    endtime_list.append(endtime)
    print(f"total Alexnet inference time : {(endtime-starttime):.6f} seconds")

    free_alexnet(A_L1_N,A_L2_N,A_L3_N,A_L4_N,
                    A_L5_N,A_L6_N,A_L7_N,A_L8_N,
                    A_L1_b,A_L2_b,A_L3_b,A_L4_b,
                    A_L5_b,A_L6_b,A_L7_b,A_L8_b,
                    A_L1_w,A_L2_w,A_L3_w,A_L4_w,
                    A_L5_w,A_L6_w,A_L7_w,A_L8_w,
                    A_L1_pool,A_L2_pool,A_L5_pool,
                    A_L1_norm,A_L2_norm,A_Result_N)

################################################################################################

def resnet18_inference(stopping, starttime_list, endtime_list):
    pid = os.getpid()
    print('Resnet18 Inference PID : {0}'.format(pid))

    dll = ctypes.CDLL('./aras_models.so', mode=ctypes.RTLD_GLOBAL)
    host2gpu_resnet18 = dll.host2gpu_resnet18
    host2gpu_resnet18.argtypes = [POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
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

    res_first_conv = dll.res_first_conv
    res_first_conv.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float)]

    res_first_pool = dll.res_first_pool
    res_first_pool.argtypes = [POINTER(c_float),POINTER(c_float)]

    res_second_conv = dll.res_second_conv
    res_second_conv.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float)]

    res_third_conv = dll.res_third_conv
    res_third_conv.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float)]

    res_third_basic = dll.res_third_basic
    res_third_basic.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float)]

    res_fourth_conv = dll.res_fourth_conv
    res_fourth_conv.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float)]

    res_fifth_conv = dll.res_fifth_conv
    res_fifth_conv.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float)]

    res_fifth_basic = dll.res_fifth_basic
    res_fifth_basic.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float)]

    res_sixth_conv = dll.res_sixth_conv
    res_sixth_conv.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float)]

    res_seventh_conv = dll.res_seventh_conv
    res_seventh_conv.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float)]

    res_Block_B_conv = dll.res_Block_B_conv
    res_Block_B_conv.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float)]

    res_Block_B_basic = dll.res_Block_B_basic
    res_Block_B_basic.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float)]

    res_eighth_conv = dll.res_eighth_conv
    res_eighth_conv.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float)]

    res_ninth_conv = dll.res_ninth_conv
    res_ninth_conv.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float)]

    res_ninth_basic = dll.res_ninth_basic
    res_ninth_basic.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float)]

    res_tenth_conv = dll.res_tenth_conv
    res_tenth_conv.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float)]

    res_eleventh_conv = dll.res_eleventh_conv
    res_eleventh_conv.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float)]

    res_Block_C_conv = dll.res_Block_C_conv
    res_Block_C_conv.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float)]

    res_Block_C_basic = dll.res_Block_C_basic
    res_Block_C_basic.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float)]

    res_twelfth_conv = dll.res_twelfth_conv
    res_twelfth_conv.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float)]

    res_thirteenth_conv = dll.res_thirteenth_conv
    res_thirteenth_conv.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float)]

    res_thirteenth_basic = dll.res_thirteenth_basic
    res_thirteenth_basic.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float)]

    res_fourteenth_conv = dll.res_fourteenth_conv
    res_fourteenth_conv.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float)]

    res_fifteenth_conv = dll.res_fifteenth_conv
    res_fifteenth_conv.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float)]

    res_Block_D_conv = dll.res_Block_D_conv
    res_Block_D_conv.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float)]

    res_Block_D_basic = dll.res_Block_D_basic
    res_Block_D_basic.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float)]

    res_sixteenth_conv = dll.res_sixteenth_conv
    res_sixteenth_conv.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float)]

    res_seventeenth_conv = dll.res_seventeenth_conv
    res_seventeenth_conv.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float)]

    res_seventeenth_basic = dll.res_seventeenth_basic
    res_seventeenth_basic.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float)]

    res_avg_pool = dll.res_avg_pool
    res_avg_pool.argtypes = [POINTER(c_float),POINTER(c_float)]

    res_fc = dll.res_fc
    res_fc.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float)]

    free_resnet18 = dll.free_resnet18
    free_resnet18.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
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

    host2gpu_resnet18(R_L1_N,R_L2_N,R_L3_N,R_L4_N,
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

    #warm up
    for i in range(5):
        res_first_conv(R_L1_N,R_L1_w,R_L1_bn,R_L1_pool,R_L1_M,R_L1_V,R_L1_G,R_L1_B)
        res_first_pool(R_L1_pool,R_L2_N)
        res_second_conv(R_L2_N,R_L2_w,R_L2_bn,R_L3_N,R_L2_M,R_L2_V,R_L2_G,R_L2_B)
        res_third_conv(R_L3_N,R_L3_w,R_L3_bn,R_L3_basic,R_L3_M,R_L3_V,R_L3_G,R_L3_B)
        res_third_basic(R_L2_N,R_L3_basic,R_L4_N)
        res_fourth_conv(R_L4_N,R_L4_w,R_L4_bn,R_L5_N,R_L4_M,R_L4_V,R_L4_G,R_L4_B)
        res_fifth_conv(R_L5_N,R_L5_w,R_L5_bn,R_L5_basic,R_L5_M,R_L5_V,R_L5_G,R_L5_B)
        res_fifth_basic(R_L4_N,R_L5_basic,R_L6_N)
        res_sixth_conv(R_L6_N,R_L6_w,R_L6_bn,R_L7_N,R_L6_M,R_L6_V,R_L6_G,R_L6_B)
        res_seventh_conv(R_L7_N,R_L7_w,R_L7_bn,R_L7_basic,R_L7_M,R_L7_V,R_L7_G,R_L7_B)
        res_Block_B_conv(R_L6_N,R_B3_w,R_B3_bn,R_B3_basic,R_B3_M,R_B3_V,R_B3_G,R_B3_B)
        res_Block_B_basic(R_L7_basic,R_B3_basic,R_L8_N)
        res_eighth_conv(R_L8_N,R_L8_w,R_L8_bn,R_L9_N,R_L8_M,R_L8_V,R_L8_G,R_L8_B)
        res_ninth_conv(R_L9_N,R_L9_w,R_L9_bn,R_L9_basic,R_L9_M,R_L9_V,R_L9_G,R_L9_B)
        res_ninth_basic(R_L8_N,R_L9_basic,R_L10_N)
        res_tenth_conv(R_L10_N,R_L10_w,R_L10_bn,R_L11_N,R_L10_M,R_L10_V,R_L10_G,R_L10_B)
        res_eleventh_conv(R_L11_N,R_L11_w,R_L11_bn,R_L11_basic,R_L11_M,R_L11_V,R_L11_G,R_L11_B)
        res_Block_C_conv(R_L10_N,R_B4_w,R_B4_bn,R_B4_basic,R_B4_M,R_B4_V,R_B4_G,R_B4_B)
        res_Block_C_basic(R_L11_basic,R_B4_basic,R_L12_N)
        res_twelfth_conv(R_L12_N,R_L12_w,R_L12_bn,R_L13_N,R_L12_M,R_L12_V,R_L12_G,R_L12_B)
        res_thirteenth_conv(R_L13_N,R_L13_w,R_L13_bn,R_L13_basic,R_L13_M,R_L13_V,R_L13_G,R_L13_B)
        res_thirteenth_basic(R_L12_N,R_L13_basic,R_L14_N)
        res_fourteenth_conv(R_L14_N,R_L14_w,R_L14_bn,R_L15_N,R_L14_M,R_L14_V,R_L14_G,R_L14_B)
        res_fifteenth_conv(R_L15_N,R_L15_w,R_L15_bn,R_L15_basic,R_L15_M,R_L15_V,R_L15_G,R_L15_B)
        res_Block_D_conv(R_L14_N,R_B5_w,R_B5_bn,R_B5_basic,R_B5_M,R_B5_V,R_B5_G,R_B5_B)
        res_Block_D_basic(R_L15_basic,R_B5_basic,R_L16_N)
        res_sixteenth_conv(R_L16_N,R_L16_w,R_L16_bn,R_L17_N,R_L16_M,R_L16_V,R_L16_G,R_L16_B)
        res_seventeenth_conv(R_L17_N,R_L17_w,R_L17_bn,R_L17_basic,R_L17_M,R_L17_V,R_L17_G,R_L17_B)
        res_seventeenth_basic(R_L16_N,R_L17_basic,R_L18_N)
        res_avg_pool(R_L18_N,R_FC_N)
        res_fc(R_FC_b,R_FC_N,R_FC_w,R_Result_N)

    stopping.value = stopping.value + 1

    print("Resnet18 Ready!")
    while(stopping.value < 3):
        continue

    starttime = time.time()
    starttime_list.append(starttime)

    res_first_conv(R_L1_N,R_L1_w,R_L1_bn,R_L1_pool,R_L1_M,R_L1_V,R_L1_G,R_L1_B)
    res_first_pool(R_L1_pool,R_L2_N)
    res_second_conv(R_L2_N,R_L2_w,R_L2_bn,R_L3_N,R_L2_M,R_L2_V,R_L2_G,R_L2_B)
    res_third_conv(R_L3_N,R_L3_w,R_L3_bn,R_L3_basic,R_L3_M,R_L3_V,R_L3_G,R_L3_B)
    res_third_basic(R_L2_N,R_L3_basic,R_L4_N)
    res_fourth_conv(R_L4_N,R_L4_w,R_L4_bn,R_L5_N,R_L4_M,R_L4_V,R_L4_G,R_L4_B)
    res_fifth_conv(R_L5_N,R_L5_w,R_L5_bn,R_L5_basic,R_L5_M,R_L5_V,R_L5_G,R_L5_B)
    res_fifth_basic(R_L4_N,R_L5_basic,R_L6_N)
    res_sixth_conv(R_L6_N,R_L6_w,R_L6_bn,R_L7_N,R_L6_M,R_L6_V,R_L6_G,R_L6_B)
    res_seventh_conv(R_L7_N,R_L7_w,R_L7_bn,R_L7_basic,R_L7_M,R_L7_V,R_L7_G,R_L7_B)
    res_Block_B_conv(R_L6_N,R_B3_w,R_B3_bn,R_B3_basic,R_B3_M,R_B3_V,R_B3_G,R_B3_B)
    res_Block_B_basic(R_L7_basic,R_B3_basic,R_L8_N)
    res_eighth_conv(R_L8_N,R_L8_w,R_L8_bn,R_L9_N,R_L8_M,R_L8_V,R_L8_G,R_L8_B)
    res_ninth_conv(R_L9_N,R_L9_w,R_L9_bn,R_L9_basic,R_L9_M,R_L9_V,R_L9_G,R_L9_B)
    res_ninth_basic(R_L8_N,R_L9_basic,R_L10_N)
    res_tenth_conv(R_L10_N,R_L10_w,R_L10_bn,R_L11_N,R_L10_M,R_L10_V,R_L10_G,R_L10_B)
    res_eleventh_conv(R_L11_N,R_L11_w,R_L11_bn,R_L11_basic,R_L11_M,R_L11_V,R_L11_G,R_L11_B)
    res_Block_C_conv(R_L10_N,R_B4_w,R_B4_bn,R_B4_basic,R_B4_M,R_B4_V,R_B4_G,R_B4_B)
    res_Block_C_basic(R_L11_basic,R_B4_basic,R_L12_N)
    res_twelfth_conv(R_L12_N,R_L12_w,R_L12_bn,R_L13_N,R_L12_M,R_L12_V,R_L12_G,R_L12_B)
    res_thirteenth_conv(R_L13_N,R_L13_w,R_L13_bn,R_L13_basic,R_L13_M,R_L13_V,R_L13_G,R_L13_B)
    res_thirteenth_basic(R_L12_N,R_L13_basic,R_L14_N)
    res_fourteenth_conv(R_L14_N,R_L14_w,R_L14_bn,R_L15_N,R_L14_M,R_L14_V,R_L14_G,R_L14_B)
    res_fifteenth_conv(R_L15_N,R_L15_w,R_L15_bn,R_L15_basic,R_L15_M,R_L15_V,R_L15_G,R_L15_B)
    res_Block_D_conv(R_L14_N,R_B5_w,R_B5_bn,R_B5_basic,R_B5_M,R_B5_V,R_B5_G,R_B5_B)
    res_Block_D_basic(R_L15_basic,R_B5_basic,R_L16_N)
    res_sixteenth_conv(R_L16_N,R_L16_w,R_L16_bn,R_L17_N,R_L16_M,R_L16_V,R_L16_G,R_L16_B)
    res_seventeenth_conv(R_L17_N,R_L17_w,R_L17_bn,R_L17_basic,R_L17_M,R_L17_V,R_L17_G,R_L17_B)
    res_seventeenth_basic(R_L16_N,R_L17_basic,R_L18_N)
    res_avg_pool(R_L18_N,R_FC_N)
    res_fc(R_FC_b,R_FC_N,R_FC_w,R_Result_N)

    endtime = time.time()
    endtime_list.append(endtime)
    print(f"total Resnet18 inference time : {(endtime-starttime):.6f} seconds")
        
    free_resnet18(R_L1_N,R_L2_N,R_L3_N,R_L4_N,
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
################################################################################################

def vgg16_inference(stopping, starttime_list, endtime_list):
    pid = os.getpid()
    print('Vgg16 Inference PID : {0}'.format(pid))

    dll = ctypes.CDLL('./aras_models.so', mode=ctypes.RTLD_GLOBAL)
    host2gpu_vgg16 = dll.host2gpu_vgg16
    host2gpu_vgg16.argtypes = [POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
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

    vgg_first_conv = dll.vgg_first_conv
    vgg_first_conv.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float)]

    vgg_second_conv = dll.vgg_second_conv
    vgg_second_conv.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float)]

    vgg_second_pool = dll.vgg_second_pool
    vgg_second_pool.argtypes = [POINTER(c_float),POINTER(c_float)]

    vgg_third_conv = dll.vgg_third_conv
    vgg_third_conv.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float)]

    vgg_fourth_conv = dll.vgg_fourth_conv
    vgg_fourth_conv.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float)]

    vgg_fourth_pool = dll.vgg_fourth_pool
    vgg_fourth_pool.argtypes = [POINTER(c_float),POINTER(c_float)]

    vgg_fifth_conv = dll.vgg_fifth_conv
    vgg_fifth_conv.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float)]

    vgg_sixth_conv = dll.vgg_sixth_conv
    vgg_sixth_conv.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float)]

    vgg_seventh_conv = dll.vgg_seventh_conv
    vgg_seventh_conv.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float)]

    vgg_seventh_pool = dll.vgg_seventh_pool
    vgg_seventh_pool.argtypes = [POINTER(c_float),POINTER(c_float)]

    vgg_eighth_conv = dll.vgg_eighth_conv
    vgg_eighth_conv.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float)]

    vgg_ninth_conv = dll.vgg_ninth_conv
    vgg_ninth_conv.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float)]

    vgg_tenth_conv = dll.vgg_tenth_conv
    vgg_tenth_conv.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float)]

    vgg_tenth_pool = dll.vgg_tenth_pool
    vgg_tenth_pool.argtypes = [POINTER(c_float),POINTER(c_float)]

    vgg_eleventh_conv = dll.vgg_eleventh_conv
    vgg_eleventh_conv.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float)]

    vgg_twelfth_conv = dll.vgg_twelfth_conv
    vgg_twelfth_conv.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float)]

    vgg_thirteenth_conv = dll.vgg_thirteenth_conv
    vgg_thirteenth_conv.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float)]

    vgg_thirteenth_pool = dll.vgg_thirteenth_pool
    vgg_thirteenth_pool.argtypes = [POINTER(c_float),POINTER(c_float)]

    vgg_first_fc = dll.vgg_first_fc
    vgg_first_fc.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float)]

    vgg_second_fc = dll.vgg_second_fc
    vgg_second_fc.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float)]

    vgg_third_fc = dll.vgg_third_fc
    vgg_third_fc.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float)]

    free_vgg16 = dll.free_vgg16
    free_vgg16.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
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

    host2gpu_vgg16(V_L1_N,V_L2_N,V_L3_N,V_L4_N,
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

    #warm up
    for i in range(5):
        vgg_first_conv(V_L1_b,V_L1_N,V_L1_w,V_L2_N)
        vgg_second_conv(V_L2_b,V_L2_N,V_L2_w,V_L2_pool)
        vgg_second_pool(V_L2_pool,V_L3_N)
        vgg_third_conv(V_L3_b,V_L3_N,V_L3_w,V_L4_N)
        vgg_fourth_conv(V_L4_b,V_L4_N,V_L4_w,V_L4_pool)
        vgg_fourth_pool(V_L4_pool,V_L5_N)
        vgg_fifth_conv(V_L5_b,V_L5_N,V_L5_w,V_L6_N)
        vgg_sixth_conv(V_L6_b,V_L6_N,V_L6_w,V_L7_N)
        vgg_seventh_conv(V_L7_b,V_L7_N,V_L7_w,V_L7_pool)
        vgg_seventh_pool(V_L7_pool,V_L8_N)
        vgg_eighth_conv(V_L8_b,V_L8_N,V_L8_w,V_L9_N)
        vgg_ninth_conv(V_L9_b,V_L9_N,V_L9_w,V_L10_N)
        vgg_tenth_conv(V_L10_b,V_L10_N,V_L10_w,V_L10_pool)
        vgg_tenth_pool(V_L10_pool,V_L11_N)
        vgg_eleventh_conv(V_L11_b,V_L11_N,V_L11_w,V_L12_N)
        vgg_twelfth_conv(V_L12_b,V_L12_N,V_L12_w,V_L13_N)
        vgg_thirteenth_conv(V_L13_b,V_L13_N,V_L13_w,V_L13_pool)
        vgg_thirteenth_pool(V_L13_pool,V_L14_N)
        vgg_first_fc(V_L14_b,V_L14_N,V_L14_w,V_L15_N)
        vgg_second_fc(V_L15_b,V_L15_N,V_L15_w,V_L16_N)
        vgg_third_fc(V_L16_b,V_L16_N,V_L16_w,V_Result_N)

    stopping.value = stopping.value + 1
    
    print("Vgg16 Ready!")
    while(stopping.value < 3):
        continue

    starttime = time.time()
    starttime_list.append(starttime)

    vgg_first_conv(V_L1_b,V_L1_N,V_L1_w,V_L2_N)
    vgg_second_conv(V_L2_b,V_L2_N,V_L2_w,V_L2_pool)
    vgg_second_pool(V_L2_pool,V_L3_N)
    vgg_third_conv(V_L3_b,V_L3_N,V_L3_w,V_L4_N)
    vgg_fourth_conv(V_L4_b,V_L4_N,V_L4_w,V_L4_pool)
    vgg_fourth_pool(V_L4_pool,V_L5_N)
    vgg_fifth_conv(V_L5_b,V_L5_N,V_L5_w,V_L6_N)
    vgg_sixth_conv(V_L6_b,V_L6_N,V_L6_w,V_L7_N)
    vgg_seventh_conv(V_L7_b,V_L7_N,V_L7_w,V_L7_pool)
    vgg_seventh_pool(V_L7_pool,V_L8_N)
    vgg_eighth_conv(V_L8_b,V_L8_N,V_L8_w,V_L9_N)
    vgg_ninth_conv(V_L9_b,V_L9_N,V_L9_w,V_L10_N)
    vgg_tenth_conv(V_L10_b,V_L10_N,V_L10_w,V_L10_pool)
    vgg_tenth_pool(V_L10_pool,V_L11_N)
    vgg_eleventh_conv(V_L11_b,V_L11_N,V_L11_w,V_L12_N)
    vgg_twelfth_conv(V_L12_b,V_L12_N,V_L12_w,V_L13_N)
    vgg_thirteenth_conv(V_L13_b,V_L13_N,V_L13_w,V_L13_pool)
    vgg_thirteenth_pool(V_L13_pool,V_L14_N)
    vgg_first_fc(V_L14_b,V_L14_N,V_L14_w,V_L15_N)
    vgg_second_fc(V_L15_b,V_L15_N,V_L15_w,V_L16_N)
    vgg_third_fc(V_L16_b,V_L16_N,V_L16_w,V_Result_N)

    endtime = time.time()
    endtime_list.append(endtime)
    print(f"total Vgg16 inference time : {(endtime-starttime):.6f} seconds")

    free_vgg16(V_L1_N,V_L2_N,V_L3_N,V_L4_N,
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

################################################################################################

stopping = Value('i', 0)  #for simultaneous starting of inferences between processes
pointer_value1 = Value('i', 0)
pointer_value2 = Value('i', 0)
pointer_value3 = Value('i', 0)

if __name__ == '__main__':
 
    manager = multiprocessing.Manager()
    starttime_list = manager.list()
    endtime_list = manager.list()

    children = []

    children.append(Process(target=alexnet_inference, args=(stopping, starttime_list, endtime_list)))
    children.append(Process(target=resnet18_inference, args=(stopping, starttime_list, endtime_list)))
    children.append(Process(target=vgg16_inference, args=(stopping, starttime_list, endtime_list)))

    for child in children:
        child.start()

    for child in children:
        child.join()

    starttime = min(starttime_list)
    endtime = max(endtime_list)
    total_inference_time = endtime-starttime

    print(f"total inferene time : {(total_inference_time):.6f}")
