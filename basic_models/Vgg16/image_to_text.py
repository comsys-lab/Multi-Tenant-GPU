#!/usr/bin/python3
import numpy as np
import cv2

def convert_image_to_text(in_path, out_path):
    img = cv2.imread(in_path)
    if img.shape != (224, 224, 3):
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LANCZOS4)

    # img.shape[0]: height, img.shape[1]: width, img.shape[2]: channel
    rgb_result=[]
    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         for k in range(img.shape[2]):
    #             rgb_result.append(img[i,j,k])

    for i in range(img.shape[2]):
        for j in range(img.shape[1]):
            for k in range(img.shape[0]):
                rgb_result.append(img[k,j,i])
    
    print(img.shape[0],img.shape[1],img.shape[2])
  
    # rgb_arrange1 = []
    # rgb_arrange2 = []
    # rgb_arrange3 = []
    # for i in range(len(rgb_result)):
    #     if i % 3 == 0:
    #         rgb_arrange1.append(rgb_result[i])
    #     elif i % 3 == 1:
    #         rgb_arrange2.append(rgb_result[i])
    #     else:
    #         rgb_arrange3.append(rgb_result[i])

    # rgb_arrange = rgb_arrange1 + rgb_arrange2 + rgb_arrange3

    out = open(out_path, "w")
    for num in rgb_result:
        out.write("%f\n"%num)
    print("Finished convert.")
    out.close()

if __name__ == '__main__':
    convert_image_to_text("data_pytorch/cat.jpg", "data_pytorch/input_cat3.txt")