#!/usr/bin/python3
import cv2

def convert_image_to_text(in_path, out_path):
    img = cv2.imread(in_path)
    if img.shape != (224, 224, 3):
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LANCZOS4)

    #img.shape[0]: height, img.shape[1]: width, img.shape[2]: channel
    rgb_result=[]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rgb_result.append(img[i,j][2])
            rgb_result.append(img[i,j][1])
            rgb_result.append(img[i,j][0])
    print(len(rgb_result))
    out = open(out_path, "w")
    for num in rgb_result:
        out.write("%f\n"%num)
    print("Finished convert.")
    out.close()

if __name__ == '__main__':
    convert_image_to_text("tiger_shark.JPEG", "input_ts11.txt")