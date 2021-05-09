import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import six


def get_files(files_dir):
    files_list = []      
    for file in os.listdir(files_dir):         
        files_list.append(file[:])
    return  files_list

files_dir = 'C:/Users/jy101/Desktop/lanenet-lane-detection/test/test21/test/result/'
files_list = get_files(files_dir) 
for i in range(len(files_list)):
    print(i+1," ",files_list[i])
    img = cv2.imread(files_dir + files_list[i] ,0 )
    kernel = np.ones((15,15),np.uint8) 
    res =  cv2.morphologyEx(img,cv2.MORPH_CLOSE, kernel)
    cv2.imwrite("./Close/" + files_list[i], res)  #, [int(cv2.IMWRITE_PNG_COMPRESSION), 0]

