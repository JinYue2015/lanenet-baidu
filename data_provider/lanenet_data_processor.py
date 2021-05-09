#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-11 下午4:58
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : lanenet_data_processor.py
# @IDE: PyCharm Community Edition
"""
实现LaneNet的数据解析类
"""
import os.path as ops

import cv2
import numpy as np
from PIL import Image
from skimage.transform import rotate
import random
try:
    from cv2 import cv2
except ImportError:
    pass


class DataSet(object):
    """
    实现数据集类
    """

    def __init__(self, dataset_info_file,label_file):
        """

        :param dataset_info_file:
        """
        self._gt_img_list, self._gt_label_list = self._init_dataset(dataset_info_file,label_file)
        self._random_dataset()
        self._next_batch_loop_count = 0

    def _init_dataset(self, dataset_info_file,label_file):
        """

        :param dataset_info_file:
        :return:
        """
        gt_img_list = []
        gt_label_list = []

        assert ops.exists(dataset_info_file), '{:s}　不存在'.format(dataset_info_file)
        assert ops.exists(label_file), '{:s}　不存在'.format(label_file)

        with open(dataset_info_file, 'r') as file:                
            #gt_img_list.append(str(_info) for _info in file)    
            for _info in file:
                info_tmp = _info.split('\n')
                gt_img_list.append(info_tmp[0])         
        with open(label_file, 'r') as file:
            for _info in file:
                info_tmp = _info.split('\n')
                gt_label_list.append(info_tmp[0]) 
        #print(gt_img_list,gt_label_list)
        #print(len(gt_img_list))
        return gt_img_list, gt_label_list

    def _random_dataset(self):
        """

        :return:
        """
        assert len(self._gt_img_list) == len(self._gt_label_list)

        random_idx = np.random.permutation(len(self._gt_img_list))
        new_gt_img_list = []
        new_gt_label_list = []

        for index in random_idx:
            new_gt_img_list.append(self._gt_img_list[index])
            new_gt_label_list.append(self._gt_label_list[index])

        self._gt_img_list = new_gt_img_list
        self._gt_label_list = new_gt_label_list
    
    def gamma_trans(self,img,gamma):
        gamma_table = [np.power(x/255.0,gamma)*255.0 for x in range(256)]
        gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
        return cv2.LUT(img,gamma_table)
    
    def next_batch(self, batch_size):
        """

        :param batch_size:
        :return:
        """
        assert len(self._gt_label_list)== len(self._gt_img_list)

        idx_start = batch_size * self._next_batch_loop_count
        idx_end = batch_size * self._next_batch_loop_count + batch_size

        if idx_end > len(self._gt_label_list):
            self._random_dataset()
            self._next_batch_loop_count = 0
            return self.next_batch(batch_size)
        else:
            gt_img_list = self._gt_img_list[idx_start:idx_end]
            gt_label_list = self._gt_label_list[idx_start:idx_end]

            gt_imgs = []
            gt_labels_binary = []
            gt_labels_instance = []
            #angle = random.uniform(-20,20)
            #r_c = random.choice([True, False])
            gamma = 0.7 #random.uniform(0.5,1.5)
            x = random.randint(0,846)
            y = random.randint(0,428)
            for gt_img_path in gt_img_list:
                img = cv2.imread(gt_img_path, cv2.IMREAD_COLOR)
                img = self.gamma_trans(img,gamma)
                img = img[y:y+1283,x:x+2538]
                #img = (rotate(img,angle)*255).astype('uint8') if r_c else img
                gt_imgs.append(img)
            for gt_label_path in gt_label_list:

                
                #二进制标签图
                label_pil_img = Image.open(gt_label_path)
                label_img = np.array(label_pil_img)

                label_binary = np.ones([label_img.shape[0], label_img.shape[1]], dtype=np.uint8)
                
                label_binary[label_img==0] =0
                label_binary[label_img==213] =0
                label_binary[label_img==206] =0
                label_binary[label_img==207] =0
                label_binary[label_img==211] =0
                label_binary[label_img==208] =0
                label_binary[label_img==216] =0
                label_binary[label_img==215] =0
                label_binary[label_img==218] =0
                label_binary[label_img==219] =0
                label_binary[label_img==232] =0
                label_binary[label_img==202] =0
                label_binary[label_img==231] =0
                label_binary[label_img==230] =0
                label_binary[label_img==228] =0
                label_binary[label_img==229] =0                
                label_binary[label_img==233] =0
                label_binary[label_img==212] =0
                label_binary[label_img==223] =0
                label_binary[label_img==249] =0
                label_binary[label_img==255] =0
                label_binary=label_binary[y:y+1283,x:x+2538]
                #label_binary = (rotate(label_binary,angle)*255).astype('uint8') if r_c else label_binary
                gt_labels_binary.append(label_binary)

                #实例分割标签图
                label_intance = np.array(label_pil_img)
                label_intance[label_img==213] =0
                label_intance[label_img==206] =0
                label_intance[label_img==207] =0
                label_intance[label_img==211] =0
                label_intance[label_img==208] =0
                label_intance[label_img==216] =0
                label_intance[label_img==215] =0
                label_intance[label_img==218] =0
                label_intance[label_img==219] =0
                label_intance[label_img==232] =0
                label_intance[label_img==202] =0
                label_intance[label_img==231] =0
                label_intance[label_img==230] =0
                label_intance[label_img==228] =0
                label_intance[label_img==229] =0 
                label_intance[label_img==233] =0
                label_intance[label_img==212] =0
                label_intance[label_img==223] =0
                label_intance[label_img==249] =0
                label_intance[label_img==255] =0
                
                label_intance[label_img==200]=1                
                label_intance[label_img==204]=1
                label_intance[label_img==209]=1
                label_intance[label_img==201]=2
                label_intance[label_img==203]=2
                label_intance[label_img==217]=3
                label_intance[label_img==210]=4
                label_intance[label_img==214]=5
                label_intance[label_img==220]=6
                label_intance[label_img==221]=6
                label_intance[label_img==222]=6
                label_intance[label_img==224]=6
                label_intance[label_img==225]=6
                label_intance[label_img==226]=6
                label_intance[label_img==205]=7
                label_intance[label_img==227]=7
                label_intance[label_img==250]=7
                label_intance = label_intance[y:y+1283,x:x+2538]
                #label_intance = (rotate(label_intance,angle)*255).astype('uint8') if r_c else label_intance
                gt_labels_instance.append(label_intance)

                #for i in range(len(self._label_ref)):
                #    for label in self._label_ref[i]:
               #         idx = np.where((label_img[:, :, :] == [label[2],label[1],label[0]]).all(axis=2)) #label RGB 转换 opencv BGR
               #         #print(label," ",idx)
                #        label_binary[idx[0],idx[1],i] = 1
               # gt_labels_binary.append(label_binary)
                
            self._next_batch_loop_count += 1
            return gt_imgs, gt_labels_binary, gt_labels_instance


if __name__ == '__main__':
    train_path=('E:/JY/dataset/baidu/train.txt')
    label_path =('E:/JY/dataset/baidu/label.txt')

    train_data_txt_path='C:/Users/SEU228/Desktop/lanenet-lane-detection/data/training_data_example/train.txt'
    train_data_label_txt_path='C:/Users/SEU228/Desktop/lanenet-lane-detection/data/training_data_example/train_label.txt'
    val = DataSet(train_path,label_path)

    a1, a2, a3 = val.next_batch(2)
    a4=a3[1]
    print(a4.shape)
    print(np.where(a4[:,:]!=0))
    #cv2.imwrite('test_binary_label.png', a2[0] * 255)
    #b1, b2, b3 = val.next_batch(50)
    #c1, c2, c3 = val.next_batch(50)
    #dd, d2, d3 = val.next_batch(50)
    