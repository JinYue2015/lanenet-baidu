#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-23 上午11:33
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : test_lanenet.py
# @IDE: PyCharm Community Edition
"""
测试LaneNet模型
"""
import os
import os.path as ops
import argparse
import time
import math
from PIL import Image
import tensorflow as tf
import glob
import glog as log
import numpy as np
import matplotlib.pyplot as plt
import cv2
try:
    from cv2 import cv2
except ImportError:
    pass

from lanenet_model import lanenet_merge_model
from lanenet_model import lanenet_cluster
from lanenet_model import lanenet_postprocess
from config import global_config
from data_provider import lanenet_data_processor

CFG = global_config.cfg
VGG_MEAN = [103.939, 116.779, 123.68]
STEP=100
WIDTH =704    #int(512)  704
HEIGHT =352    # int(256) 352
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='The image path or the src image save dir')
    parser.add_argument('--weights_path', type=str, help='The model weights path')
    parser.add_argument('--is_batch', type=str, help='If test a batch of images', default='true')
    parser.add_argument('--batch_size', type=int, help='The batch size of the test images', default=32)
    parser.add_argument('--save_dir', type=str, help='Test result image save dir', default=None)
    parser.add_argument('--use_gpu', type=int, help='If use gpu set 1 or 0 instead', default=1)

    return parser.parse_args()


def minmax_scale(input_arr):
    """

    :param input_arr:
    :return:
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr

def gamma_trans(img,gamma):
    gamma_table = [np.power(x/255.0,gamma)*255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    #print(gamma_table)
    return cv2.LUT(img,gamma_table)

def test_lanenet_batch(image_dir, weights_path, batch_size, use_gpu, save_dir=None):
    """

    :param image_dir:
    :param weights_path:
    :param batch_size:
    :param use_gpu:
    :param save_dir:
    :return:
    """
    assert ops.exists(image_dir), '{:s} not exist'.format(image_dir)

    log.info('开始获取图像文件路径...')
    image_path_list = glob.glob('{:s}/**/*.jpg'.format(image_dir), recursive=True) + \
                      glob.glob('{:s}/**/*.png'.format(image_dir), recursive=True) + \
                      glob.glob('{:s}/**/*.jpeg'.format(image_dir), recursive=True)

    input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, HEIGHT, WIDTH, 3], name='input_tensor')
    phase_tensor = tf.constant('test', tf.string)

    net = lanenet_merge_model.LaneNet(phase=phase_tensor, net_flag='vgg')
    binary_seg_ret, multi_seg_image,instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_model')


    saver = tf.train.Saver()

    # Set sess configuration
    if use_gpu:
        sess_config = tf.ConfigProto(device_count={'GPU': 1})
    else:
        sess_config = tf.ConfigProto(device_count={'GPU': 0})
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    with sess.as_default():

        saver.restore(sess=sess, save_path=weights_path)

        epoch_nums = int(math.ceil(len(image_path_list) / batch_size))
        for idx in range(1):
            save_dir=save_dir if idx ==0 else save_dir+"_nogamma"
            if not ops.exists(save_dir):
                os.makedirs(save_dir)
            for epoch in range(epoch_nums):
                log.info('[Epoch:{:d}] 开始图像读取和预处理...'.format(epoch))
                t_start = time.time()
                image_path_epoch = image_path_list[epoch * batch_size:(epoch + 1) * batch_size]
                image_list_epoch = [gamma_trans(cv2.imread(tmp, cv2.IMREAD_COLOR),0.7) if idx==0 else cv2.imread(tmp, cv2.IMREAD_COLOR) for tmp in image_path_epoch]
                image_vis_list = image_list_epoch
                image_list_epoch = [cv2.resize(tmp, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
                                    for tmp in image_list_epoch]
                image_list_epoch = [tmp - VGG_MEAN for tmp in image_list_epoch]
                t_cost = time.time() - t_start
                log.info('[Epoch:{:d}] 预处理{:d}张图像, 共耗时: {:.5f}s, 平均每张耗时: {:.5f}'.format(
                    epoch, len(image_path_epoch), t_cost, t_cost / len(image_path_epoch)))

                t_start = time.time()
                multi_images = sess.run(
                     multi_seg_image, feed_dict={input_tensor: image_list_epoch})
                t_cost = time.time() - t_start
                log.info('[Epoch:{:d}] 预测{:d}张图像车道线, 共耗时: {:.5f}s, 平均每张耗时: {:.5f}s'.format(
                    epoch, len(image_path_epoch), t_cost, t_cost / len(image_path_epoch)))

            
                for index, multi_image in enumerate(multi_images): 
                    resize_image = [cv2.resize(multi_image ,(3384, 1710),interpolation= cv2.INTER_NEAREST) ]
                    multi_image =np.array( resize_image[0],np.uint8)
                    #multi_image_3 = np.zeros([multi_image.shape[0],multi_image.shape[1],3],np.uint8)
                    #multi_image_3[multi_image==0]=[0,0,0]
                    #multi_image_3[multi_image==1]=[70,130,180]
                    #multi_image_3[multi_image==2]=[0,0,142]
                    #multi_image_3[multi_image==3]=[153,153,153]
                    #multi_image_3[multi_image==4]=[128,64,128]                
                    #multi_image_3[multi_image==5]=[190,153,153]
                    #multi_image_3[multi_image==6]=[128,128,0]
                    #multi_image_3[multi_image==7]=[255,128,0]

                    #mask_image = cv2.addWeighted(image_vis_list[index],1,multi_image_3[:,:,(2,1,0)],1,0)
                    #if save_dir is None:
                
                    #plt.ion()
                    #plt.figure('dst_image')
                    #plt.imshow(multi_image_3)
                    #plt.figure('src_image')
                    #plt.imshow(image_vis_list[index][:, :, (2, 1, 0)])
                    #plt.figure('mask_image')
                    #plt.imshow(mask_image[:,:,(2,1,0)])
                    #plt.pause(0.5)
                    #plt.show()
                    #plt.ioff()

                    result_image = np.zeros([multi_image.shape[0],multi_image.shape[1]],np.uint8)
                    result_image[multi_image==0] =0
                    result_image[multi_image==1] =200
                    result_image[multi_image==2] =201
                    result_image[multi_image==3] =217
                    result_image[multi_image==4] =210
                    result_image[multi_image==5] =214
                    result_image[multi_image==6] =220
                    result_image[multi_image==7] =205

                    if save_dir is not None:
                        image_name = ops.split(image_path_epoch[index])[1]
                        image_name = image_name.split('.')[0]+".png"
                        print(image_name)
                        image_save_path = ops.join(save_dir, image_name)
                        cv2.imwrite(image_save_path, result_image)
                        #mask_name = image_name.split('.')[0]+"mask.jpg"
                        #mask_save_path = ops.join(save_dir,mask_name)
                        #cv2.imwrite(mask_save_path,mask_image)

    sess.close()

    return


if __name__ == '__main__':
    # init args
    args = init_args()


    train_dataset_file = 'E:/datasets/baidu/val.txt'
    label_dataset_file = 'E:/datasets/baidu/vallabel.txt'
    test_dataset = lanenet_data_processor.DataSet(train_dataset_file,label_dataset_file)

    #if args.save_dir is not None and not ops.exists(args.save_dir):
    #    log.error('{:s} not exist and has been made'.format(args.save_dir))
    #    os.makedirs(args.save_dir)

    if args.is_batch.lower() == 'false':
        # test hnet model on single image
        test_lanenet(test_dataset, args.weights_path, args.use_gpu)
    
    else:
        # test hnet model on a batch of image
        test_lanenet_batch(image_dir=args.image_path, weights_path=args.weights_path,
                          save_dir=args.save_dir, use_gpu=args.use_gpu, batch_size=args.batch_size)
    
#python test_result.py --weights_path=modeld2/baidu_lanenet_vgg_2019-03-05-15-05-35.ckpt-50000 --image_path=E:/datasets/baidu/ColorImage_TestSet --save_dir=result --use_gpu=1 --batch_size=4


    


