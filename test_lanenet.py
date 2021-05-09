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
STEP=500
WIDTH =704 #512
HEIGHT =352 #256
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
save_path = "save_image_0301"
if not ops.exists(save_path):
    os.makedirs(save_path)

def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='The image path or the src image save dir')
    parser.add_argument('--weights_path', type=str, help='The model weights path')
    parser.add_argument('--is_batch', type=str, help='If test a batch of images', default='false')
    parser.add_argument('--batch_size', type=int, help='The batch size of the test images', default=32)
    parser.add_argument('--save_dir', type=str, help='Test result image save dir', default=None)
    parser.add_argument('--use_gpu', type=int, help='If use gpu set 1 or 0 instead', default=1)
    parser.add_argument('--phase' ,type = str, help='If test train list or val list',default='val')
    
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

def mean_iou_multiply(multi_seg_ret, labels):
    """
    Args:
        multi_seg_ret :     Tensor,    [batch_size , width , height ]
        labels:     Tensor,    [batch_size , width , height ]
    Returns:
        a mean iou between the predict and label
    """

    

    ###最后输出是multi_seg_ret  [batch=1 , width =256, height = 512 ]
    batch=multi_seg_ret.shape[0]
    shape=(multi_seg_ret.shape[1],multi_seg_ret.shape[2])
    
    ##label都转换成onehot格式 [1, 256, 512, 8]
    #muti_seg_onehot = tf.one_hot(indices=multi_seg_ret,depth=8,on_value=1,off_value=0)
    labels_onehot = tf.one_hot(indices=labels, depth= 8, on_value=1, off_value=0)
    
    ones=np.ones(shape=[batch,shape[0],shape[1]],dtype=np.float32)
    iou=[]
    pred_pos_list=[]
    label_pos_list=[]
    #tp_list=[]
    #计算每一类的IOU,背景即第0类不参与计算
    for i in range(0,labels_onehot.shape[-1]):
        #label中的value=1
        single_label=tf.cast(tf.reshape(labels_onehot[:,:,:,i],shape=(batch,shape[0],shape[1])),tf.float32)#[batch,height,width]
        #prediction中value=1,2,3,4,5,6,7
        single_pred=tf.cast(multi_seg_ret,tf.float32)#[batch,height,width]

        # 返回每个batch中预测正确的点的数目，结果为浮点数，例如batch=2时为[6.0,3.0],shape:(2,)
        mask_pred=tf.cast(tf.equal(single_pred,i*1.0),tf.float32)
        tp=tf.reduce_sum(tf.cast(tf.multiply(single_label,mask_pred), tf.float32),reduction_indices=[1,2])
        
        # 返回batch中预测第i类的所有点的数目，结果为浮点数，例如[4.0,3.0],shape:(2,)
        pred_pos = tf.reduce_sum(tf.cast(tf.cast(tf.multiply(mask_pred, ones), tf.bool),tf.float32), reduction_indices=[1, 2])
        pred_pos_list.append(pred_pos)

        # 返回batch大小的label中第i类的所有点的数目，结果为浮点数，例如[4.0,3.0],shape:(2,)
        label_pos = tf.reduce_sum(tf.cast(tf.cast(tf.multiply(single_label, ones),tf.bool), tf.float32), reduction_indices=[1, 2])
        label_pos_list.append(label_pos)

        fp = pred_pos - tp
        fn = label_pos - tp
        batch_iou=tf.reshape((tp)/(tp+fp+fn+0.00001),shape=(batch,1))#[batch,1]
        iou.append(batch_iou)        

    iou=tf.stack(iou,axis=0)#[num_classes,batch,1]
    iou=tf.transpose(tf.reshape(iou,iou.shape[:2]), perm=[1, 0]) #[batch,num_classes]
    #score_each_batch = tf.reduce_mean(iou, axis=1)#每个batch上的score,shape:[batch]
    iou_score=tf.reduce_mean(iou)

    return pred_pos_list,label_pos_list,iou

def test_lanenet(test_dataset, weights_path, use_gpu):
    """

    :param image_path:
    :param weights_path:
    :param use_gpu:
    :return:
    """
    #assert ops.exists(image_path), '{:s} not exist'.format(image_path)



    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, HEIGHT, WIDTH, 3], name='input_tensor')
    labels_tensor = tf.placeholder(dtype=tf.int64, shape = [1, 1710, 3384], name='labels_tensor')
    output_tensor = tf.placeholder(dtype=tf.int64, shape = [1, 1710, 3384], name='output_tensor')
    phase_tensor = tf.constant('test', tf.string)

    net = lanenet_merge_model.LaneNet(phase=phase_tensor, net_flag='vgg')
    binary_seg_ret, multi_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_model')

    pred_pos_list,label_pos_list,iou = mean_iou_multiply(output_tensor,labels_tensor)

    cluster = lanenet_cluster.LaneNetCluster()
    postprocessor = lanenet_postprocess.LaneNetPoseProcessor()

    saver = tf.train.Saver()

    # Set sess configuration
    if use_gpu:
        sess_config = tf.ConfigProto(device_count={'GPU': 0})
    else:
        sess_config = tf.ConfigProto(device_count={'CPU': 0})
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)  #

    with sess.as_default():

        saver.restore(sess=sess, save_path=weights_path)
        s_miou=[]
        m_miou=0
        accuracy_list=[[],[],[],[],[],[],[],[]]
        for i in range(STEP):
            img,_,instance_label= test_dataset.next_batch(1)

            log.info('开始读取图像数据并进行预处理')
            t_start = time.time()
            image_vis=img[0]
            image = [cv2.resize(tmp, ( WIDTH,HEIGHT), interpolation=cv2.INTER_LINEAR) for tmp in img]
            image = [tmp - VGG_MEAN for tmp in image]
            labels  =instance_label
            log.info('图像读取完毕, 耗时: {:.5f}s'.format(time.time() - t_start))

            t_start = time.time()
            binary_seg_image, multi_seg_image,  instance_seg_image = sess.run([binary_seg_ret,multi_seg_ret , instance_seg_ret],
                                                            feed_dict={input_tensor: image})
            
            resize_image = [cv2.resize(multi_seg_image[0] ,(3384, 1710),interpolation= cv2.INTER_NEAREST) ]

            pred_pos,label_pos ,miou_l = sess.run([pred_pos_list,label_pos_list,iou],feed_dict={output_tensor:resize_image,labels_tensor:labels})
            
            count=[]
            p_list=[]
            l_list=[]
            k=0
            for p_pos,l_pos,mmiou in zip(pred_pos,label_pos,miou_l[0]):
                #print("xxx",p_pos,"zzz",l_pos)
                p_list.append(p_pos[0])
                l_list.append(l_pos[0])
                if(p_pos[0]+l_pos[0]!=0.0):
                    accuracy_list[k].append(mmiou)
                    count.append(k)
                k=k+1
            miou = sum(miou_l[0])/len(count)
            t_cost = time.time() - t_start
            print("第 %d 张iou:"%(i+1),miou ,"\n",count,"\n",miou_l[0],"\n",p_list,"\n",l_list)

            s_miou.append(miou)
            log.info('单张图像车道线预测耗时: {:.5f}s'.format(t_cost))


            #binary_seg_image[0] = postprocessor.postprocess(binary_seg_image[0])
            #mask_image = cluster.get_lane_mask(binary_seg_ret=binary_seg_image[0],
            #                                   instance_seg_ret=instance_seg_image[0])
        
            multi_image =np.array( resize_image[0],np.uint8)
            multi_image_3 = np.zeros([multi_image.shape[0],multi_image.shape[1],3],np.uint8)
            multi_image_3[multi_image==0]=[0,0,0]
            multi_image_3[multi_image==1]=[70,130,180]
            multi_image_3[multi_image==2]=[0,0,142]
            multi_image_3[multi_image==3]=[153,153,153]
            multi_image_3[multi_image==4]=[128,64,128]
            multi_image_3[multi_image==5]=[190,153,153]
            multi_image_3[multi_image==6]=[128,128,0]
            multi_image_3[multi_image==7]=[255,128,0]

            mask_image = cv2.addWeighted(image_vis, 1.0, multi_image_3[:,:,(2,1,0)], 1.0, 0)
                    
        
            #for i in range(4):
            #    instance_seg_image[0][:, :, i] = minmax_scale(instance_seg_image[0][:, :, i])
            #embedding_image = np.array(instance_seg_image[0], np.uint8)

            #plt.ion()
            #plt.figure('mask_image')
            #plt.imshow(mask_image[:, :, (2, 1, 0)])
            #plt.figure('label_image')
            #plt.imshow(labels[0])
            #plt.figure('src_image')
            #plt.imshow(image_vis[:, :, (2, 1, 0)])
            #plt.figure('embedding_image')
            #plt.imshow(embedding_image[:, :, (2, 1, 0)])
            #plt.figure('binary_image')
            #plt.imshow(binary_seg_image[0] * 255, cmap='gray')
            #plt.figure('instance_image')
            #plt.imshow(multi_image_3)
            #plt.pause(0.5)
            
            if(miou<0.2 or miou>0.5):                           #True  #l_list[7]>1000 

                #src_path = ops.join(save_path,str(miou)+"src.jpg")
                lab_path = ops.join(save_path,str(miou)+"label.jpg")
                dst_path = ops.join(save_path,str(miou)+"dst.jpg")
                #plt.imsave(src_path,image_vis[:, :, (2, 1, 0)])
                plt.imsave(lab_path,labels[0])
                plt.imsave(dst_path,mask_image[:,:,(2,1,0)])
        m_miou = sum(s_miou)/STEP
        m_accuracy = [sum(accuracy)/len(accuracy) for accuracy in accuracy_list]
    sess.close()
    #print(s_miou)
    print("训练集随机%d张图片miou:"%STEP,m_miou,"\n每类精度:",m_accuracy,"\n类平均:",sum(m_accuracy)/len(m_accuracy))
    return



if __name__ == '__main__':
    # init args
    args = init_args()

    if args.phase =='val':
        train_dataset_file = 'E:/datasets/baidu/val.txt'
        label_dataset_file = 'E:/datasets/baidu/vallabel.txt'
    else:
        train_dataset_file = 'E:/datasets/baidu/train.txt'
        label_dataset_file = 'E:/datasets/baidu/label.txt'
    
    test_dataset = lanenet_data_processor.DataSet(train_dataset_file,label_dataset_file)


    if args.is_batch.lower() == 'false':
        test_lanenet(test_dataset, args.weights_path, args.use_gpu)
    

#python test_lanenet.py --weights_path=modeld2/baidu_lanenet_vgg_2019-02-25-14-49-15.ckpt-52000 --use_gpu=1

    


