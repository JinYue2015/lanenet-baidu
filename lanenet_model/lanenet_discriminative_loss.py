#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-11 下午3:48
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : lanenet_discriminative_loss.py
# @IDE: PyCharm Community Edition
"""
实现LaneNet的Discriminative Loss函数
"""
import tensorflow as tf


def discriminative_loss_single(
        prediction,
        correct_label,
        feature_dim,
        label_shape,
        delta_v,
        delta_d,
        param_var,
        param_dist,
        param_reg):
    """
    论文equ(1)提到的实例分割损失函数
    :param prediction: inference of network
    :param correct_label: instance label
    :param feature_dim: feature dimension of prediction
    :param label_shape: shape of label
    :param delta_v: cut off variance distance
    :param delta_d: cut off cluster distance
    :param param_var: weight for intra cluster variance
    :param param_dist: weight for inter cluster distances
    :param param_reg: weight regularization
    """

    # 像素对齐为一行
    correct_label = tf.reshape(
        correct_label, [
            label_shape[1] * label_shape[0]])
    reshaped_pred = tf.reshape(
        prediction, [
            label_shape[1] * label_shape[0], feature_dim])

    # 统计实例个数
    unique_labels, unique_id, counts = tf.unique_with_counts(correct_label)
    counts = tf.cast(counts, tf.float32)
    num_instances = tf.size(unique_labels)

    # 计算pixel embedding均值向量
    segmented_sum = tf.unsorted_segment_sum(
        reshaped_pred, unique_id, num_instances)
    mu = tf.div(segmented_sum, tf.reshape(counts+0.1, (-1, 1)))
    mu_expand = tf.gather(mu, unique_id)

    # 计算公式的loss(var)
    distance = tf.norm(tf.subtract(mu_expand, reshaped_pred), axis=1)
    distance = tf.subtract(distance, delta_v)
    distance = tf.clip_by_value(distance, 0., distance)
    distance = tf.square(distance)

    l_var = tf.unsorted_segment_sum(distance, unique_id, num_instances)
    l_var = tf.div(l_var, counts)
    l_var = tf.reduce_sum(l_var)
    l_var = tf.divide(l_var, tf.cast(num_instances, tf.float32))

    # 计算公式的loss(dist)
    mu_interleaved_rep = tf.tile(mu, [num_instances, 1])
    mu_band_rep = tf.tile(mu, [1, num_instances])
    mu_band_rep = tf.reshape(
        mu_band_rep,
        (num_instances *
         num_instances,
         feature_dim))

    mu_diff = tf.subtract(mu_band_rep, mu_interleaved_rep)

    # 去除掩模上的零点
    intermediate_tensor = tf.reduce_sum(tf.abs(mu_diff), axis=1)
    zero_vector = tf.zeros(1, dtype=tf.float32)
    bool_mask = tf.not_equal(intermediate_tensor, zero_vector)
    mu_diff_bool = tf.boolean_mask(mu_diff, bool_mask)

    mu_norm = tf.norm(mu_diff_bool, axis=1)
    mu_norm = tf.subtract(2. * delta_d, mu_norm)
    mu_norm = tf.clip_by_value(mu_norm, 0., mu_norm)
    mu_norm = tf.square(mu_norm)
    
    l_dist = tf.reduce_mean(mu_norm)

    # 计算原始Discriminative Loss论文中提到的正则项损失
    l_reg = tf.reduce_mean(tf.norm(mu, axis=1))

    # 合并损失按照原始Discriminative Loss论文中提到的参数合并
    param_scale = 1.
    l_var = param_var * l_var
    l_dist = param_dist * l_dist
    l_reg = param_reg * l_reg

    loss = param_scale * (l_var + l_dist + l_reg)

    return loss, l_var, l_dist, l_reg,unique_labels


def discriminative_loss(prediction, correct_label, feature_dim, image_shape,
                        delta_v, delta_d, param_var, param_dist, param_reg):
    """
    按照论文的思想迭代计算loss损失
    :return: discriminative loss and its three components
    """
    def cond(label, batch, out_loss, out_var, out_dist, out_reg,out_unique_labels, i):
        return tf.less(i, tf.shape(batch)[0])

    def body(label, batch, out_loss, out_var, out_dist, out_reg,out_unique_labels, i):
        disc_loss, l_var, l_dist, l_reg,unique_labels = discriminative_loss_single(
            prediction[i], correct_label[i], feature_dim, image_shape, delta_v, delta_d, param_var, param_dist, param_reg)

        out_loss = out_loss.write(i, disc_loss)
        out_var = out_var.write(i, l_var)
        out_dist = out_dist.write(i, l_dist)
        out_reg = out_reg.write(i, l_reg)
        out_unique_labels= out_unique_labels.write(i,unique_labels)
        return label, batch, out_loss, out_var, out_dist, out_reg, out_unique_labels, i + 1

    # TensorArray is a data structure that support dynamic writing
    output_ta_loss = tf.TensorArray(dtype=tf.float32,
                                    size=0,
                                    dynamic_size=True)
    output_ta_var = tf.TensorArray(dtype=tf.float32,
                                   size=0,
                                   dynamic_size=True)
    output_ta_dist = tf.TensorArray(dtype=tf.float32,
                                    size=0,
                                    dynamic_size=True)
    output_ta_reg = tf.TensorArray(dtype=tf.float32,
                                   size=0,
                                   dynamic_size=True)
    output_ta_unique_labels = tf.TensorArray(dtype=tf.float32,
                                   size=0,
                                   dynamic_size=True)
    #print("correct_label:",correct_label)
    #print("prediction:",prediction)
    
    _, _, out_loss_op, out_var_op, out_dist_op, out_reg_op, out_unique_labels, _ = tf.while_loop(
        cond, body, [
            correct_label, prediction, output_ta_loss, output_ta_var, output_ta_dist, output_ta_reg,output_ta_unique_labels, 0])
    out_loss_op = out_loss_op.stack()
    out_var_op = out_var_op.stack()
    out_dist_op = out_dist_op.stack()
    out_reg_op = out_reg_op.stack()
    #out_unique_labels=out_unique_labels.stack()
    
    disc_loss = tf.reduce_mean(out_loss_op)
    l_var = tf.reduce_mean(out_var_op)
    l_dist = tf.reduce_mean(out_dist_op)
    l_reg = tf.reduce_mean(out_reg_op)

    return disc_loss, l_var, l_dist, l_reg,out_unique_labels