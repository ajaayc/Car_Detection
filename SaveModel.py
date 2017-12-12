#!/usr/bin/python3
"""
Created on Tue Dec 12 18:15:03 2017

@author: rh
"""
import tensorflow as tf

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('models/train/model.ckpt-2000.meta')
    saver.restore(sess, "model-2000.ckpt")