# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 17:55:53 2018

@author: rahul.ghosh
"""

import tensorflow as tf


class Hidden_Layer():

    def __init__(self, shape):
        with tf.name_scope("Variables"):
            self.weights = tf.Variable(tf.truncated_normal(shape=shape, stddev=0.01), name='weights')
            self.biases = tf.Variable(tf.random_normal(shape=[shape[1]]), name='biases')

    def feed_forward(self, input_data):
        with tf.name_scope("Operations"):
            x_w = tf.matmul(input_data, self.weights, name="multiply_weights")
            bias_add = tf.add(x_w, self.biases, name="add_bias")
            output_data = tf.nn.relu(bias_add, name="relu_activation")
            return output_data


class Out_Layer():

    def __init__(self, shape):
        with tf.name_scope("Variables"):
            self.weights = tf.Variable(tf.truncated_normal(shape=shape, stddev=0.01), name='weights')
            self.biases = tf.Variable(tf.random_normal(shape=[shape[1]]), name='biases')

    def feed_forward(self, input_data):
        with tf.name_scope("Operations"):
            x_w = tf.matmul(input_data, self.weights, name="multiply_weights")
            bias_add = tf.add(x_w, self.biases, name="add_bias")
            output_data = tf.nn.softmax(bias_add, name="softmax")
            return output_data
