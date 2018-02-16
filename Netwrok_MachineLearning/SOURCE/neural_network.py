# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 17:55:53 2018

@author: rahul.ghosh
"""

import tensorflow as tf


class Hidden_Layer():

    def __init__(self, shape):
        self.weights = tf.Variable(tf.truncated_normal(shape=shape, stddev=0.01))
        print(shape[0])
        self.biases = tf.Variable(tf.random_normal([shape[1]]))

    def feed_forward(self, input_data):
        output_data = tf.nn.relu(tf.add(tf.matmul(input_data, self.weights), self.biases))
        return output_data


class Out_Layer():

    def __init__(self, shape):
        self.weights = tf.Variable(tf.truncated_normal(shape=shape, stddev=0.01))
        self.biases = tf.Variable(tf.random_normal([shape[1]]))

    def feed_forward(self, input_data):
        output_data = tf.add(tf.matmul(input_data, self.weights), self.biases)
        return output_data
