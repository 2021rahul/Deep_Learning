# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 17:55:53 2018

@author: rahul.ghosh
"""

import tensorflow as tf
import utils


class Hidden_Layer():

    def __init__(self, shape):
        with tf.name_scope("Variables"):
            with tf.name_scope("Weights"):
                self.weights = tf.Variable(tf.truncated_normal(shape=shape, stddev=0.01))
                utils.variable_summaries(self.weights)
            with tf.name_scope("Biases"):
                self.biases = tf.Variable(tf.random_normal(shape=[shape[1]]))
                utils.variable_summaries(self.biases)

    def feed_forward(self, input_data):
        with tf.name_scope("Operations"):
            x_w = tf.matmul(input_data, self.weights, name="multiply_weights")
            bias_add = tf.add(x_w, self.biases, name="add_bias")
            tf.summary.histogram('pre_activations', bias_add)
            output_data = tf.nn.relu(bias_add, name="relu_activation")
            tf.summary.histogram('relu_activation', output_data)
            return output_data


class Out_Layer():

    def __init__(self, shape):
        with tf.name_scope("Variables"):
            with tf.name_scope("Weights"):
                self.weights = tf.Variable(tf.truncated_normal(shape=shape, stddev=0.01), name='weights')
                utils.variable_summaries(self.weights)
            with tf.name_scope("Biases"):
                self.biases = tf.Variable(tf.random_normal(shape=[shape[1]]), name='biases')
                utils.variable_summaries(self.biases)

    def feed_forward(self, input_data):
        with tf.name_scope("Operations"):
            x_w = tf.matmul(input_data, self.weights, name="multiply_weights")
            bias_add = tf.add(x_w, self.biases, name="add_bias")
            tf.summary.histogram('pre_activations', bias_add)
            output_data = tf.nn.softmax(bias_add, name="softmax")
            tf.summary.histogram('softmax_activations', output_data)
            return output_data
