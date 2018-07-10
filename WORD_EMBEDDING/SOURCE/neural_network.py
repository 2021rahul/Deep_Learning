#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 22:08:14 2018

@author: rahul
"""

import tensorflow as tf
import config
import math

class Embedding_Layer():

    def __init__(self, shape):
        self.embedding = tf.get_variable("embedding", shape=shape, dtype=tf.float32)

    def lookup(self, input_data):
        output = tf.nn.embedding_lookup(self.embedding, input_data)
        return output


class NCE_Layer():

    def __init__(self, shape):
        self.weights = tf.Variable(tf.truncated_normal(shape=shape, stddev=1.0 / math.sqrt(shape[1])))
        self.biases = tf.Variable(tf.zeros(shape=shape[0]))

    def loss(self, input_data, labels):
        loss = tf.reduce_mean(tf.nn.nce_loss(self.weights, self.biases, labels, input_data,
                                             num_sampled=config.NUM_SAMPLED,
                                             num_classes=config.VOCABULARY_SIZE))
        return loss
