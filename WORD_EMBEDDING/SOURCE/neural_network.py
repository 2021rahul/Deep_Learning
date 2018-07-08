#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 22:08:14 2018

@author: rahul
"""

import tensorflow as tf


class Embedding_Layer():

        def __init__(self, shape):
            self.embedding = tf.get_variable("embedding", shape=shape, dtype=tf.float32)
            #self.embedding = tf.Variable(tf.random_uniform(shape=[self.vocabulary_size,self.embedding_size],
                                                        minval=-1.0,
                                                        maxval=1.0))

        def lookup(self, input_data):
            output = tf.nn.embedding_lookup(self.embedding, input_data)
            return output