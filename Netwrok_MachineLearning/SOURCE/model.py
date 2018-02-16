# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 17:55:53 2018

@author: rahul.ghosh
"""

import tensorflow as tf
import config
import numpy as np
import neural_network


class MODEL():

    def __init__(self):
        self.train_inputs = tf.placeholder("float", shape=[None, config.NUM_FEATURES])
        self.train_labels = tf.placeholder("float", shape=[None, config.NUM_CLASS])
        self.loss = None
        self.weights = None

    def dense_to_one_hot(self, labels):
        num_labels = labels.shape[0]
        index_offset = np.arange(num_labels) * config.NUM_CLASS
        labels_one_hot = np.zeros((num_labels, config.NUM_CLASS))
        labels_one_hot.flat[index_offset + labels.ravel()] = 1
        return labels_one_hot

    def loss_function(self, Y, predict):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=predict))

    def build(self):
        input_data = self.train_inputs
        for i in range(config.HIDDEN_LAYERS):
            h = neural_network.Hidden_Layer(config.SHAPE[i])
            h_output = h.feed_forward(input_data)
            input_data = h_output

        out_layer = neural_network.Out_Layer(config.SHAPE[-1])
        output = out_layer.feed_forward(h_output)

        predict = tf.argmax(output, axis=1)
        self.loss = self.loss_function(self.train_labels, predict)

    def train(self, data):
        optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(self.loss)

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            print('All variables Initialized')

            total_batch = data.size/config.BATCH_SIZE
            for epoch in range(config.NUM_EPOCHS):
                avg_cost = 0
                for batch in range(total_batch):
                    batch_X, batch_Y = data.generate_batch()
                    feed_dict = {self.train_inputs: batch_X, self.train_labels: batch_Y}
                    _, loss_val = session.run([optimizer, self.loss], feed_dict=feed_dict)
                    avg_cost += loss_val / total_batch
                print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
