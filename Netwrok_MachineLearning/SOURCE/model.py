# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 17:55:53 2018

@author: rahul.ghosh
"""

import tensorflow as tf
import config
import numpy as np
import neural_network
import os


class MODEL():

    def __init__(self):
        self.inputs = tf.placeholder(shape=[None, config.NUM_FEATURES], dtype=tf.float32)
        self.labels = tf.placeholder(shape=[None, config.NUM_CLASS], dtype=tf.float32)
        self.loss = None
        self.weights = None
        self.output = None



    def loss_function(self, Y, predict):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=predict))

    def build(self):
        input_data = self.inputs
        for i in range(config.HIDDEN_LAYERS):
            h = neural_network.Hidden_Layer(config.SHAPE[i])
            h_output = h.feed_forward(input_data)
            input_data = h_output

        out_layer = neural_network.Out_Layer(config.SHAPE[-1])
        self.output = out_layer.feed_forward(h_output)
        self.loss = tf.reduce_mean(-tf.reduce_sum(self.labels * tf.log(self.output), axis=0))

    def train(self, data):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(self.loss)
        saver = tf.train.Saver()
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            print('All variables Initialized')

            total_batch = int(data.size/config.BATCH_SIZE)
            for epoch in range(config.NUM_EPOCHS):
                avg_cost = 0
                for batch in range(total_batch):
                    batch_X, batch_Y = data.generate_batch()
                    feed_dict = {self.inputs: batch_X, self.labels: batch_Y}
                    _, loss_val = session.run([optimizer, self.loss], feed_dict=feed_dict)
                    print("batch:", batch, " loss: ", loss_val)
                    avg_cost += loss_val / total_batch
                print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))

            save_path = saver.save(session, os.path.join(config.MODEL_DIR, "model.ckpt"))
            print("Model saved in path: %s" % save_path)


    def test(self, dataX, dataY):
        with tf.Session() as session:
            saver = tf.train.Saver()
            saver.restore(session, os.path.join(config.MODEL_DIR, "model.ckpt"))
            for i in range(len(dataX)):
                feed_dict = {self.inputs: [dataX[i]]}
                predicted = np.rint(session.run(self.output, feed_dict=feed_dict))
                print('Actual:', dataY[i], 'Predicted:', predicted)
