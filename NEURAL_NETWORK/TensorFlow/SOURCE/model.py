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
        self.output = None

    def build(self):
        input_data = self.inputs

        with tf.name_scope("h1_layer"):
            h1_layer = neural_network.Hidden_Layer(shape=(config.NUM_FEATURES, 8))
            h = h1_layer.feed_forward(input_data)

        with tf.name_scope("h2_layer"):
            h2_layer = neural_network.Hidden_Layer(shape=(8, 4))
            h = h2_layer.feed_forward(h)

        with tf.name_scope("out_layer"):
            out_layer = neural_network.Out_Layer(shape=(4, config.NUM_CLASS))
            self.output = out_layer.feed_forward(h)

        with tf.name_scope("cost_function"):
            self.loss = tf.reduce_mean(-tf.reduce_sum(self.labels * tf.log(self.output), axis=0))

    def train(self, data):
        with tf.name_scope("train"):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(self.loss)

        saver = tf.train.Saver()
        merged_summary_op = tf.summary.merge_all()

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            print('All variables Initialized')
            summary_writer = tf.summary.FileWriter(config.OUT_DIR, session.graph)

            total_batch = int(data.size/config.BATCH_SIZE)
            for epoch in range(config.NUM_EPOCHS):
                avg_cost = 0
                for batch in range(total_batch):
                    batch_X, batch_Y = data.generate_batch()
                    feed_dict = {self.inputs: batch_X, self.labels: batch_Y}
                    _, loss_val = session.run([optimizer, self.loss], feed_dict=feed_dict)
                    print("batch:", batch, " loss: ", loss_val)
                    avg_cost += loss_val / total_batch

#                summary_str = session.run([merged_summary_op], feed_dict=feed_dict)
#                summary_writer.add_summary(summary_str, epoch)
                print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))

            summary_writer.close()
            save_path = saver.save(session, os.path.join(config.MODEL_DIR, "model" + str(config.BATCH_SIZE) + "_" + str(config.NUM_EPOCHS) + ".ckpt"))
            print("Model saved in path: %s" % save_path)

    def test(self, data):
        saver = tf.train.Saver()
        with tf.Session() as session:
            saver.restore(session, os.path.join(config.MODEL_DIR, "model" + str(config.BATCH_SIZE) + "_" + str(config.NUM_EPOCHS) + ".ckpt"))
            for i in range(len(data.dataX)):
                feed_dict = {self.inputs: [data.dataX[i]]}
                predicted = np.rint(session.run(self.output, feed_dict=feed_dict))
                print('Actual:', data.dataY[i], 'Predicted:', predicted)
