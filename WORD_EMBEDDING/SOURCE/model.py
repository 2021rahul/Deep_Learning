#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 17:41:46 2018

@author: rahul
"""


import os
import tensorflow as tf
import neural_network
import config


class MODEL():

    def __init__(self):
        self.inputs = tf.placeholder(shape=[None], dtype=tf.int32)
        self.labels = tf.placeholder(shape=[None, 1], dtype=tf.int32)
        self.embeddings = None
        self.loss = None
        self.output = None

    def build(self):
        inputs = self.inputs
        embedding_layer = neural_network.Embedding_Layer([config.VOCABULARY_SIZE, config.EMBEDDING_SIZE])
        embedding = embedding_layer.lookup(inputs)
        self.embeddings = embedding_layer.embedding

        nce_layer = neural_network.NCE_Layer([config.VOCABULARY_SIZE, config.EMBEDDING_SIZE])
        self.loss = nce_layer.loss(embedding, self.labels)

    def train(self, data):
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(self.loss)

        norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keep_dims=True))
        normalized_embeddings = self.embeddings / norm
        saver = tf.train.Saver()
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            print('All variables Initialized')
            average_loss = 0
            for step in range(config.NUM_STEPS):
                batch_inputs, batch_labels = data.generate_batch()
                feed_dict = {self.inputs: batch_inputs, self.labels: batch_labels}
                _, loss_val = session.run([optimizer, self.loss], feed_dict=feed_dict)
                average_loss += loss_val
                if step % 2000 == 0:
                    if step > 0:
                        average_loss /= 2000
                    print('Average loss at step ', step, ': ', average_loss)
                    average_loss = 0
            self.embeddings = normalized_embeddings.eval()

            save_path = saver.save(session, os.path.join(config.MODEL_DIR, "model" + str(config.SKIP_WINDOW) + "_" + str(config.NUM_STEPS)))
            print("Model saved in path: %s" % save_path)
