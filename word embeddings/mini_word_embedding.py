#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 17:41:46 2018

@author: rahul
"""

import math
import numpy as np
from six.moves import xrange
import matplotlib.pyplot as plt
import tensorflow as tf
import input


class Word_Embedding():

    def __init__(self, batch, vocabulary_size, embedding_size, num_sampled, valid_size, valid_window, num_skips, num_steps):
        self.num_sampled = num_sampled
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.valid_size = valid_size
        self.valid_window = valid_window
        self.num_steps = num_steps
        self.batch = batch

        self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch.batch_size])
        self.train_labels = tf.placeholder(tf.int32, shape=[self.batch.batch_size, 1])
        self.embeddings = tf.Variable(tf.random_uniform(shape=[self.vocabulary_size,self.embedding_size],
                                                        minval=-1.0,
                                                        maxval=1.0))
        self.loss = None

    def get_valid_examples(self):
        return np.random.choice(self.valid_window, self.valid_size, replace=False)

    def build_model(self):
        embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)
        nce_weights = tf.Variable(tf.truncated_normal([self.vocabulary_size, self.embedding_size], stddev=1.0 / math.sqrt(self.embedding_size)))
        nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]))
        self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                                  biases=nce_biases,
                                                  labels=self.train_labels,
                                                  inputs=embed,
                                                  num_sampled=self.num_sampled,
                                                  num_classes=self.vocabulary_size))

    def train_model(self, data, reverse_dictionary):
        valid_dataset = tf.constant(self.get_valid_examples(), dtype=tf.int32)
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(self.loss)
        norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keep_dims=True))
        normalized_embeddings = self.embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
        similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            print('All variables Initialized')
            average_loss = 0
            for step in xrange(self.num_steps):
                batch_inputs, batch_labels = self.batch.generate(data)
                feed_dict = {self.train_inputs: batch_inputs, self.train_labels: batch_labels}
                _, loss_val = session.run([optimizer, self.loss], feed_dict=feed_dict)
                average_loss += loss_val
                if step % 2000 == 0:
                    if step > 0:
                        average_loss /= 2000
                    print('Average loss at step ', step, ': ', average_loss)
                    average_loss = 0
                if step % 10000 == 0:
                    sim = similarity.eval()
                    for i in xrange(self.valid_size):
                        valid_word = reverse_dictionary[self.get_valid_examples()[i]]
                        top_k = 8
                        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                        log_str = 'Nearest to %s:' % valid_word
                        for k in xrange(top_k):
                            close_word = reverse_dictionary[nearest[k]]
                            log_str = '%s %s,' % (log_str, close_word)
                        print(log_str)
            final_embeddings = normalized_embeddings.eval()
        return final_embeddings

    def plot_with_labels(self, low_dim_embs, labels, filename):
        assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
        plt.figure(figsize=(18, 18))
        for i, label in enumerate(labels):
            x, y = low_dim_embs[i, :]
            plt.scatter(x, y)
            plt.annotate(label,
                         xy=(x, y),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')
        plt.savefig(filename)
