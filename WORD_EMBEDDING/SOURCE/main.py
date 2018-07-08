#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 17:56:05 2018

@author: rahul
"""

import mini_word_embedding
import input
import config
from sklearn.manifold import TSNE
from six.moves import xrange
import os


data = input.Data(config.url, config.filename, config.expected_bytes)
filename = data.maybe_download()
vocabulary = data.read()
print('Data size', len(vocabulary))
data, count, dictionary, reverse_dictionary = data.build_dataset(words=vocabulary, 
                                                                 n_words=config.vocabulary_size)
del vocabulary
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

batch = input.Batch(batch_size=8,
                    num_skips=2,
                    skip_window=1)

batch_input, labels = batch.generate(data)
for i in range(8):
    print(batch_input[i], reverse_dictionary[batch_input[i]],
          '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

batch = input.Batch(batch_size=config.batch_size,
                    num_skips=config.num_skips,
                    skip_window=config.skip_window)

word_embedding = mini_word_embedding.Word_Embedding(batch,
                                                    config.vocabulary_size,
                                                    config.embedding_size,
                                                    config.num_sampled,
                                                    config.valid_size,
                                                    config.valid_window,
                                                    config.num_skips,
                                                    config.num_steps)

word_embedding.build_model()
final_embeddings = word_embedding.train_model(data, reverse_dictionary)

tsne = TSNE(perplexity=30,
            n_components=2,
            init='pca',
            n_iter=5000,
            method='exact')
plot_only = 500
low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
labels = [reverse_dictionary[i] for i in xrange(plot_only)]
word_embedding.plot_with_labels(low_dim_embs, labels, os.path.join(os.getcwd(), 'tsne.png'))
