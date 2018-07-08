#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
import os
import zipfile
import numpy as np
import random
from six.moves import urllib
import tensorflow as tf


class Data():

    def __init__(self, url, filename, size):
        self.url = url
        self.filename = filename
        self.size = size

    def maybe_download(self):
        local_filename = os.path.join(os.getcwd(), self.filename)
        if not os.path.exists(local_filename):
            local_filename, _ = urllib.request.urlretrieve(self.url + self.filename, local_filename)
        statinfo = os.stat(local_filename)
        if statinfo.st_size == self.size:
            print('Found and verified', self.filename)
        else:
            print(statinfo.st_size)
            raise Exception('Failed to verify ' + local_filename + '. Can you get to it with a browser?')

    def read(self):
        with zipfile.ZipFile(self.filename) as f:
            data = tf.compat.as_str(f.read(f.namelist()[0])).split()
        return data

    def build_dataset(self, words, n_words):
        count = [['UNK', -1]]
        count.extend(collections.Counter(words).most_common(n_words - 1))
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)
        data = list()
        unk_count = 0
        for word in words:
            index = dictionary.get(word, 0)
            if index == 0:
                unk_count += 1
            data.append(index)
        count[0][1] = unk_count
        reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        return data, count, dictionary, reversed_dictionary


class Batch():

    def __init__(self, batch_size, num_skips, skip_window):
        self.data_index = 0
        self.batch_size = batch_size
        self.num_skips = num_skips
        self.skip_window = skip_window
        assert self.batch_size % self.num_skips == 0
        assert self.num_skips <= 2 * self.skip_window

    def generate(self, data):
        batch = np.ndarray(shape=(self.batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32)
        span = 2 * self.skip_window + 1
        buffer = collections.deque(maxlen=span)
        if self.data_index + span > len(data):
            self.data_index = 0
        buffer.extend(data[self.data_index:self.data_index + span])
        self.data_index += span
        for i in range(self.batch_size // self.num_skips):
            context_words = [w for w in range(span) if w != self.skip_window]
            words_to_use = random.sample(context_words, self.num_skips)
            for j, context_word in enumerate(words_to_use):
                batch[i * self.num_skips + j] = buffer[self.skip_window]
                labels[i * self.num_skips + j, 0] = buffer[context_word]
            if self.data_index == len(data):
                buffer[:] = data[:span]
                self.data_index = span
            else:
                buffer.append(data[self.data_index])
                self.data_index += 1
        self.data_index = (self.data_index + len(data) - span) % len(data)
        return batch, labels
