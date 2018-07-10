#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
import os
import zipfile
import numpy as np
import random
import config


class DATA():

    def __init__(self):
        self.data = None
        self.count = None
        self.dictionary = None
        self.reverse_dictionary = None
        self.batch_size = config.BATCH_SIZE
        self.vocabulary_size = config.VOCABULARY_SIZE
        self.size = None
        self.data_index = 0
        self.num_skips = config.NUM_SKIP
        self.skip_window = config.SKIP_WINDOW

    def read(self, filename):
        filename = os.path.join(config.DATA_DIR, filename)
        with zipfile.ZipFile(filename, 'r') as file:
            data = file.read(file.namelist()[0]).decode('utf8').split()
        return data

    def build_dataset(self, filename):
        words = self.read(filename)
        self.count = [['UNK', -1]]
        self.count.extend(collections.Counter(words).most_common(self.vocabulary_size - 1))
        self.dictionary = dict()
        for word, _ in self.count:
            self.dictionary[word] = len(self.dictionary)
        self.data = list()
        unk_count = 0
        for word in words:
            index = self.dictionary.get(word, 0)
            if index == 0:
                unk_count += 1
            self.data.append(index)
        self.count[0][1] = unk_count
        self.reverse_dictionary = dict(zip(self.dictionary.values(), self.dictionary.keys()))

    def generate_batch(self):
        batch = np.ndarray(shape=(self.batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32)
        span = 2 * self.skip_window + 1
        buffer = collections.deque(maxlen=span)
        if self.data_index + span > len(self.data):
            self.data_index = 0
        buffer.extend(self.data[self.data_index:self.data_index + span])
        self.data_index += span
        for i in range(self.batch_size // self.num_skips):
            context_words = [w for w in range(span) if w != self.skip_window]
            words_to_use = random.sample(context_words, self.num_skips)
            for j, context_word in enumerate(words_to_use):
                batch[i * self.num_skips + j] = buffer[self.skip_window]
                labels[i * self.num_skips + j, 0] = buffer[context_word]
            if self.data_index == len(self.data):
                buffer[:] = self.data[:span]
                self.data_index = span
            else:
                buffer.append(self.data[self.data_index])
                self.data_index += 1
        self.data_index = (self.data_index + len(self.data) - span) % len(self.data)
        return batch, labels
