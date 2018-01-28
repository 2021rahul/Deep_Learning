#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 17:51:32 2018

@author: rahul
"""

url = 'http://mattmahoney.net/dc/'
filename = 'text8.zip'
expected_bytes = 31344016
vocabulary_size = 50000
batch_size = 128
embedding_size = 128
skip_window = 1
num_skips = 2
num_sampled = 64
valid_size = 16
valid_window = 100
num_steps = 100001
