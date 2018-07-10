#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 17:56:05 2018

@author: rahul
"""

import data
import model
import config
import utilities


if __name__ == "__main__":
    # READ DATA
    train_data = data.DATA()
    train_data.build_dataset(config.TRAIN_FILENAME)
    # BUILD MODEL
    net = model.MODEL()
    net.build()
    # TRAIN MODEL
    net.train(train_data)
    # PLOT EMBEDDINGS
    utilities.plot_with_labels(net.embeddings, train_data)
