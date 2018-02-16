# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 17:16:29 2018

@author: rahul.ghosh
"""

import os

# DIRECTORY INFORMATION
ROOT_DIR = os.path.abspath('../')
DATA_DIR = os.path.join(ROOT_DIR, 'DATASET/')
OUT_DIR = os.path.join(ROOT_DIR, 'RESULT/')

FILENAME = "train_iris.csv"

# DATA INFORMATION
NUM_FEATURES = 3
NUM_CLASS = 3
BATCH_SIZE = 9

# MODEL INFORMATION
HIDDEN_LAYERS = 1
SHAPE = [(NUM_FEATURES, 3),
         (3, NUM_CLASS)]


# RANDOM NUMBER GENERATOR INFORMATION
SEED = 128
