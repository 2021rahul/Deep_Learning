# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 17:27:28 2018

@author: rahul.ghosh
"""

import pandas as pd
import numpy as np
import os
import config
from torch.utils.data.dataset import Dataset


class DATA(Dataset):

    def __init__(self):
        self.batch_size = config.BATCH_SIZE
        self.data_index = 0
        self.dataX = None
        self.dataY = None
        self.size = None

    def read(self, filename):
        data = pd.read_csv(os.path.join(config.DATA_DIR, filename), header=None)
        data = np.asarray(data, dtype=np.float32)
        self.size, self.num_features = data.shape
        self.dataX = data[:, :-1]
        self.dataY = data[:, -1]

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.dataX[index], self.dataY[index]
