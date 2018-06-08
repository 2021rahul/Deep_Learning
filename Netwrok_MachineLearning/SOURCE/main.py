# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 20:00:55 2018

@author: rahul.ghosh
"""

import data
import model
import config

if __name__ == "__main__":
    data = data.DATA()
    data.read(config.TRAIN_FILENAME)
    
    model = model.MODEL()
    model.build()
    model.train(data)
    
    data.read(config.TEST_FILENAME)
    model.test(data.dataX, data.dataY)
