# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 20:00:55 2018

@author: rahul.ghosh
"""

import data
import model
import config
import neural_network

if __name__ == "__main__":
    data = data.DATA()
    data.read()
    
    model = model.MODEL()
    model.build()
#    model.train(data)
