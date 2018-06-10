# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 17:55:53 2018

@author: rahul.ghosh
"""

import torch.nn as nn


class Hidden_Layer(nn.Module):

    def __init__(self, shape):
        super(Hidden_Layer, self).__init__()
        self.layer = nn.Relu(nn.Linear(shape[0], shape[1]))

    def feed_forward(self, input_data):
        output_data = self.layer(input_data)
        return output_data


class Out_Layer(nn.Module):

    def __init__(self, shape):
        super(Out_Layer, self).__init__()
        self.layer = nn.Linear(shape[0], shape[1])

    def feed_forward(self, input_data):
        output_data = self.layer(input_data)
        return output_data
