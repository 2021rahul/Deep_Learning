# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 17:55:53 2018

@author: rahul.ghosh
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import config
import os
torch.manual_seed(config.SEED)


class MODEL(nn.Module):

    def __init__(self):
        super(MODEL, self).__init__()
        self.HiddenLayer1 = nn.Linear(config.SHAPE[0][0], config.SHAPE[0][1])
        self.HiddenLayer2 = nn.Linear(config.SHAPE[1][0], config.SHAPE[1][1])
        self.OutputLayer = nn.Linear(config.SHAPE[2][0], config.SHAPE[2][1])

    def forward(self, x):
        output = nn.functional.relu(self.HiddenLayer1(x))
        output = nn.functional.relu(self.HiddenLayer2(output))
        output = self.OutputLayer(output)
        return output


class operators():

    def __init__(self, net):
        self.net = net
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.1)

    def train(self, data):
        for epoch in range(config.NUM_EPOCHS):
            X = Variable(torch.Tensor(data.dataX).float())
            Y = Variable(torch.Tensor(data.dataY).long())
            self.optimizer.zero_grad()
            out = self.net(X)
            loss = self.loss(out, Y)
            loss.backward()
            self.optimizer.step()
            print("Epoch:", (epoch + 1), "loss =", "{:.3f}".format(loss.data[0]))
            torch.save(self.net, os.path.join(config.MODEL_DIR, "model" + str(config.BATCH_SIZE) + "_" + str(config.NUM_EPOCHS) + ".pt"))

    def test(self, data):
        self.net = torch.load(os.path.join(config.MODEL_DIR, "model" + str(config.BATCH_SIZE) + "_" + str(config.NUM_EPOCHS) + ".pt"))
        X = Variable(torch.Tensor(data.dataX).float())
        Y = Variable(torch.Tensor(data.dataY).long())
        out = self.net(X)
        _, predicted = torch.max(out.data, 1)
        print('Accuracy of the network %d %%' % (100 * torch.sum(Y == predicted) / 60))
