# -*- coding: utf-8 -*-
# pytorch SMLP with only one hidden layer for multiclass classification

from torch.nn import Linear, Tanh, Module, Softmax
import torch
import numpy as np
from torchsummary import summary

class presingle(torch.nn.Module):
    def __init__(self):
        super(presingle, self).__init__()
        self.presingle = torch.nn.Linear(1, 1, bias=False) # bias are set to False
        torch.nn.init.ones_(self.presingle.weight)
    def forward(self, x):
        x = self.presingle(x)
        return x

# model definition
class SMLP(Module):
    # define model elements
    def __init__(self, n_inputs, n_hiddens, n_classes):
        super(SMLP, self).__init__()
        self.presingle = [presingle() for i in range(n_inputs)]
        self.input = n_inputs
        for i in range(n_inputs):
            setattr(self, "presingle_" + str(i + 1), self.presingle[i])
        self.hidden1 = Linear(n_inputs, n_hiddens)  # You can define several hidden layers
        self.act1 = Tanh()                          # This activation function depends on what you like
        self.hidden2 = Linear(n_hiddens, n_classes)
        self.act2 = Softmax(dim=1)


    def forward(self, x):
        lst = []
        for n in range(self.input):
            lst.append(torch.nn.functional.relu(self.presingle[n](x[:, n].unsqueeze(1))))
        input_cat = torch.cat((lst), 1)
        X = self.hidden1(input_cat)
        X = self.act1(X)
        X = self.hidden2(X)
        X = self.act2(X)
        return X

"""
n_input = 3     # neurons of the input layer
n_hidden = 10   # neurons of the hidden layer
n_class = 2     # neurons of the output layer
model = SMLP(n_input, n_hidden, n_class)
print(model)
"""