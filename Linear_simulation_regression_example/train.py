# -*- coding: utf-8 -*-
# pytorch smlp for multiclass classification
import numpy as np
import pandas as pd
from numpy import vstack
from numpy import argmax
from pandas import read_excel
from sklearn.preprocessing import LabelEncoder
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import torch
from SMLP_model import SMLP
from linear_simulation_data import linear_simulation_data
from load_feature_importance import load_feature_importance
from sklearn.metrics import accuracy_score
# dataset definition
class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, path):
        # load the excel file as a dataframe
        df = read_excel(path)
        print(df)
        # store the inputs and outputs
        self.X = df.values[:,:-1]
        self.y = df.values[:, -1]
        # ensure input data is floats
        self.X = self.X.astype('float32')
        # label encode target
        self.y = self.y.astype('float32')
    # number of rows in the dataset
    def __len__(self):
        return len(self.X)
    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

# prepare the dataset
def prepare_data(train_path,test_path):
    # load the dataset
    # calculate split
    train, test = CSVDataset(train_path), CSVDataset(test_path)
    # prepare data loaders
    train_dl = DataLoader(train, batch_size=1000, shuffle=True)
    test_dl = DataLoader(test, batch_size=1000, shuffle=False)
    return train_dl, test_dl

# train the model
def train_model(train_dl, model):
    criterion = torch.nn.MSELoss().to(device)
    optimizer = Adam(model.parameters(),lr=0.005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10000, gamma=0.1)
    # enumerate epochs
    for epoch in range(n_epoch):
        model.train()
        loss_epoch = 0
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            targets = targets.to(device).unsqueeze(1)
            inputs = inputs.to(device)
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            loss = criterion(yhat, targets)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()
            loss_epoch += loss.item()
        scheduler.step()
        if epoch % 100 == 0:
            # eval
            model.eval()
            loss_eval = 0
            for i, (inputs, targets) in enumerate(test_dl):
                # evaluate the model on the test set
                targets = targets.to(device).unsqueeze(1)
                inputs = inputs.to(device)
                yhat = model(inputs)
                with torch.no_grad():
                    loss = criterion(yhat, targets)
                    loss_eval += loss.item()
            trainLoss = loss_epoch/len(train_dl)
            evalLoss = loss_eval / len(test_dl)
            print("[%d]Train Loss : %.6f" % (epoch, trainLoss))
            print("[%d]Eval Loss : %.6f" % (epoch, evalLoss))
    torch.save(model.state_dict(), r"BPnet.pth")
    # save the model


_ = input("Generate new random data (True / False):")
if _:
    linear_simulation_data() # generate linear simulation data
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_input = 3         # neurons of the input layer
n_hidden = 10       # neurons of the hidden layer
n_class = 1         # neurons of the output layer
n_epoch = 30000     # training epochs
train_dl, test_dl = prepare_data(r"train.xlsx", r"test.xlsx")  # simulation data
# define the network
model = SMLP(n_input, n_hidden, n_class).to(device)
print(model)
# train the model
train_model(train_dl, model)
# evaluate the model
load_feature_importance(model)

