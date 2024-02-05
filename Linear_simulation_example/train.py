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
        self.y = LabelEncoder().fit_transform(self.y)
        labels = np.zeros((self.y.shape[0], n_class))
        for i in range(len(self.y)):
            labels[i, self.y[i]] = 1
        self.y = labels
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
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = Adam(model.parameters(),lr=0.005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10000, gamma=0.1) # lr decay
    # enumerate epochs
    for epoch in range(n_epoch):
        model.train()
        loss_epoch = 0
        predictions = []
        actuals = []
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            targets = targets.to(device)
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
            if epoch == n_epoch - 1:
                yhat = yhat.detach().cpu().numpy()
                actual = targets.cpu().numpy()
                actual = argmax(actual, axis=1)
                # convert to class labels
                yhat = argmax(yhat, axis=1)
                for x in list(actual):
                    actuals.append(x)
                for y in list(yhat):
                    predictions.append(y)
        scheduler.step()
        if epoch % 100 == 0:
            # eval
            model.eval()
            loss_eval = 0
            for i, (inputs, targets) in enumerate(test_dl):
                # evaluate the model on the test set
                targets = targets.to(device)
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

# evaluate the model
def evaluate_model(test_dl, model):
    predictions, actuals = [], []
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        targets = targets.to(device)
        inputs = inputs.to(device)
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().cpu().numpy()
        actual = targets.cpu().numpy()
        actual = argmax(actual, axis=1)
        # convert to class labels
        yhat = argmax(yhat, axis=1)
        # reshape for stacking
        actual = actual.reshape((len(actual), 1))
        yhat = yhat.reshape((len(yhat), 1))
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    acc = accuracy_score(actuals, predictions)
    return acc

linear_simulation_data() # generate linear simulation data
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_input = 3         # neurons of the input layer
n_hidden = 10       # neurons of the hidden layer
n_class = 2         # neurons of the output layer
n_epoch = 30000     # training epochs
train_dl, test_dl = prepare_data(r"train.xlsx", r"test.xlsx")  # simulation data
# define the network
model = SMLP(n_input, n_hidden, n_class).to(device)
print(model)
# train the model
train_model(train_dl, model)
# evaluate the model
acc = evaluate_model(test_dl, model)
print('Accuracy: %.4f' % acc)
load_feature_importance(model)

