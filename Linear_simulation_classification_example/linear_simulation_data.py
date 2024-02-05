import numpy as np
import pandas as pd


def sigmoid(Z):
    """
    Implements the sigmoid activation in bumpy

    Arguments:
    Z -- numpy array of any shape

    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z, useful during backpropagation
    """

    cache = Z
    A = 1 / (1 + (np.exp((-Z))))
    return A

def linear_simulation_data():
    total_input = np.random.normal(0.00001,1,size=(10000,3))
    total_label = total_input[:,0] + 0.5 * total_input[:,1] + 0.1 * total_input[:,2]
    total_label = sigmoid(total_label)
    median = np.median(total_label)
    for i in range(total_label.shape[0]):
        if total_label[i] <= median:
            total_label[i] = 1
        else:
            total_label[i] = 0
    for i in range(3):
        x = total_input[:, i]
        Min = np.min(x)
        Max = np.max(x)
        x = (x - Min) / (Max - Min)
        for num in range(len(x)):
            if x[num] > 1.0:
                x[num] = 1.0
            if x[num] <= 0.00001:
                x[num] = 0.00001
        total_input[:, i] = x
    train_input = total_input[:9000,:]
    test_input = total_input[9000:,:]
    train_label = total_label[:9000]
    train_label = train_label[:,np.newaxis]
    train = np.concatenate([train_input, train_label], axis=1)
    test_label = total_label[9000:]
    test_label = test_label[:,np.newaxis]
    test = np.concatenate([test_input,test_label],axis=1)
    df_train = pd.DataFrame(train)
    df_test = pd.DataFrame(test)
    df_train.to_excel(r"train.xlsx",index=False)
    df_test.to_excel(r"test.xlsx",index=False)
