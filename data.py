import scipy.io
import torch
import numpy as np

class DataPreprocessor:
    def __init__(self):
        test = scipy.io.loadmat('../test.mat')
        train = scipy.io.loadmat('../train.mat')

        self.Xtrain = train['X']
        self.ytrain = np.ravel(train['y'])-1
        self.Xtest = test['X']
        self.ytest = np.ravel(test['y'])-1



    def normalize(self, x):
        return x/255 # deep learning models prefer number [0, 1]

    
    def prepare_data(self):
        self.Xtrain = np.transpose(self.Xtrain, (3, 0, 1, 2))
        self.Xtrain = np.reshape(self.Xtrain, (self.Xtrain.shape[0], self.Xtrain.shape[1] * self.Xtrain.shape[2] * self.Xtrain.shape[3]))
        self.Xtrain = np.array([self.normalize(xi) for xi in self.Xtrain])
        
        self.Xtest = np.transpose(self.Xtest, (3, 0, 1, 2))
        self.Xtest = np.reshape(self.Xtest, (self.Xtest.shape[0], self.Xtest.shape[1] * self.Xtest.shape[2] * self.Xtest.shape[3]))
        self.Xtest = np.array([self.normalize(xi) for xi in self.Xtest])


    def to_tensors(self):
        self.Xtrain = torch.tensor(self.Xtrain, dtype=torch.float32)
        self.ytrain = torch.tensor(self.ytrain, dtype=torch.float32).reshape(-1, 1)
        self.Xtest = torch.tensor(self.Xtest, dtype=torch.float32)
        self.ytest = torch.tensor(self.ytest, dtype=torch.float32).reshape(-1, 1)


    def return_data(self):
        return self.Xtrain, self.ytrain, self.Xtest, self.ytest
