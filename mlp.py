import torch
from data import DataPreprocessor
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import math

class MLP:

    def __init__(self, input_size, hidden_size, output_size):
        self.w1 = (torch.randn(input_size, hidden_size) * 0.01).detach().requires_grad_()
        self.b1 = torch.zeros(1, hidden_size, requires_grad=True)
        self.w2 = (torch.randn(hidden_size, output_size) * 0.01).detach().requires_grad_()
        self.b2 = torch.zeros(1, output_size, requires_grad=True)

        self.z1: torch.Tensor | None = None
        self.a1: torch.Tensor | None = None
        self.z2: torch.Tensor | None = None
        self.a2: torch.Tensor | None = None



    def forward(self, X):
        self.z1 = torch.matmul(X, self.w1) + self.b1 #z1=X⋅W1+b1
        self.a1 = torch.nn.functional.leaky_relu(self.z1, negative_slope=0.01)  # Hidden layer activation
        self.z2 = torch.matmul(self.a1, self.w2) + self.b2
        return self.z2


    def backward(self,X,y,output,lr=0.01):
        assert self.a1 is not None
        assert self.z1 is not None
        m=X.shape[0]
        dz2 = torch.softmax(output, dim=1) - y
        dw2=torch.matmul(self.a1.T,dz2)/m
        db2=torch.sum(dz2,dim=0)/m

        da1=torch.matmul(dz2,self.w2.T)
        dz1 = da1.clone()
        dz1[self.z1 <= 0] = 0
        dw1=torch.matmul(X.T,dz1)/m
        db1 = torch.sum(dz1, dim=0) / m

        with torch.no_grad():
            self.w1 -= lr * dw1
            self.b1 -= lr * db1
            self.w2 -= lr * dw2
            self.b2 -= lr * db2  



    def train(self, X, y, epochs=1000, lr=0.001):
        optimizer = torch.optim.SGD([self.w1, self.b1, self.w2, self.b2], lr=lr)
        losses = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self.forward(X)
            loss = torch.nn.functional.cross_entropy(output, torch.argmax(y, dim=1))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
        return losses


preprocessor = DataPreprocessor()
preprocessor.prepare_data()
preprocessor.to_tensors()
Xtrain, ytrain, Xtest, ytest = preprocessor.return_data()

input_size = 3072
hidden_size = 128
output_size = 10 
model = MLP(input_size, hidden_size, output_size)

#Train  modeland store the losses
losses = model.train(Xtrain, ytrain, 100, 0.001)

with torch.no_grad():
    test_output = model.forward(Xtest)
    print(test_output)
    predicted_class = torch.argmax(test_output, dim=1).float()
    print(predicted_class)


ytest = ytest.squeeze()
assert isinstance(ytest, torch.Tensor), f"ytest must be a torch.Tensor, got {type(ytest)}"

length = ytest.shape[0]
count = 0
for i in range(length):
    if ytest[i] == predicted_class[i]:
        count += 1
    
print(count/length)


accuracy = torch.mean((predicted_class == ytest).float())
print(f"Test Accuracy: {accuracy.item() * 100:.2f}%")
