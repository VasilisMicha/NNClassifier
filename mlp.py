import torch
from data import DataPreprocessor
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import math


class MLP:

    def __init__(self, input_size, hidden_size, output_size):
        w1 = torch.empty(input_size, hidden_size)
        self.w1 = torch.nn.init.kaiming_uniform_(
            w1, a=0, mode="fan_in", nonlinearity="relu"
        )
        self.b1 = torch.zeros(hidden_size)

        w2 = torch.empty(hidden_size, hidden_size)  # hidden_size to hidden_size
        self.w2 = torch.nn.init.kaiming_uniform_(
            w2, a=0, mode="fan_in", nonlinearity="relu"
        )
        self.b2 = torch.zeros(hidden_size)

        w3 = torch.empty(hidden_size, output_size)
        self.w3 = torch.nn.init.xavier_uniform_(w3)
        self.b3 = torch.zeros(output_size)

        self.z1: torch.Tensor | None = None
        self.a1: torch.Tensor | None = None
        self.z2: torch.Tensor | None = None
        self.a2: torch.Tensor | None = None

    def forward(self, X):
        # Layer 1
        self.z1 = torch.matmul(X, self.w1) + self.b1
        self.a1 = torch.relu(self.z1)

        # Layer 2 (New Hidden Layer)
        self.z2 = torch.matmul(self.a1, self.w2) + self.b2  # Use a1 as input
        self.a2 = torch.relu(self.z2)  # New ReLU activation

        # Layer 3 (Output Layer)
        self.z3 = torch.matmul(self.a2, self.w3) + self.b3  # Use a2 as input
        return self.z3

    def backward(self, X, y, output, lr=0.01):
        assert self.a1 is not None and self.a2 is not None
        assert self.z1 is not None and self.z2 is not None
        m = X.shape[0]

        # --- STEP 1: Output Layer (L3) Gradients (dz3) ---
        probabilities = torch.softmax(output, dim=1)  # output is now z3
        y_one_hot = torch.nn.functional.one_hot(y, num_classes=output.shape[1])
        dz3 = probabilities - y_one_hot  # The error for the final logits

        # dw3 and db3
        dw3 = torch.matmul(self.a2.T, dz3) / m  # Use a2 as input
        db3 = torch.sum(dz3, dim=0) / m

        # --- STEP 2: Hidden Layer 2 (L2) Gradients (dz2) ---
        # 2a. Backpropagate error through w3 (da2)
        da2 = torch.matmul(dz3, self.w3.T)

        # 2b. Apply ReLU derivative for layer 2 (dz2)
        relu_derivative_mask_2 = self.z2.gt(0).float()
        dz2 = da2 * relu_derivative_mask_2

        # dw2 and db2
        dw2 = torch.matmul(self.a1.T, dz2) / m  # Use a1 as input
        db2 = torch.sum(dz2, dim=0) / m

        # --- STEP 3: Hidden Layer 1 (L1) Gradients (dz1) ---
        # 3a. Backpropagate error through w2 (da1)
        da1 = torch.matmul(dz2, self.w2.T)

        # 3b. Apply ReLU derivative for layer 1 (dz1)
        relu_derivative_mask_1 = self.z1.gt(0).float()
        dz1 = da1 * relu_derivative_mask_1

        # dw1 and db1
        dw1 = torch.matmul(X.T, dz1) / m
        db1 = torch.sum(dz1, dim=0) / m

        # --- STEP 4: Update Weights ---
        with torch.no_grad():
            self.w1 -= lr * dw1
            self.b1 -= lr * db1
            self.w2 -= lr * dw2
            self.b2 -= lr * db2
            self.w3 -= lr * dw3  # Update w3 and b3
            self.b3 -= lr * db3

    def train(self, X, y, epochs=1000, lr=0.001):
        losses = []
        for epoch in range(epochs):
            output = self.forward(X)

            criterion = torch.nn.CrossEntropyLoss()
            loss = criterion(output, y)
            losses.append(loss)
            # update weights
            self.backward(X, y, output, lr)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss:.4f}")
            validation_output = self.forward(Xvalid)
            loss = criterion(validation_output, yvalid)
            print(
                f"validation accuracy: {calculate_accuracy(Xvalid, yvalid)}, loss: {loss}"
            )
        return losses


def calculate_accuracy(X, y):
    with torch.no_grad():
        train_output = torch.softmax(model.forward(X), dim=1)
        predicted_class = torch.argmax(train_output, dim=1).float()
    length = y.shape[0]
    count = 0
    for i in range(length):
        if y[i] == predicted_class[i]:
            count += 1

    return 100 * count / length


preprocessor = DataPreprocessor()
preprocessor.prepare_data()
preprocessor.include_validation()
preprocessor.to_tensors()
Xtrain, ytrain, Xtest, ytest, Xvalid, yvalid = preprocessor.return_data()

input_size = 3 * 32 * 32
hidden_size = 512
output_size = 10
model = MLP(input_size, hidden_size, output_size)

# Train  modeland store the losses
losses = model.train(Xtrain, ytrain, 200, 0.005)

print(calculate_accuracy(Xtest, ytest))
print(calculate_accuracy(Xtrain, ytrain))
