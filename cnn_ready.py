import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import optim
from torch import nn
from torch._higher_order_ops import out_dtype
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchmetrics
from data import DataPreprocessor


class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        """
        Building blocks of convolutional neural network.

        Parameters:
            * in_channels: Number of channels in the input image (for grayscale images, 1)
            * num_classes: Number of classes to predict. In our problem, 10 (i.e digits from  0 to 9).
        """
        super(CNN, self).__init__()

        # 1st convolutional layer
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=16, kernel_size=3, padding=1
        )
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 2nd convolutional layer
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1
        )
        # Fully connected layer
        self.fc1 = nn.Linear(32 * 8 * 8, num_classes)

    def forward(self, x):
        """
        Define the forward pass of the neural network.

        Parameters:
            x: Input tensor.

        Returns:
            torch.Tensor
                The output tensor after passing through the network.
        """
        x = F.relu(self.conv1(x))  # Apply first convolution and ReLU activation
        x = self.pool(x)  # Apply max pooling
        x = F.relu(self.conv2(x))  # Apply second convolution and ReLU activation
        x = self.pool(x)  # Apply max pooling
        x = torch.flatten(x)  # Flatten the tensor
        x = self.fc1(x)  # Apply fully connected layer
        return x


def classes_to_weights(ytrain, nc=10):
    classes = np.array(ytrain).astype(int)
    weights = np.bincount(classes, minlength=nc)
    weights[weights == 0] = 1
    weights = 1 / weights
    weights /= weights.sum()
    return torch.Tensor(weights)


preprocessor = DataPreprocessor()
preprocessor.to_tensors()
Xtrain, ytrain, Xtest, ytest = preprocessor.return_data()
batch_size = 256
Xtrain = Xtrain[0:batch_size, :, :, :]
ytrain = ytrain[0:batch_size]
print(ytrain)

device = "cpu"

model = CNN(in_channels=3, num_classes=10).to(device)

criterion = nn.CrossEntropyLoss(weight=classes_to_weights(ytrain))
optimizer = optim.Adam(model.parameters(), lr=0.001)

length1 = len(Xtrain)
num_epochs = 100
for epoch in range(num_epochs):
    running_loss = 0.0

    for i in range(length1):
        data = torch.squeeze(Xtrain[i]).to(device)
        targets = ytrain[i].to(device)
        scores = model(data)
        loss = criterion(scores, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * length1

    print(
        f"Epoch [{epoch + 1}/{num_epochs}]----------------------------------------------------------------"
    )
    epoch_loss = running_loss / length1
    print(f"--- Epoch {epoch+1} FINISHED. Avg. Epoch Loss: {epoch_loss:.4f} ---")


length2 = len(Xtrain)
count = 0
for i in range(length2):
    data = torch.squeeze(Xtest[i]).to(device)
    targets = ytest[i].to(device)
    outputs = model(data)
    _, pred = torch.topk(outputs, 1)
    pred = torch.squeeze(pred)
    print(f"{pred}-{ytest[i]}, ")
    if pred == ytest[i]:
        count += 1

print(f"Accuracy: {count/length2}")
