import torch
from data import DataPreprocessor
import torch.optim as optim

preprocessor = DataPreprocessor()
preprocessor.prepare_data()
preprocessor.to_tensors()
Xtrain, ytrain, Xtest, ytest = preprocessor.return_data()

input_size = 3 * 19 * 32
hidden_size = 512
output_size = 10

model = torch.nn.Sequential(
    torch.nn.Linear(input_size, hidden_size),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_size, hidden_size),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_size, hidden_size),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_size, 10),
    torch.nn.Softmax(dim=1),
)

loss_fn = torch.nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 100
y = None
for n in range(num_epochs):
    y = model(Xtrain)
    loss = loss_fn(y, ytrain)
    print(n + 1, loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

assert y is not None
assert isinstance(
    ytest, torch.Tensor
), f"ytest must be a torch.Tensor, got {type(ytest)}"

y_pred = model(Xtest)
predicted_class = torch.argmax(y_pred, dim=1).float()
print(y_pred.size())
print(ytest.size())
accuracy = torch.mean((predicted_class == ytest).float())
print(f"Test Accuracy: {accuracy.item() * 100:.2f}%")
