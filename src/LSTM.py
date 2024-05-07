import os
import torch
import numpy as np
import pandas as pd

from torch import nn,optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

batchsize = 100
training_data = datasets.   (/path/   , train = True, transform = transforms.ToTensor())
test_data = datasets.  (/path/   train = Flase, transform = transforms.ToTensor())

train_dataloader = DataLoader (training_data, batch_size = batchsize)
test_dataloader = DataLoader (test_data, batchsize = batchsize)

# define hyperparameters
sequence_len = 28
input_len = 28
hidden_size = 128
num_layers = 2
num_classes = 10
num_epochs = 5
learning rate = 0.01

class LSTM(nn.Module):
    def __init__(selfself, input_len, hidden_size, num_class, num_layers):
        super (LSTM, self). __init__()
        self.hidden_size  = hidden_size
        self.num_layers = num_layers
        self.lstm = nn. LSTM(input_len, hidden_size, num_layers, batch_first = True)
        self.output_layer = nn.Linear(hidden_size, num_classes)

    def forward(selfself, X):
        hidden_states = torch.zeros(self.num_layers, X.size(0), self.hidden_size)
        cell_states = torch.zeros(self.num_layers, X.size(0), self.hidden_size)
        out, _ = self.lstm(X, (hidden_states, cell_states))
        out = self.output_layer(out([:, -1, :]))
        return out
model = LSTM(input_len, hidden_size, num_classes, num_layers)
print(model)

loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD (model.paraeters(), lr=learning_rate)
def train(num_epochs, model, train_dataloader, loss_func):
    total_steps = len (train_dataloader)

    for epoch in range(num_epochs):
        for batch, (images, labels) in enumerate(train_dataloader):
            images = images.reshape (-1, sequence_len, input_len)

            output = model(images)
            loss = loss_func(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (batch+1)%100 == 0:
                print (f"Epoch: {epoch+1}; Batch {batch+1} / {total_steps}; Loss: {loss.item():>4f}")

train(num_epochs, model, train_dataloader, loss_func)

test_images, test_labels = next(iter(test_dataloader))
test_labels

test_output = model (test_images.view(-1, 28, 28))

predicted = torch.max(test_output, 1)
predicted

correct = [1 for i in range(100) if predicted[i] == test_images[i]]

percentage_correct = sum(correct)/100

percentage_correct

def test_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            #reshapeimages
            X = X.reshape(-1, 28, 28)
            pred = model (X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct/= size
    print (f"Test Error:\n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")
    return 100*correct

test_loop((test_dataloader, model, loss_func, adam))






