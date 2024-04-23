import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import numpy as np
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

input_size = 784 
hidden_size = [256]
model_name = '256'
num_classes = 10
num_epochs = 100
batch_size = 64
learning_rate = 0.01
weight_decay = 0
nb_order = 2
l2 = False
nb = False
seed = 42


set_seed(seed)

# MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def neuralBalance(inl, oul, order):
    incoming = torch.linalg.norm(inl.weight, dim=1, ord=order)
    outgoing = torch.linalg.norm(oul.weight, dim=0, ord=order)
    optimal_l = torch.sqrt(outgoing/incoming)
    inl.weight.data *= optimal_l.unsqueeze(1)
    oul.weight.data /= optimal_l

train_dataset = torchvision.datasets.MNIST(os.getcwd(), train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(os.getcwd(), train=False, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.relu = nn.ReLU()
        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_size[0])])
        for i in range(len(hidden_size)-1):
            self.layers.append(nn.Linear(hidden_size[i], hidden_size[i+1]))
        self.layers.append(nn.Linear(hidden_size[len(hidden_size)-1], num_classes))
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.relu(layer(x))
        x = self.layers[-1](x)
        return x

model = NeuralNet(input_size, hidden_size, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
if l2:
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
else:
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

hist = {}
hist['train_loss'] = []
hist['test_loss'] = []
hist['test_acc'] = []

for epoch in range(num_epochs):
    train_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        train_loss+=loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss / len(train_loader):.4f}')

    if nb:
        lay = list(model.layers.children())
        for i in range(len(lay)-1):
            neuralBalance(lay[i], lay[i+1], order = nb_order)
        
    hist['train_loss'].append(train_loss / len(train_loader))

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        test_loss = 0
        for images, labels in test_loader:
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            test_loss += loss.item()

        avg_loss = test_loss / len(test_loader)

        hist['test_loss'].append(avg_loss)
        hist['test_acc'].append(100 * correct / total)

        print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')
        print(f'Loss of the network on the 10000 test images: {avg_loss:.4f}%')

import pickle

with open(f'personalExps\\MnistFC\\hist\\MnistModel-{model_name}-L2={l2}-l2Lambda={weight_decay}-nb={nb}.pkl', 'wb') as f:
    pickle.dump(hist, f)

torch.save(model.state_dict(), f'personalExps\\MnistFC\\models\\MnistModel-{model_name}-L2={l2}-l2Lambda={weight_decay}-nb={nb}.pt')



    


