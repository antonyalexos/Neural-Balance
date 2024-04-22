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

seed = 42
set_seed(seed)

input_size = 784 
hidden_size1 = 256
hidden_size2 = 64
num_classes = 10
num_epochs = 100
batch_size = 64
learning_rate = 0.01
weight_decay = 0.01

# MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def neuronalNeuralBalance(inl, oul):

    ninc = torch.zeros_like(inl)
    noul = torch.zeros_like(oul)

    for i in range(inl.data.shape[0]):

        inc = torch.sqrt(torch.sum(torch.square(inl.data[i]))).item()
        outg = torch.sqrt(torch.sum(torch.square(oul.data[:,i]))).item()

        opt = np.sqrt(outg/inc)

        ninc[i] = inl.data[i]*opt
        noul[:, i] = oul.data[:,i]/opt

    inl.data = ninc
    oul.data = noul

train_dataset = torchvision.datasets.MNIST(os.getcwd(), train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(os.getcwd(), train=False, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(784, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 128)
        self.fc7 = nn.Linear(128, 64)
        self.fc8 = nn.Linear(64, 32)
        self.fc9 = nn.Linear(32, 16)
        self.fc10 = nn.Linear(16, 10)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.fc5(out)
        out = self.relu(out)
        out = self.fc6(out)
        out = self.relu(out)
        out = self.fc7(out)
        out = self.relu(out)
        out = self.fc8(out)
        out = self.relu(out)
        out = self.fc9(out)
        out = self.relu(out)
        out = self.fc10(out)
        return out

model = NeuralNet().to(device)

criterion = nn.CrossEntropyLoss()
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

        # l2_norm = 0.0
        # for param in model.parameters():
        #     l2_norm += torch.norm(param, p=2) ** 2  # p=2 corresponds to the L2 norm
        # l2_norm = torch.sqrt(l2_norm)
        # loss+=weight_decay*l2_norm

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss / len(train_loader):.4f}')

    
    neuronalNeuralBalance(model.fc1.weight, model.fc2.weight)
    neuronalNeuralBalance(model.fc2.weight, model.fc3.weight)
    neuronalNeuralBalance(model.fc3.weight, model.fc4.weight)
    neuronalNeuralBalance(model.fc4.weight, model.fc5.weight)
    neuronalNeuralBalance(model.fc5.weight, model.fc6.weight)
    neuronalNeuralBalance(model.fc6.weight, model.fc7.weight)
    neuronalNeuralBalance(model.fc7.weight, model.fc8.weight)
    neuronalNeuralBalance(model.fc8.weight, model.fc9.weight)
    neuronalNeuralBalance(model.fc9.weight, model.fc10.weight)
        
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

with open('/baldig/proteomics2/ian/Neural-Balance/personalExps/hist/XLargeMnistModelNBHist.pkl', 'wb') as f:
    pickle.dump(hist, f)

torch.save(model.state_dict(), '/baldig/proteomics2/ian/Neural-Balance/personalExps/models/XLargeMnistModelNB.pt')



    


