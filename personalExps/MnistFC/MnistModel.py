import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
import os
from sklearn.model_selection import train_test_split

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

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
hidden_size = [256, 128]
model_name = '256-128'
num_classes = 10
num_epochs = 100
batch_size = 64
learning_rate = 0.01
weight_decay = 0.0001   
nb_order = 2
l2 = True
nb = False
seed = 55
fullBalanceAtStart = False
typeOfTraining = 'Clean'

class MNIST:
    def __init__(self, batchsize, test=False, norm=True):
        # Define transformations
        # transforms.ToTensor() already converts the data to a tensor and scales it to [0, 1]
        transform_list = [transforms.ToTensor()]
        if norm:
            # Now normalize around the mean 0.5 and std 0.5 after scaling to [0, 1]
            transform_list.append(transforms.Normalize((0.5,), (0.5,)))
        
        transform = transforms.Compose(transform_list)

        # Load CIFAR10 data with transforms
        cifar10_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        cifar10_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

        # Extract labels for stratification
        # labels = [label for _, label in cifar10_train]

        # Create a stratified sample of the data indices\


        # _, subset_indices = train_test_split(
        #     range(len(cifar10_train)),
        #     test_size=0.01,  # 1% of the data
        #     random_state=seed,
        #     stratify=labels
        # )

        # Use Subset to select part of the training data


        # train_subset = Subset(cifar10_train, subset_indices)

        if not test:
            self.dataset = DataLoader(cifar10_train, batch_size=batchsize, shuffle=True)
        else:
            self.dataset = DataLoader(cifar10_test, batch_size=batchsize, shuffle=False)

def neuralBalance(inl, oul, order):
    incoming = torch.linalg.norm(inl.weight, dim=1, ord=order)
    outgoing = torch.linalg.norm(oul.weight, dim=0, ord=order)
    optimal_l = torch.sqrt(outgoing/incoming)
    inl.weight.data *= optimal_l.unsqueeze(1)
    oul.weight.data /= optimal_l

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

for iteration in range(1):

    set_seed(seed+iteration)

    train_loader = MNIST(64, test=False).dataset
    test_loader = MNIST(64, test=True).dataset

    model = NeuralNet(input_size, hidden_size, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    if l2:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    hist = {}
    hist['train_loss'] = []
    hist['test_loss'] = []
    hist['test_acc'] = []

    lay = list(model.layers.children())
    if fullBalanceAtStart:
        while(True):
            restart=False
            for i in range(len(lay)-1):
                lay1, lay2 = lay[i], lay[i+1]
                incoming = torch.linalg.norm(lay1.weight, dim=1, ord=nb_order)
                outgoing = torch.linalg.norm(lay2.weight, dim=0, ord=nb_order)
                optimal_l = torch.sqrt(outgoing/incoming).sum()/incoming.shape[0]
                print(optimal_l)
                if optimal_l > 1.001 or optimal_l < .999:
                    restart=True
                neuralBalance(lay1, lay2, order = nb_order)
            if not restart:
                break

    for epoch in range(num_epochs):
        train_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            train_loss+=loss.item()

            optimizer.zero_grad()

            (for p in self.parameters()) and add (p**2).sum()

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

    with open(f'/baldig/proteomics2/ian/Neural-Balance/personalExps/MnistFC/hist/MNIST_{model_name}_NbAtStart_{fullBalanceAtStart}_{typeOfTraining}_iteration_{iteration}.pkl', 'wb') as f:
        pickle.dump(hist, f)

    torch.save(model.state_dict(), f'/baldig/proteomics2/ian/Neural-Balance/personalExps/MnistFC/models/MNIST_{model_name}_NbAtStart_{fullBalanceAtStart}_{typeOfTraining}_iteration_{iteration}.pt')



    


