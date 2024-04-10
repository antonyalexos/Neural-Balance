import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the neural network architecture
class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 10))
            
    def forward(self, x):
        return self.layers(x)

# Set device to CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NN().to(device) # Move model to CUDA device

# Function to calculate accuracy and loss
def accuracy_and_loss(model, loss_function, dataloader, device):
    model.eval() # Put the model in evaluation mode
    total_correct = 0
    total_loss = 0
    total_examples = 0
    n_batches = 0
    
    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device) # Move data to CUDA device
            outputs = model(images)
            batch_loss = loss_function(outputs, labels)
            n_batches += 1
            total_loss += batch_loss.item()
            _, predicted = torch.max(outputs, dim=1)
            total_examples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
    
    accuracy = total_correct / total_examples
    mean_loss = total_loss / n_batches
    return accuracy, mean_loss

def define_and_train(model, dataset_training, dataset_test, loss_function, n_epochs=500, device=device):
    trainloader = DataLoader(dataset_training, batch_size=1000, shuffle=True)
    testloader = DataLoader(dataset_test, batch_size=1000)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0)#1e-7)
    for epoch in range(n_epochs):
        model.train() # Put the model in training mode
        total_loss = 0
        total_correct = 0
        total_examples = 0
        n_mini_batches = 0
        
        for mini_batch in trainloader:
            images, labels = mini_batch[0].to(device), mini_batch[1].to(device) # Move data to CUDA device
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            
            n_mini_batches += 1
            total_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)
            total_examples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
        
        epoch_training_accuracy = total_correct / total_examples
        epoch_training_loss = total_loss / n_mini_batches
        epoch_val_accuracy, epoch_val_loss = accuracy_and_loss(model, loss_function, testloader, device)
        
        print(f'Epoch {epoch+1} loss: {epoch_training_loss:.3f} acc: {epoch_training_accuracy:.3f} val_loss: {epoch_val_loss:.3f} val_acc: {epoch_val_accuracy:.3f}')

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset_training = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataset_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
small_train_size = int(0.01 * len(dataset_training))  # 1% of the training data
small_dataset_training, _ = torch.utils.data.random_split(dataset_training, [small_train_size, len(dataset_training) - small_train_size])

# Instantiate the model and loss function, move model to CUDA if available
loss_function = nn.CrossEntropyLoss()

# Train the model
define_and_train(model, small_dataset_training, dataset_test, loss_function, device=device)