from torch.nn.modules.utils import _pair
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import pickle
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def neuralBalanceFCToFC(inl, oul, order=2, gradient=1):
    incoming = torch.linalg.norm(inl, dim=1, ord=order)
    outgoing = torch.linalg.norm(oul, dim=0, ord=order)
    optimal_l = (1 - gradient + gradient * torch.sqrt(outgoing/incoming))
    inl.data *= optimal_l.unsqueeze(1)
    oul.data /= optimal_l

def neuralBalanceLocalToLocal(inp, out, gradient=1):
    inweights = inp.clone().sum(dim=1).sum(dim=1).data
    outweights = out.clone().sum(dim=1).sum(dim=1).reshape(1, out.shape[3], out.shape[4], 3, 3)
    out = out.reshape(out.shape[0], out.shape[1], out.shape[2], out.shape[3], out.shape[4], 3, 3)
    for outer_y in tqdm(range(inweights.shape[1])):
        for outer_x in range(inweights.shape[2]):
            inner_y, inner_x = outer_y-1, outer_x-1
            valid_inner_x = list(i for i in range(inner_x-1, inner_x+2) if i >= 0 and i < outweights.shape[2])
            valid_inner_y = list(i for i in range(inner_y-1, inner_y+2) if i >= 0 and i < outweights.shape[1])
            sum=0
            for y_kernel in valid_inner_y:
                for x_kernel in valid_inner_x:
                    sum+=outweights[0][y_kernel][x_kernel][inner_y-y_kernel+1][inner_x-x_kernel+1]**2
            outg = torch.sqrt(sum).item()
            inc = inweights[0][outer_y][outer_x].square().sum(dim=-1)
            optimal_l = torch.sqrt(outg/inc)
            inp[0, :, :, outer_y, outer_x]*=(1 - gradient + gradient * optimal_l)
            for y_kernel in valid_inner_y:
                for x_kernel in valid_inner_x:
                            out.data[0,:,:,y_kernel,x_kernel,inner_y-y_kernel+1,inner_x-x_kernel+1]/=(1 - gradient + gradient * optimal_l)
    out = out.reshape(out.shape[0], out.shape[1], out.shape[2], out.shape[3], out.shape[4], 9)      


def neuralBalanceLocalToFC(inp, out, gradient=1):
    inw = inp.clone().sum(dim=1).sum(dim=1)
    for i in range(inw.shape[1]):
        for j in range(inw.shape[2]):
            inc = inw[0][i][j].square().sum()
            outg = torch.sum(torch.square(out[:,(i*inp.shape[1])+j])).item()/out.shape[0]
            optimal_l = torch.sqrt(outg/inc)
            inp[0, :, :, i, j]*=(1 - gradient + gradient * optimal_l)
            out[:,(i*inp.shape[1])+j]/=(1 - gradient + gradient * optimal_l)


class LocallyConnected2d(nn.Module):
    def __init__(self, in_channels, out_channels, output_size, kernel_size, stride, bias=True):
        super(LocallyConnected2d, self).__init__()
        self.output_size = _pair(output_size)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, *self.output_size, kernel_size**2)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, *self.output_size)
            )
        else:
            self.register_parameter('bias', None)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)

    def forward(self, x):
        _, c, h, w = x.size()
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out

class LocallyConnectedNetwork(nn.Module):
    def __init__(self):
        super(LocallyConnectedNetwork, self).__init__()
        self.lc1 = LocallyConnected2d(3, 3, 30, 3, 1) 
        self.lc2 = LocallyConnected2d(3, 3 , 28, 3, 1)  
        self.fc1 = nn.Linear(3*28*28, 1024)
        self.fc2 = nn.Linear(1024, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.lc1(x))
        x = self.relu(self.lc2(x))
        x = x.view(x.size(0), -1)  
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class CIFAR10:
    def __init__(self, batchsize, test=False, norm=True):
        # Define transformations
        # transforms.ToTensor() already converts the data to a tensor and scales it to [0, 1]
        transform_list = [transforms.ToTensor(), transforms.Grayscale(num_output_channels=3)]
        if norm:
            # Now normalize around the mean 0.5 and std 0.5 after scaling to [0, 1]
            transform_list.append(transforms.Normalize((0.5,), (0.5,)))
        
        transform = transforms.Compose(transform_list)

        # Load CIFAR10 data with transforms
        cifar10_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        cifar10_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

        # Extract labels for stratification
        labels = [label for _, label in cifar10_train]

        # Create a stratified sample of the data indices
        _, subset_indices = train_test_split(
            range(len(cifar10_train)),
            test_size=0.1,  # 10% of the data
            random_state=42,
            stratify=labels
        )

        # Use Subset to select part of the training data
        train_subset = Subset(cifar10_train, subset_indices)

        if not test:
            self.dataset = DataLoader(train_subset, batch_size=batchsize, shuffle=True)
        else:
            self.dataset = DataLoader(cifar10_test, batch_size=batchsize, shuffle=False)

train_loader = CIFAR10(64, test=False).dataset
test_loader = CIFAR10(64, test=True).dataset

model = LocallyConnectedNetwork().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

def train(model, train_loader, loss_fn, optimizer, epochs=500):
    hist = {}
    hist['train_loss'] = []
    hist['test_acc'] = []
    for epoch in range(epochs):
        if epochs%1 == 0:
            with torch.no_grad():
                neuralBalanceLocalToLocal(model.lc1.weight, model.lc2.weight, gradient=.2)
                neuralBalanceLocalToFC(model.lc2.weight, model.fc1.weight, gradient=.2)
                neuralBalanceFCToFC(model.fc1.weight, model.fc2.weight, gradient=.2)
        model.train()
        total_loss = 0
        for images, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch}/{epochs}, Training Loss: {avg_loss:.4f}')
        hist['train_loss'].append(avg_loss)
        hist['test_acc'].append(test(model, test_loader))
    return hist


def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
    return accuracy
# Train and test the model
history = train(model, train_loader, loss_fn, optimizer)
with open('/baldig/proteomics2/ian/Neural-Balance/personalExps/locallyConnected/hist/locally_connected-0_1-cifar_Gradient`.pkl', 'wb') as f:
    pickle.dump(history, f)