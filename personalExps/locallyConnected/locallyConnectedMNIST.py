from torch.nn.modules.utils import _pair
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pickle

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def neuralBalanceLocalToLocal(inp, out):
    inweights = inp.data
    outweights = out.reshape(1, out.shape[3], out.shape[4], 3, 3)
    for outer_y in range(inweights.shape[3]):
        for outer_x in range(inweights.shape[4]):
            inner_y, inner_x = outer_y-1, outer_x-1
            valid_inner_x = list(i for i in range(inner_x-1, inner_x+2) if i >= 0 and i < outweights.shape[2])
            valid_inner_y = list(i for i in range(inner_y-1, inner_y+2) if i >= 0 and i < outweights.shape[1])
            sum=0
            for y_kernel in valid_inner_y:
                for x_kernel in valid_inner_x:
                    sum+=outweights[0][y_kernel][x_kernel][inner_y-y_kernel+1][inner_x-x_kernel+1]**2
            outg = torch.sqrt(sum).item()
            inc = inweights[0][0][0][outer_y][outer_x].square().sum(dim=-1)
            optimal_l = torch.sqrt(outg/inc)
            inweights[0][0][0][outer_y][outer_x]*=optimal_l
            for y_kernel in valid_inner_y:
                for x_kernel in valid_inner_x:
                    outweights[0][y_kernel][x_kernel][inner_y-y_kernel+1][inner_x-x_kernel+1]/=optimal_l

    inp.data = inweights
    out.data = outweights.reshape(1, 1, 1, out.shape[3], out.shape[4], 9)

def neuralBalanceLocalToFC(inp, out):
    for i in range(inp.shape[3]):
        for j in range(inp.shape[4]):
            inc = inp[0][0][0][i][j].square().sum()
            outg = torch.sum(torch.square(out[:,(i*inp.shape[3])+j])).item()/out.shape[0]
            optimal_l = torch.sqrt(outg/inc)
            inp[0][0][0][i][j]*=optimal_l
            out[:,(i*inp.shape[3])+j]/=optimal_l


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
        self.lc1 = LocallyConnected2d(1, 1, 26, 3, 1) 
        self.lc2 = LocallyConnected2d(1, 1, 24, 3, 1)  
        self.fc = nn.Linear(1 * 24 * 24, 10)

    def forward(self, x):
        x = self.lc1(x)
        x = self.lc2(x)
        x = x.view(x.size(0), -1)  
        x = self.fc(x)
        return x

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = LocallyConnectedNetwork().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, train_loader, loss_fn, optimizer, epochs=50):
    hist = {}
    hist['train_loss'] = []
    hist['test_acc'] = []
    for epoch in range(epochs):
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
        with torch.no_grad():
            neuralBalanceLocalToLocal(model.lc1.weight, model.lc2.weight)
            neuralBalanceLocalToFC(model.lc2.weight, model.fc.weight)
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
with open('/baldig/proteomics2/ian/Neural-Balance/personalExps/locallyConnected/hist/locally_connected_mnist_NB.pkl', 'wb') as f:
    pickle.dump(history, f)