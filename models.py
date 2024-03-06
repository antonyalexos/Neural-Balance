import os
import torch
from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms

from utils import *


class CustomLinear(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(CustomLinear, self).__init__()
        # Use the nn.Linear layer internally
        self.linear = nn.Linear(input_features, output_features, bias)
        
    @property
    def weight(self):
        return self.linear.weight

    def forward(self, input):
        # Forward pass through the internal nn.Linear layer
        return self.linear(input)
    
    def neural_balance(self, previous_layer, return_norms = False, ord = 2, random = 0.1):
        shape = previous_layer.weight.shape[0]
        norm = []
        
        if random:
            random_index_num = int(random*shape)
            random_indexes = np.random.choice(shape, random_index_num, replace=False)
            random_indexes_tensor = torch.from_numpy(random_indexes).long().to(previous_layer.weight.device)
            
            incoming = torch.linalg.norm(previous_layer.weight[random_indexes_tensor], dim=1, ord=2)
            outgoing = torch.linalg.norm(self.linear.weight[:, random_indexes_tensor], dim=0, ord=2)
            optimal_l = torch.sqrt(outgoing/incoming)
            previous_layer.weight[random_indexes_tensor].data *= optimal_l.unsqueeze(1)
            self.linear.weight[:,random_indexes_tensor].data /= optimal_l

        else:
            incoming = torch.linalg.norm(previous_layer.weight, dim=1, ord=2)
            outgoing = torch.linalg.norm(self.linear.weight, dim=0, ord=2)
            optimal_l = torch.sqrt(outgoing/incoming)
            previous_layer.weight.data *= optimal_l.unsqueeze(1)
            self.linear.weight.data /= optimal_l
            
        if return_norms:
            norm.append(torch.linalg.norm(incoming/outgoing, dim = 0, ord=2))
            return torch.mean(torch.stack(norm))
        else: return torch.tensor([])
            
    
    
class MLP(nn.Module):
    '''
    Multilayer Perceptron.
    '''
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
          nn.Flatten(),
          CustomLinear(28 * 28 * 1, 256),
          nn.ReLU(),
          CustomLinear(256, 128),
          nn.ReLU(),
          CustomLinear(128, 64),
          nn.ReLU(),
          CustomLinear(64, 32),
          nn.ReLU(),
          CustomLinear(32, 16),
          nn.ReLU(),
          CustomLinear(16, 10)
        )


    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)

    def compute_l1_loss(self, w):
        return torch.abs(w).sum()

    def compute_l2_loss(self, w):
        return torch.square(w).sum()
            
            
            
            
            
            
            