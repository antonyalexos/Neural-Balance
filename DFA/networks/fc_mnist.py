import torch
import torch.nn as nn

from utils.DFATrainingHook import DFATrainingHook
from utils.OutputTrainingHook import OutputTrainingHook


class Network(nn.Module):
    def __init__(self, batch_size, n_hidden, train_mode, device):
        super(Network, self).__init__()
        self.batch_size = batch_size
        self.train_mode = train_mode
        self.device = device

        last_input_size = 784
        fc_hidden = []
        bn_hidden = []
        for _ in range(n_hidden):
            fc_hidden.append(nn.Linear(last_input_size, 800).to(device))
            last_input_size = 800
        fc_dfa = [DFATrainingHook(train_mode, device) for _ in range(n_hidden)]
        self.fc_hidden = nn.ModuleList(fc_hidden)
        self.fc_dfa = nn.ModuleList(fc_dfa)

        self.fc_out = nn.Linear(800, 10).to(device)
        self.output_hook = OutputTrainingHook(train_mode)


    def forward(self, x):
        grad_at_output = torch.zeros([x.shape[0], 10], requires_grad=False).to(self.device)
        dir_der_at_output = torch.zeros([x.shape[0], 10], requires_grad=False).to(self.device)

        x = torch.flatten(x, 1)

        for fc, fc_dfa in zip(self.fc_hidden, self.fc_dfa):
            x = fc(x)
            x = nn.functional.relu(x)
            x = fc_dfa(x, dir_der_at_output, grad_at_output)

        x = self.fc_out(x)

        x = self.output_hook(x, dir_der_at_output, grad_at_output)

        return x