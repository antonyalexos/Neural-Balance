from datetime import datetime
import argparse
from os.path import join, exists, split
import os
import sys

import numpy as np
import time
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.utils import shuffle
from keras.preprocessing.text import Tokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from models import *

# Construct the argument parser
ap = argparse.ArgumentParser()

# Add the arguments to the parser
ap.add_argument("--epochs", required=False, default = 50, type=int,
   help="Batch size for model")
ap.add_argument("--lr", required=False, default=1e-3, type=float,
   help="constant learning rate for model")
ap.add_argument("--dataset", required=False, default = 'mnist', type=str, choices = ['mnist', 'cifar10'], help="choose dataset")
ap.add_argument("--gpu", required=False, default = '0', type=str,
   help="Choose GPU to use")
ap.add_argument("--batch_size", required=False, default = 256, type=int,
   help="Choose batch_size for the dataset")
ap.add_argument("--l1_weight", required=False, default = 0, type=float,
   help="Multiplier for L1 Regularizer")
ap.add_argument("--l2_weight", required=False, default = 0, type=float,
   help="Multiplier for L2 Regularizer")
ap.add_argument("--seed", required=False, default = 1, type=int,
   help="Choose seed")
ap.add_argument("--neural_balance", required=False, default = 1, type=int,
   help="Whether we train with neural balance or not")
ap.add_argument("--random", required=False, default = 0.0, type=float,
   help="Whether we train we balance only some neurons randomly")
ap.add_argument("--neural_balance_epoch", required=False, default = 1, type=int,
   help="Every how many epochs we are doing neural balance.")
ap.add_argument("--order", required=False, default = 2, type=int,
   help="Order of norm when doing neural balance.")

args, leftovers = ap.parse_known_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

# create the log file 

if not os.path.exists('logs'):
    os.makedirs('logs')

# dd/mm/YY H:M:S
now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
file_folder = 'logs/' + dt_string
if not os.path.exists(file_folder):
    os.mkdir(file_folder)
logger = log(file = dt_string+"/"+dt_string+".logs")

for arg in vars(args):
    logger.info("{} = {}".format(arg, getattr(args, arg)))

## This is where the fun begins
if args.gpu=='cpu':
    device = 'cpu'
elif args.gpu=='cuda':
    device = 'cuda'
else:
    device = 'cuda:'+args.gpu
    
device = torch.device(device)

logger.info(f"Training on {device}")

if args.dataset:
    # Prepare MNIST dataset
    train_dataset = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
    test_dataset = MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)

# Initialize the MLP
mlp = MLP().to(device)

# print(model)
# print(f"The model has {trainable:,} trainable parameters and {frozen:,} frozen parameters")
logger.info("Model Summary = {}".format(mlp))

# logger.info(f"The model has {trainable:,} trainable parameters and {frozen:,} frozen parameters")



# Define the loss function and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=args.lr)


# The actual training and validation:

history = {
    "t_loss" : [],
    "v_loss" : [],
    "t_acc" : [],
    "v_acc" : []
}


    
test_accuracies, norms = train(model = mlp, trainloader = trainloader, testloader = testloader, epochs = args.epochs, loss_function = loss_function, optimizer = optimizer, neural_balance = args.neural_balance, l1_weight=args.l1_weight, l2_weight=args.l2_weight, random = args.random, neural_balance_epoch = args.neural_balance_epoch, order = args.order, logger = logger, device = device)

max_acc = max(test_accuracies)
logger.info(f"Best val accuracy: {max_acc}")
np.save(file_folder+ "/test_accuracies.npy", test_accuracies)

torch.save(mlp.state_dict(), file_folder + '/model_weights.pth')


#     history["t_loss"].append(train_loss)
#     history["v_loss"].append(valid_loss)
#     history["t_acc"].append(train_acc)
#     history["v_acc"].append(valid_acc)

    # Saves best only

#     if valid_loss < best_valid_loss:
#         best_valid_loss = valid_loss
#         torch.save(model.state_dict(), f"{file_folder}/model_{epoch+1}.pt")

    # Print details about each epoch:
#     logger.info(f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
#     logger.info(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
# #     print(f"\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%")
#     logger.info(f"\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%")
#     if args.dataset=='cola':
#         logger.info(f"\t Matthews Correlation Coeffecient: {mcc*100:.2f}%")
#     elif args.dataset=='sts-b':
#         logger.info(f"\t Matthews Correlation Coeffecient: {mcc*100:.2f}%")
            

