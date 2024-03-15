import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
from tqdm import tqdm
from collections import Counter
import torch.nn.functional as F
import time
from sklearn.utils import shuffle
import logging
import os
import sys
import time

from models import *


def train(model, trainloader = None, testloader = None, epochs = 0, loss_function = None, optimizer = None, neural_balance = True, l1_weight=0, l2_weight=0, random = 0.0, neural_balance_epoch = 1, order = 2, logger = None, device = 'cuda:0'):

    train_accuracies = []
    norms = []
    test_accuracies = []

    # for neural balance
    if neural_balance:   
        linear_layers = []
        for layer in model.layers:
            if(isinstance(layer, CustomLinear)):
                linear_layers.append(layer)
    try:

        # Run the training loop
        for epoch in range(epochs): # 5 epochs at maximum

            start_time = time.time()

            # Print epoch
            print(f'Starting epoch {epoch+1}')
            running_loss = 0.0
            correct_predictions = 0
            total_predictions = 0

            # Iterate over the DataLoader for training data
            for i, data in enumerate(trainloader, 0):

                # Get inputs
                inputs, targets = data
                inputs, targets = inputs.to(device), targets.to(device)

                # Zero the gradients
                optimizer.zero_grad()

                # Perform forward pass
                outputs = model(inputs)

                # Compute loss
                loss = loss_function(outputs, targets)

                # Specify L1 and L2 weights
                l1_weight = l1_weight#0.0001
                l2_weight = l2_weight#0.0001

                # Compute L1 and L2 loss component
                parameters = []
                for parameter in model.parameters():
                    parameters.append(parameter.view(-1))
                l1 = l1_weight * model.compute_l1_loss(torch.cat(parameters))
                l2 = l2_weight * model.compute_l2_loss(torch.cat(parameters))

                # Add L1 and L2 loss components
                loss += l1
                loss += l2

                # Perform backward pass
                loss.backward()

                # Perform optimization
                optimizer.step()

                # Print statistics
                running_loss += loss.item()

               # Get predictions from the maximum value
                _, predicted = torch.max(outputs.data, 1)

                # Increment the correct predictions counter
                correct_predictions += (predicted == targets).sum().item()

                # Increment the total predictions counter
                total_predictions += targets.size(0)

            if neural_balance:   
                if epoch%neural_balance_epoch==0:
                    for count, linear in enumerate(linear_layers):
                        if count==0:
                            continue

                        norm = linear.neural_balance(linear_layers[count-1], order = order, random = random)
                        norms.append(norm)


            # Calculate accuracy as the percentage of correct predictions
            accuracy = 100 * correct_predictions / total_predictions

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            # Print average loss and accuracy after every epoch
            logger.info(f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
            logger.info(f'Avg. Loss: {running_loss / len(trainloader):.5f}, Accuracy: {accuracy:.2f}%')

            # check on test set
            test_accuracies.append(test(model, testloader, logger, device))

    except KeyboardInterrupt:
        print('Interrupted')
        max_acc = max(test_accuracies)
        logger.info(f"Best val accuracy: {max_acc}")
        sys.exit(130)

    return test_accuracies, norms



def test(model, testloader, logger, device):
    # Testing the model
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    logger.info('Accuracy of the network on the 10000 test images: {:.2f} %'.format(100 * correct / total))
    return (100 * correct / total)




def log(file):
    
    path = 'logs/'
    log_file = os.path.join(path, file)
    
    if not os.path.exists('logs'):
        os.makedirs('logs')
      
    if not os.path.isfile(log_file):
        open(log_file, "w+").close()

    console_logging_format = "%(levelname)s %(message)s"
    file_logging_format = "%(levelname)s: %(asctime)s: %(message)s"

    # configure logger
    logging.basicConfig(level=logging.INFO, format=console_logging_format)
    logger = logging.getLogger()
    
    # create a file handler for output file
    handler = logging.FileHandler(log_file)

    # set the logging level for log file
    handler.setLevel(logging.INFO)
    
    # create a logging format
    formatter = logging.Formatter(file_logging_format)
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)

    return logger

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs