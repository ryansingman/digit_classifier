import torch
import torch.nn as nn
import torch.optim as optim

class Trainer():
    """
    Trains neural net
    """
    def __init__(self, neural_net, optim_config):
        """
        Initializes trainer 
        Inputs:
            neural_net -- neural net module to optimize
            optim_config -- optimizer configuration
        """
        self.nn = neural_net

        self.optim = optim.SGD(self.nn.parameters(), lr = optim_config['learning_rate'], momentum=optim_config['momentum'])

        self.criterion = nn.NLLLoss()

    def train(self, img_in, truth_out):
        """
        Trains model
        Inputs:
            img_in -- input image (as torch tensor)
            truth_out -- truth output (as torch tensor)
        """
        # zero gradients
        self.optim.zero_grad()

        # compute forward pass
        pred_out = self.nn.forward(img_in)

        # compute loss
        loss = self.criterion(pred_out, truth_out)

        # perform backward pass and update weights and biases
        loss.backward()
        self.optim.step()

        return loss.item()
