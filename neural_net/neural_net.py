import torch
import torch.nn as nn

import pdb

class NeuralNet(nn.Module):
    """
    Neural network implementation
    """
    def __init__(self, config):
        """
        Initializes neural net
        Inputs:
            config -- configuration dictionary
        """
        super(NeuralNet, self).__init__()

        self.config = config
        self.training_mode = True

        # generate model
        self.gen_model()

    def gen_model(self):
        """
        Generates model based on config and training/testing
        Side Effects:
            self.model -- generates / updates model
        """
        # unpack config
        sizes = [self.config['input_size'], *self.config['hidden_layer_sizes'], self.config['output_size']]

        # initialize hidden layer(s) and output layer
        if self.training_mode:
            self.hls = [nn.Linear(l_size, sizes[n+1], bias=True) for n, l_size in enumerate(sizes[:-2])]
            self.relus = [nn.ReLU(inplace=True)] * len(self.hls) 
            self.ol = nn.Linear(sizes[-2], sizes[-1], bias=True)
            self.dropouts = [nn.Dropout(self.config['dropout'])] * len(self.hls)

            layers = [layer for layers in zip(self.hls, self.dropouts, self.relus) for layer in layers]

        else:
            layers = [layer for layers in zip(self.hls, self.relus) for layer in layers]

        self.model = nn.Sequential(*layers, self.ol, nn.LogSoftmax(dim=1))
            

    def forward(self, x):
        """
        Forward propagation
        Inputs:
            x -- input tensor
        Returns:
            y -- output tensor
        """
        return self.model(x)

    def predict(self, x):
        """
        Predicts digit in img
        Inputs:
            x -- input image
        Returns:
            prediction -- digit prediction
        """
        with torch.no_grad():
            output = self.forward(x)
            return torch.argmax(output)

    def start_testing_mode(self):
        """
        Removes training layers from model for testing
        Side Effects:
            updates self.model
        """
        self.training_mode = False
        self.gen_model() 
