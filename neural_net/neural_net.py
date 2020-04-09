import torch
import torch.nn as nn

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

        # unpack config
        sizes = [config['input_size'], *config['hidden_layer_sizes'], config['output_size']]

        # initialize hidden layer(s) and output layer
        hls = [nn.Linear(l_size, sizes[n+1], bias=True) for n, l_size in enumerate(sizes[:-2])]
        ol = nn.Linear(sizes[-2], sizes[-1], bias=True)
        
        self.model = nn.Sequential(*hls, ol, nn.LogSoftmax(dim=1))

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
