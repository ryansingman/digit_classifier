import matplotlib
matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt

import numpy as np
import torch
from torchvision import datasets, transforms

from tqdm import tqdm
import pdb

from neural_net.neural_net import NeuralNet
from neural_net.trainer import Trainer

DISPLAY_FLAG = False 
DEBUG_MODE = False 

IMG_SIZE = 784
BATCH_SIZE = 64

if __name__ == '__main__':
    # define image transformation
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    # read training labels and images
    trainset = datasets.MNIST('dataset/', download=True, \
                               train=True, transform=transform) 
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = BATCH_SIZE, shuffle=True)

    # set up neural net and optimizer config
    nn_config = {'input_size': IMG_SIZE, 'output_size': 10, \
                 'num_hidden_layers': 5, 'hidden_layer_sizes': (1000, 256, 128, 64, 32)}
    optim_config = {'learning_rate': 0.005, 'momentum': 0.9}

    # initialize nn and trainer
    nn = NeuralNet(nn_config)
    trainer = Trainer(nn, optim_config)

    # train model
    with tqdm(total = len(trainloader)) as pbar:
        for ii, (imgs, labels) in enumerate(trainloader):

            loss = trainer.train(imgs.view(imgs.shape[0], -1), labels)

            if DEBUG_MODE and (ii % 10) == 9:
                print(loss)

            pbar.update(1)

    # read training labels and images
    testset = datasets.MNIST('dataset/', download=True, \
                               train=False, transform=transform) 
    testloader = torch.utils.data.DataLoader(testset, batch_size = 1, shuffle=True)

    num_correct = 0
    with tqdm(total = len(testloader)) as pbar:
        for img, label in testloader:
            prediction = nn.predict(img.view(img.shape[0], -1))

            # display testing image in mpl plot
            if DISPLAY_FLAG:
                plt.imshow(img.reshape(28, 28), cmap='gray')
                plt.title("Testing Image: {} (truth) {} (prediction)".format(label.item(), prediction))
                plt.show()

            if prediction == label.item():
                num_correct += 1

            pbar.update(1)

    print(num_correct / len(testloader))
