import matplotlib
matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt

from data_reader.label_reader import LabelReader
from data_reader.image_reader import ImageReader

import pdb

DATASET_PATH = 'dataset/'
TRAINING_LABEL_FILE = 'train-labels-idx1-ubyte'
TRAINING_IMAGE_FILE = 'train-images-idx3-ubyte'

if __name__ == '__main__':
    # read training labels
    training_label_reader = LabelReader(datapath = DATASET_PATH + TRAINING_LABEL_FILE)

    # read training images
    training_image_reader = ImageReader(datapath = DATASET_PATH + TRAINING_IMAGE_FILE)

    # display training image in mpl plot
    for label, img in zip(training_label_reader, training_image_reader):
        plt.imshow(img, cmap='gray')
        plt.title("Training Image: {}".format(label))
        plt.show()
