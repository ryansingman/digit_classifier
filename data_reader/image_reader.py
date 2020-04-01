import numpy as np
import struct

from data_reader.reader_exceptions import HeaderError

class ImageReader():
    """Reads images for given data set (MNIST database format)"""
    def __init__(self, datapath):
        """
        Initializes image reader
        Inputs:
            datapath -- path to image file
        """
        self.magic_number = 2051        # header 'magic number'

        self._open(datapath)        # opens data file        
        self._read_header()         # reads and verifies header
        self._gen_img_dtype()       # generates image dtype

        self.item_num = 0
        
    def __next__(self):
        """
        Returns next image in data set
        Returns:
            image -- integer value of label
        """
        if self.item_num < self.num_items:
            self.item_num += 1
            return self._read_image()
        else:
            self.close()
            raise StopIteration

    def __iter__(self):
        """Returns self"""
        return self

    def _read_image(self):
        """
        Reads image from file
        Returns:
            image_mat -- numpy matrix w/ image dtype from file
        """
        buf = self.image_file.read(self.num_rows * self.num_cols)
        return np.squeeze(np.frombuffer(buf, dtype = self.img_dtype))

    def _gen_img_dtype(self):
        """
        Generates image dtype
        """
        self.img_dtype = np.dtype((np.uint8, (self.num_rows, self.num_cols))).newbyteorder('>')

    def _open(self, datapath):
        """
        Opens file
        Inputs:
            datapath -- path to image file
        """
        self.image_file = open(datapath, 'rb')

    def _read_header(self):
        """
        Reads image file header and verifies
        """
        mag_num = int(struct.unpack('>i', self.image_file.read(4))[0])
        if mag_num != self.magic_number:
            raise HeaderError

        self.num_items = int(struct.unpack('>i', self.image_file.read(4))[0])
        self.num_rows = int(struct.unpack('>i', self.image_file.read(4))[0])
        self.num_cols = int(struct.unpack('>i', self.image_file.read(4))[0])

    def close(self):
        """
        Closes file and terminates
        """
        self.image_file.close()
