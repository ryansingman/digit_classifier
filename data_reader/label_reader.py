import numpy as np
import struct

from data_reader.reader_exceptions import HeaderError

class LabelReader():
    """Reads labels for given data set (MNIST database format)"""
    def __init__(self, datapath):
        """
        Initializes label reader
        Inputs:
            datapath -- path to label file
        """
        self.magic_number = 2049        # header 'magic number'

        self._open(datapath)        # opens data file        
        self._read_header()         # reads and verifies header

        self.item_num = 0
        
    def __next__(self):
        """
        Returns next label in data set
        Returns:
            label -- integer value of label
        """
        if self.item_num < self.num_items:
            self.item_num += 1
            return int.from_bytes(struct.unpack('>c', self.label_file.read(1))[0], byteorder='little')
        else:
            self.close()
            raise StopIteration

    def __iter__(self):
        """Returns self"""
        return self

    def _open(self, datapath):
        """
        Opens file
        Inputs:
            datapath -- path to label file
        """
        self.label_file = open(datapath, 'rb')

    def _read_header(self):
        """
        Reads label file header and verifies
        """
        mag_num = int(struct.unpack('>i', self.label_file.read(4))[0])
        if mag_num != self.magic_number:
            raise HeaderError

        self.num_items = int(struct.unpack('>i', self.label_file.read(4))[0])

    def close(self):
        """
        Closes file and terminates
        """
        self.label_file.close()
