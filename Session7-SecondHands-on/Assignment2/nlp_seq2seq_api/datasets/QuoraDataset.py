import torch
from torchtext.legacy import data
from torchtext.legacy.data import Field, BucketIterator

import pandas as pd

from .SeqDataset import SeqDataset

class QuoraDataset(SeqDataset):
    def __init__(self, data_path, seed, batch_size, device, split_ratio=[0.7, 0.3]):
        # super(QuoraDataset, self).__init__(data_path, seed, batch_size, device, split_ratio)
        self.split_ratio = split_ratio
        self.data_path = data_path
        self.seed = seed
        self.device = device
        self.batch_size = batch_size
        self.seq_data = self.load_data(self.data_path)

    def load_data(self, data_path):
        seq_data = pd.read_csv(data_path, sep='\t')
        seq_data = seq_data[seq_data['is_duplicate']==1][['question1','question2']]
        seq_data.reset_index(drop=True,inplace=True)
        seq_data.columns = ['src','trg']

        return seq_data
    