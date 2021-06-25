import torch
from torchtext.legacy import data
from torchtext.legacy.data import Field, BucketIterator

import pandas as pd
import os

from .SeqDataset import SeqDataset

class SSTDataset(SeqDataset):
    def __init__(self, data_path, seed, batch_size, device, split_ratio=[0.7, 0.3]):
        # super(QuoraDataset, self).__init__(data_path, seed, batch_size, device, split_ratio)
        self.split_ratio = split_ratio
        self.data_path = data_path
        self.seed = seed
        self.device = device
        self.batch_size = batch_size

        self.ranges = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        self.labels = ['very negative', 'negative', 'neutral', 'positive', 'very positive']
        self.label = [0, 1, 2, 3, 4]

        self.seq_data = self.load_data(self.data_path)
    
    def get_labels(self):
        return self.labels

    def load_data(self, sst_path):
        sst_sents = pd.read_csv(os.path.join(sst_path, 'datasetSentences.txt'), delimiter='\t')
        sst_phrases = pd.read_csv(os.path.join(sst_path, 'dictionary.txt'), delimiter='|', names=['phrase','phrase_id'])
        sst_labels  = pd.read_csv(os.path.join(sst_path, 'sentiment_labels.txt'), delimiter='|')
        
        sst_sentences_phrases = pd.merge(sst_sents, sst_phrases, how='inner', left_on=['sentence'], right_on=['phrase'])
        sst = pd.merge(sst_sentences_phrases, sst_labels, how='inner', left_on=['phrase_id'], right_on=['phrase ids'])[['sentence','sentiment values']]
        sst['labels'] = pd.cut(sst['sentiment values'], bins=self.ranges, labels=self.labels, include_lowest=True)
        sst_data = sst[['sentence', 'labels']]
        sst_data.columns = ['src','trg']

        return sst_data
    
    def create_dataset(self):
        try:
            SRC = Field(sequential = True, tokenize = 'spacy', batch_first =True, include_lengths=True)
            TRG = data.LabelField(tokenize ='spacy', is_target=True, batch_first =True, sequential =True)
            fields = [('src', SRC),('trg', TRG)]
            example = [data.Example.fromlist([self.seq_data.src[i],self.seq_data.trg[i]], fields) for i in range(self.seq_data.shape[0])] 
            
            # Creating dataset
            Dataset = data.Dataset(example, fields)
            (self.train_data, self.test_data) = Dataset.split(split_ratio=self.split_ratio, random_state=random.seed(self.seed))
            
            print(f"Number of training examples: {len(self.train_data.examples)}")
            print(f"Number of testing examples: {len(self.test_data.examples)}")

            # build vocabulary
            SRC.build_vocab(self.train_data,  
                 vectors = "glove.6B.100d", 
                 unk_init = torch.Tensor.normal_)

            TRG.build_vocab(self.train_data)

            print(f"Unique tokens in source vocabulary: {len(SRC.vocab)}")
            print(f"Unique tokens in target vocabulary: {len(TRG.vocab)}")

            return self.train_data, self.test_data, SRC, TRG
        except Exception as e:
            raise e
    