from .datasets import *
from .models import *
from .train_test import *
# from .utils import *

import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.legacy import data
from torchtext.legacy.data import Field, BucketIterator

import spacy
import numpy as np

import random
import math
import time
import os
import pickle

available_data = ['quora', 'wikipedia qa', 'ambig', 'commonsense']
data_dict = {'quora': QuoraDataset, 'wikipedia qa': WikiDataset, 'ambig': AmbigNqQADataset, 'commonsense': CommonsenseQADataset}

available_models = ['lstm encoder-decoder sequence model']
model_dict = {'lstm encoder-decoder sequence model': Seq2Seq}

class NLPSeq2SeqPipeline():
    def __init__(self, data_path, data_name, model_name, model_params, seed, batch_size, device, clip = 1, split_ratio=[0.7, 0.3]):
        self.split_ratio = split_ratio
        self.data_path = data_path
        self.seed = seed
        self.device = device
        self.batch_size = batch_size
        self.data_name = data_name
        self.model_name = model_name
        # self.epochs = epochs
        self.clip = clip
        assert self.data_name.lower() in available_data, f"Please select data from: {available_data}"
        assert self.model_name.lower() in available_models, f"Please select model from: {available_models}"
        print('Loading data...')
        self.train_iterator, self.test_iterator, self.SRC, self.TRG, self.orig_data = data_dict[self.data_name](self.data_path, self.seed, self.batch_size, self.device, self.split_ratio).iterate_dataset()
        print('Sample Data:-')
        print(self.orig_data.head())
        print('Data is loaded')
        print('\n')

        print('Loading model...')
        self.model_params = model_params
        self.model_params['input_dim'] = len(self.SRC.vocab)
        self.model_params['output_dim'] = len(self.TRG.vocab)
        self.model_params['device'] = self.device
        self.model = model_dict[self.model_name](**self.model_params).to(self.device)
        self.model.apply(self.init_weights)
        print('Model Loaded...')
        print('Model Structure:- ')
        print(self.model)
        print(f'The model has {self.count_parameters():,} trainable parameters')
        self.optimizer = optim.Adam(self.model.parameters())
        TRG_PAD_IDX = self.TRG.vocab.stoi[self.TRG.pad_token]
        self.criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)
        print('Model Built')

    def init_weights(self, m):
        for name, param in m.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def train_model(self, epochs, save_path):
        try:
            best_valid_loss = float('inf')
            for epoch in range(epochs):
                
                start_time = time.time()
                
                train_loss = train(self.model, self.train_iterator, self.optimizer, self.criterion, self.clip)
                valid_loss = evaluate(self.model, self.test_iterator, self.criterion)
                
                end_time = time.time()
                
                epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)
                
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    torch.save(self.model.state_dict(), os.path.join(save_path,f'epoch{epoch}_loss{best_valid_loss:.2f}_model.pt'))
                
                print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
                print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
                print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
        except Exception as e:
            raise e
    
    def get_model(self):
        try:
            return self.model
        except Exception as e:
            raise e

    def evaluate_model(self, model_path=None):
        try:
            if model_path is not None:
                test_loss = evaluate(self.load_model(model_path), self.test_iterator, self.criterion)
            else:
                test_loss = evaluate(self.model, self.test_iterator, self.criterion)
            print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
        except Exception as e:
            raise e
    
    def load_model(self, model_path):
        try:
            self.model.load_state_dict(torch.load(model_path))
            print('model loaded')
        except Exception as e:
            raise e
    
    def save_model(self, save_path, model_nm):
        try:
            torch.save(self.model.state_dict(), os.path.join(save_path,f'{model_nm}.pt'))
            print('model saved')
        except Exception as e:
            raise e
    
    def save_tokenizer_inv(self, base_path, src_tok_name, trg_tok_name=None, stoi=True):
        try:
            if stoi:
                with open(os.path.join(base_path, f'{src_tok_name}.pkl'), 'wb') as tokens: 
                    pickle.dump(self.SRC.vocab.stoi, tokens)

                if trg_tok_name is not None:
                    with open(os.path.join(base_path, f'{trg_tok_name}.pkl'), 'wb') as tokens:
                        pickle.dump(self.TRG.vocab.stoi, tokens)
            else:
                with open(os.path.join(base_path, f'{src_tok_name}.pkl'), 'wb') as tokens: 
                    pickle.dump(self.SRC.vocab.itos, tokens)

                if trg_tok_name is not None:
                    with open(os.path.join(base_path, f'{trg_tok_name}.pkl'), 'wb') as tokens:
                        pickle.dump(self.TRG.vocab.itos, tokens)
            print('tokenizers saved')
        except Exception as e:
            raise e
    
    def load_tokenizer(self, tok_file_path=None):
        try:
            tokenizer_file = open(tok_file_path, 'rb')
            tokenizer = pickle.load(tokenizer_file)

            return tokenizer
        except Exception as e:
            raise e
    
    