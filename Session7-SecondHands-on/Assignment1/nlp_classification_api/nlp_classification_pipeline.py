from .datasets import *
from .models import *
from .train_test import *
from .utils import *

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
import os, pickle

available_data = ['sst']
data_dict = {'sst': SSTDataset}

available_models = ['basic classification model']
model_dict = {'basic classification model': NLPBasicClassifier}

class NLPClassificationPipeline():
    def __init__(self, data_path, data_name, model_name, model_params, seed, batch_size, device, split_ratio=[0.7, 0.3]):
        self.split_ratio = split_ratio
        self.data_path = data_path
        self.seed = seed
        self.device = device
        self.batch_size = batch_size
        self.data_name = data_name
        self.model_name = model_name
        # self.epochs = epochs
        assert self.data_name.lower() in available_data, f"Please select data from: {available_data}"
        assert self.model_name.lower() in available_models, f"Please select model from: {available_models}"
        print('Loading data...')
        self.train_iterator, self.test_iterator, self.SRC, self.TRG, self.orig_data = data_dict[self.data_name](self.data_path, self.seed, self.batch_size, self.device, self.split_ratio).iterate_dataset()
        print('Data is loaded')
        print('\n')

        print('Loading model...')
        self.model_params = model_params
        self.model_params['vocab_size'] = len(self.SRC.vocab)
        self.model_params['output_dim'] = len(self.TRG.vocab)
        PAD_IDX = self.SRC.vocab.stoi[self.SRC.pad_token]
        UNK_IDX = self.SRC.vocab.stoi[self.SRC.unk_token]
        pretrained_embeddings = self.SRC.vocab.vectors

        self.model_params['pad_index'] = PAD_IDX
        self.model = model_dict[self.model_name](**self.model_params).to(self.device)
        self.model.embedding.weight.data.copy_(pretrained_embeddings)
        self.model.embedding.weight.data[UNK_IDX] = torch.zeros(self.model_params['embedding_dim'])
        self.model.embedding.weight.data[PAD_IDX] = torch.zeros(self.model_params['embedding_dim'])
        print('Model Loaded...')
        print('Model Structure:- ')
        print(self.model)
        print(f'The model has {self.count_parameters():,} trainable parameters')
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss()
        self.criterion = self.criterion.to(self.device)
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

    def train_model(self, epochs, save_path, save_name):
        try:
            best_valid_loss = float('inf')
            train_losses = []
            train_accs = []
            valid_losses = []
            valid_accs = []

            for epoch in range(epochs):
                start_time = time.time()
                # train the model
                train_loss, train_acc = train(self.model, self.train_iterator, self.optimizer, self.criterion)
                
                # evaluate the model
                valid_loss, valid_acc = evaluate(self.model, self.test_iterator, self.criterion)
                
                end_time = time.time()
                epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)

                # save the best model
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    torch.save(self.model.state_dict(), os.path.join(save_path, 'saved_weights.pt'))
                
                train_losses.append(train_loss)
                train_accs.append(train_acc)
                valid_losses.append(valid_loss)
                valid_accs.append(valid_acc)

                print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
                print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
                print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}% \n')

            torch.save(self.model.state_dict(), os.path.join(save_path, f'epoch{epochs}_{save_name}_saved_weights.pt'))

            return train_losses, train_accs, valid_losses, valid_accs
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
                test_loss, test_acc = evaluate(self.load_model(model_path), self.test_iterator, self.criterion)
            else:
                test_loss, test_acc = evaluate(self.model, self.test_iterator, self.criterion)
            print(f'| Test Loss: {test_loss:.3f} | Test Accuracy: {test_acc:.3f} |')

            self.conf_matrix()
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
    
    def conf_matrix(self):
        try:
            eval_df = evaluation_pred(self.model, self.test_iterator)
            print('Evaluation Metrics on Test Data:-')
            f1_macro, acc = print_accuracy(eval_df, 'trg', 'pred')
            print(f'F1 Macro Score: {f1_macro}')
            print(f'Accuracy: {acc} %')
            print('\n')
            print('Confusion Matrix:-')
            plot_confusion_matrix(eval_df['trg'].values.tolist(), eval_df['pred'].values.tolist(), classes=self.TRG.vocab.stoi.keys())
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
    
    def classify_text(self, text, categories=None, tok_file_path=None, model_path=None, device=None):
        try:
            if tok_file_path is not None:
                tokenizer = self.load_tokenizer(tok_file_path)
            else:
                tokenizer = self.SRC.vocab.stoi
            if model_path is None:
                model = self.model
            else:
                model = self.load_model(model_path)
            if device is None:
                device = self.device
            nlp = spacy.load('en')

            if categories is None:
                categories = self.TRG.vocab.itos
            # categories = {0:'very negative', 1:'negative', 2:'neutral', 3:'positive', 4:'very positive'}
            
            # tokenize the tweet 
            tokenized = [tok.text for tok in nlp.tokenizer(text)] 
            # convert to integer sequence using predefined tokenizer dictionary
            indexed = [tokenizer[t] for t in tokenized]        
            # compute no. of words        
            length = [len(indexed)]
            # convert to tensor                                    
            tensor = torch.LongTensor(indexed).to(device)   
            # reshape in form of batch, no. of words           
            tensor = tensor.unsqueeze(1).T  
            # convert to tensor                          
            length_tensor = torch.LongTensor(length)
            # Get the model prediction                  
            prediction = model(tensor, length_tensor)

            # _, pred = torch.max(prediction, 1) 
            pred = prediction.argmax(1, keepdim = True)
            
            return categories[pred.item()]
        except Exception as e:
            raise e
    
    def predict(self, text=None, categories=None, iterator=None, tok_file_path=None, model_path=None, device=None):
        try:
            if text is not None:
                pred = self.classify_text(self, text, categories, tok_file_path, model_path, device)
                return pred
            else:
                if tok_file_path is None:
                    tokenizer = self.SRC.vocab.itos
                else:
                    tokenizer = self.load_tokenizer(tok_file_path)
                if model_path is None:
                    model = self.model
                else:
                    model = self.load_model(model_path)
                if device is None:
                    device = self.device
                if categories is None:
                    categories = self.TRG.vocab.itos
                
                eval_df = evaluation_pred(model, iterator, itos=True, tokenizer=tokenizer)
                eval_df['pred'] = eval_df['pred'].replace(categories)

                return eval_df
        except Exception as e:
            raise e
    
    def get_classified_test_data(self, num=10, correct=True, categories=None, tok_file_path=None, model_path=None, device=None):
        try:
            if categories is None:
                categories = self.TRG.vocab.itos
            eval_df = self.predict(iterator=self.test_iterator, tok_file_path=tok_file_path, model_path=model_path, device=device)
            # all_orig_src = pd.DataFrame({'osrc': self.orig_data['src'], 'tokens': self.orig_data['src'].str.split(' ')})
            # print(all_orig_src)
            # print(eval_df.apply(lambda r: all_orig_src[all_orig_src['tokens'].apply(lambda x: set([w for w in r['src'] if w.isalpha()]).issubset(x))]['osrc'].values[0], axis=1))
            if correct:
                classified_texts = eval_df[eval_df['trg'] == eval_df['pred']].sample(num).reset_index(drop=True)
            else:
                classified_texts = eval_df[eval_df['trg'] != eval_df['pred']].sample(num).reset_index(drop=True)
            print("*"*40)
            if correct:
                print("***** Correctly Classified Text: *******")
            else:
                print("***** Incorrectly Classified Text: *******")
            print("*"*40)
            for i, (_, row) in enumerate(classified_texts.iterrows()):
                print(f"{i+1}) Text: {row['src']}")
                print(f"   Target Sentiment: {categories[row['trg']]}")
                print(f"   Predicted Sentiment: {categories[row['pred']]}")
                print()
        except Exception as e:
            raise e