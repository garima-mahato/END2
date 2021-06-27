
# Session 7 - Second Hands-On
---

## Assignment
---

1) Assignment 1 (500 points):

> 1) Submit the Assignment 5 as Assignment 1. To be clear,

>> 1) ONLY use datasetSentences.txt. (no augmentation required)

>> 2) Your dataset must have around 12k examples.

>> 3) Split Dataset into 70/30 Train and Test (no validation)

>> 4) Convert floating-point labels into 5 classes (0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0) 

>> 5) Upload to github and proceed to answer these questions asked in the S7 - Assignment Solutions, where these questions are asked:

>>> 1) Share the link to your github repo (100 pts for code quality/file structure/model accuracy)

>>> 2) Share the link to your readme file (200 points for proper readme file)

>>> 3) Copy-paste the code related to your dataset preparation (100 pts)

>>> 4) Share your training log text (you MUST have been testing for test accuracy after every epoch) (200 pts)

>>> 5) Share the prediction on 10 samples picked from the test dataset. (100 pts)

2) Assignment 2 (300 points):

> 1) Train model we wrote in the class on the following two datasets taken from this link (Links to an external site.): 

>> 1) http://www.cs.cmu.edu/~ark/QA-data/ (Links to an external site.)

>> 2) https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs (Links to an external site.)

> 2) Once done, please upload the file to github and proceed to answer these questions in the S7 - Assignment Solutions, where these questions are asked:

>> 1) Share the link to your github repo (100 pts for code quality/file structure/model accuracy) (100 pts)

>> 2) Share the link to your readme file (100 points for proper readme file), this file can be the second part of your Part 1 Readme (basically you can have only 1 Readme, describing both assignments if you want) (100 pts)

>> 3) Copy-paste the code related to your dataset preparation for both datasets.  (100 pts)

## Solution
---

## Assignment 1
---

[Link to GitHub Code](https://github.com/garima-mahato/END2/blob/main/Session7-SecondHands-on/Assignment1/END2_Session7_Assignment1_SentimentClassification_On_SSTDataset.ipynb)

[Link to Colab Code](https://githubtocolab.com/garima-mahato/END2/blob/main/Session7-SecondHands-on/Assignment1/END2_Session7_Assignment1_SentimentClassification_On_SSTDataset.ipynb)

#### Classification API Structure:

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session7-SecondHands-on/Assignment1/assets/code_folder_structure.PNG)

[Link to Classification API](https://github.com/garima-mahato/END2/tree/main/Session7-SecondHands-on/Assignment1/nlp_classification_api)

### 1) Dataset

**Text**

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session5-FirstHands-on/assets/sst_sent_tree.PNG)

**Label**

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session5-FirstHands-on/assets/sentiments.jpg)

Sentiment:

| S. No. | Label |
|---|---|
| 1 | Very Negative |
| 2 | Negative |
| 3 | Neutral |
| 4 | Positive |
| 5 | Very Positive |

#### Original Data

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session7-SecondHands-on/Assignment1/assets/data.PNG)

## 2) EDA

### EDA - Original Dataset

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session5-FirstHands-on/assets/eda_sent_dist.PNG)

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session5-FirstHands-on/assets/eda_avg_sent_len_comp.png)

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session5-FirstHands-on/assets/eda_word_freq_comp.png)

**Word Cloud for each of the 5 sentiments in Training Data**

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session5-FirstHands-on/assets/eda_aug_word_ccloud_train.png)

**Word Cloud for each of the 5 sentiments in Test Data**

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session5-FirstHands-on/assets/eda_aug_word_ccloud_test.png)

From the above clouds, we can see that the most common appearing words like **film** and **movie** appear in all sentiments and so can be considered stopword for the dataset.


### Data Extraction Code

```
class NLPClassificationDataset():
    def __init__(self, data_path, seed, batch_size, device, split_ratio=[0.7, 0.3]):
        self.split_ratio = split_ratio
        self.data_path = data_path
        self.seed = seed
        self.device = device
        self.batch_size = batch_size
        self.seq_data = self.load_data(self.data_path)

    def load_data(self, data_path):
        raise NotImplementedError
    
    def get_data(self):
        return self.seq_data
    
    def create_dataset(self, vectors = "glove.6B.100d", unk_init = torch.Tensor.normal_):
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
            if vectors is None and unk_init is None:
                SRC.build_vocab(self.train_data)
            else:
                SRC.build_vocab(self.train_data,  
                 vectors = vectors, 
                 unk_init = unk_init)

            TRG.build_vocab(self.train_data)

            print(f"Unique tokens in source vocabulary: {len(SRC.vocab)}")
            print(f"Unique tokens in target vocabulary: {len(TRG.vocab)}")

            return self.train_data, self.test_data, SRC, TRG, self.seq_data
        except Exception as e:
            raise e
    
    def iterate_dataset(self):
        try:
            self.train_data, self.test_data, SRC, TRG, data = self.create_dataset()
            self.train_iterator, self.test_iterator = BucketIterator.splits((self.train_data, self.test_data),batch_size=self.batch_size,sort_key = lambda x: len(x.src),sort_within_batch=True,device = self.device)
        
            return self.train_iterator, self.test_iterator, SRC, TRG, data
        except Exception as e:
            raise e

## SST Dataset
from .NLPClassificationDataset import NLPClassificationDataset

class SSTDataset(NLPClassificationDataset):
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
```


## 3) Model Building

**Model Code:**

```
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class NLPBasicClassifier(nn.Module):
    
    # Define all the layers used in model
    def __init__(self, vocab_size, embedding_dim, hidden_dim1, hidden_dim2, output_dim, n_layers, bidirectional, dropout, pad_index):
        # Constructor
        super().__init__()

        # embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_index)

        # lstm layer
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim1,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            batch_first=True)
        self.fc1 = nn.Linear(hidden_dim1 * 2, hidden_dim2)
        self.fc2 = nn.Linear(hidden_dim2, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # activation function
        self.act = nn.Softmax() #\ F.log_softmax(outp)

    def forward(self, text, text_lengths):
        # text = [batch size,sent_length]
        embedded = self.embedding(text)
        # embedded = [batch size, sent_len, emb dim]

        # packed sequence
        packed_embedded = pack_padded_sequence(embedded, text_lengths.to('cpu'), batch_first=True) # unpad

        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        # packed_output shape = (batch, seq_len, num_directions * hidden_size)
        # hidden shape  = (num_layers * num_directions, batch, hidden_size)

        # concat the final forward and backward hidden state
        cat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        # output, output_lengths = pad_packed_sequence(packed_output)  # pad the sequence to the max length in the batch

        rel = self.relu(cat)
        dense1 = self.fc1(rel)

        drop = self.dropout(dense1)
        preds = self.fc2(drop)

        return preds
```
**Model Structure:-**

```
NLPBasicClassifier(
  (embedding): Embedding(16388, 100, padding_idx=1)
  (lstm): LSTM(100, 256, num_layers=2, batch_first=True, bidirectional=True)
  (fc1): Linear(in_features=512, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=5, bias=True)
  (relu): ReLU()
  (dropout): Dropout(p=0.2, inplace=False)
  (act): Softmax(dim=None)
)
The model has 4,015,253 trainable parameters
```

### 4) Training and Testing

**Training Logs:**

```
Epoch: 01 | Epoch Time: 0m 1s
	Train Loss: 1.533 | Train Acc: 29.72%
	 Val. Loss: 1.472 |  Val. Acc: 34.75% 

 99%|█████████▉| 397360/400000 [00:29<00:00, 27112.99it/s]Epoch: 02 | Epoch Time: 0m 1s
	Train Loss: 1.377 | Train Acc: 38.91%
	 Val. Loss: 1.344 |  Val. Acc: 41.22% 

Epoch: 03 | Epoch Time: 0m 1s
	Train Loss: 1.237 | Train Acc: 44.73%
	 Val. Loss: 1.327 |  Val. Acc: 42.01% 

Epoch: 04 | Epoch Time: 0m 1s
	Train Loss: 1.142 | Train Acc: 48.97%
	 Val. Loss: 1.379 |  Val. Acc: 40.51% 

Epoch: 05 | Epoch Time: 0m 1s
	Train Loss: 0.997 | Train Acc: 56.26%
	 Val. Loss: 1.435 |  Val. Acc: 43.44% 

Epoch: 06 | Epoch Time: 0m 1s
	Train Loss: 0.823 | Train Acc: 65.49%
	 Val. Loss: 1.598 |  Val. Acc: 40.62% 

Epoch: 07 | Epoch Time: 0m 1s
	Train Loss: 0.682 | Train Acc: 72.18%
	 Val. Loss: 1.883 |  Val. Acc: 39.53% 

Epoch: 08 | Epoch Time: 0m 1s
	Train Loss: 0.508 | Train Acc: 80.66%
	 Val. Loss: 2.120 |  Val. Acc: 40.53% 

Epoch: 09 | Epoch Time: 0m 1s
	Train Loss: 0.369 | Train Acc: 86.48%
	 Val. Loss: 2.592 |  Val. Acc: 37.79% 

Epoch: 10 | Epoch Time: 0m 1s
	Train Loss: 0.268 | Train Acc: 90.00%
	 Val. Loss: 3.311 |  Val. Acc: 38.86% 

```

#### Training aand Testing Visualization

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session7-SecondHands-on/Assignment1/assets/train_test_acc_loss_graph.PNG)

#### Train vs Test Accuracy

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session7-SecondHands-on/Assignment1/assets/train_test_acc_comp.PNG)

#### Train vs Test Loss

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session7-SecondHands-on/Assignment1/assets/train_test_loss_comp.PNG)

### 5) Prediction

#### 10 Correctly Classified Texts From Test Data

```
****************************************
***** Correctly Classified Text: *******
****************************************
1) Text: A perfectly pleasant if slightly  comedy .
   Target Sentiment: positive
   Predicted Sentiment: positive

2) Text: Unfolds as one of the most politically audacious films of recent decades from any country , but especially from France .
   Target Sentiment: positive
   Predicted Sentiment: positive

3) Text: There 's a sheer unbridled delight in the way the story  ... 
   Target Sentiment: positive
   Predicted Sentiment: positive

4) Text: The Four Feathers is definitely   , but if you go in knowing that , you might have fun in this cinematic  . 
   Target Sentiment: positive
   Predicted Sentiment: positive

5) Text: ... the story , like  's Bolero , builds to a  that encompasses many more paths than we started with .
   Target Sentiment: positive
   Predicted Sentiment: positive

6) Text: A comedy that is warm , inviting , and surprising . 
   Target Sentiment: very positive
   Predicted Sentiment: very positive

7) Text:  and largely devoid of the depth or  that would make watching such a graphic treatment of the crimes  .
   Target Sentiment: negative
   Predicted Sentiment: negative

8) Text: can be as tiresome as 9 seconds of Jesse  '  Castro  , which are 
   Target Sentiment: negative
   Predicted Sentiment: negative

9) Text:  , I 'd rather watch them on the Animal Planet . 
   Target Sentiment: very negative
   Predicted Sentiment: very negative

10) Text: Good -   sequel . 
   Target Sentiment: positive
   Predicted Sentiment: positive

```


#### 10 Incorrectly Classified Texts From Test Data

```
****************************************
***** Incorrectly Classified Text: *******
****************************************
1) Text: The movie is n't just hilarious : It 's witty and inventive , too , and in hindsight , it is n't even all that dumb .
   Target Sentiment: very positive
   Predicted Sentiment: positive

2) Text: Stands as a document of what it felt like to be a New Yorker -- or , really , to be a human being -- in the weeks after 9\/11 .  
   Target Sentiment: positive
   Predicted Sentiment: neutral

3) Text: It works its magic with such exuberance and passion that the film 's length becomes a part of its fun .
   Target Sentiment: very positive
   Predicted Sentiment: positive

4) Text: It does n't do the original any particular  , but neither does it exude any charm or personality .
   Target Sentiment: negative
   Predicted Sentiment: neutral

5) Text: Do n't expect any surprises in this checklist of  cliches ... 
   Target Sentiment: very negative
   Predicted Sentiment: negative

6) Text: The film 's tone and pacing are off almost from the get - go .
   Target Sentiment: negative
   Predicted Sentiment: neutral

7) Text: It 's deep -  by a  to  every bodily  gag in There 's Something About Mary and  a parallel clone - gag . 
   Target Sentiment: neutral
   Predicted Sentiment: negative

8) Text: The lower your expectations , the more you 'll enjoy it . 
   Target Sentiment: negative
   Predicted Sentiment: positive

9) Text: A film without surprise geared toward maximum comfort and familiarity . 
   Target Sentiment: negative
   Predicted Sentiment: positive

10) Text: A chiller resolutely without chills . 
   Target Sentiment: very negative
   Predicted Sentiment: neutral
```


### 6) Evaluation

#### Confusion Matrix on testing data

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session7-SecondHands-on/Assignment1/assets/conf_matrix.PNG)

#### Evaluation Metrics on Test Data

**F1 Macro Score: 0.38181634331169134**

**Accuracy: 38.98405197873597 %**

---
---

## Assignment 2

The datasets are subclasses of:

```
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

class SeqDataset():
    def __init__(self, data_path, seed, batch_size, device, split_ratio=[0.7, 0.3]):
        self.split_ratio = split_ratio
        self.data_path = data_path
        self.seed = seed
        self.device = device
        self.batch_size = batch_size
        self.seq_data = self.load_data(self.data_path)
        
    def load_data(self, data_path):
        raise NotImplementedError
    
    def get_data(self):
        return self.seq_data
    
    def create_dataset(self):
        try:
            SRC = Field(tokenize = 'spacy',init_token = '<sos>',eos_token = '<eos>',lower = True)
            TRG = Field(tokenize = 'spacy',init_token = '<sos>',eos_token = '<eos>',lower = True)
            fields = [('src', SRC),('trg', TRG)]
            example = [data.Example.fromlist([self.seq_data.src[i],self.seq_data.trg[i]], fields) for i in range(self.seq_data.shape[0])] 
            
            # Creating dataset
            Dataset = data.Dataset(example, fields)
            (self.train_data, self.test_data) = Dataset.split(split_ratio=self.split_ratio, random_state=random.seed(self.seed))
            
            print(f"Number of training examples: {len(self.train_data.examples)}")
            print(f"Number of testing examples: {len(self.test_data.examples)}")

            # build vocabulary
            SRC.build_vocab(self.train_data, min_freq = 2)
            TRG.build_vocab(self.train_data, min_freq = 2)

            print(f"Unique tokens in source vocabulary: {len(SRC.vocab)}")
            print(f"Unique tokens in target vocabulary: {len(TRG.vocab)}")

            return self.train_data, self.test_data, SRC, TRG
        except Exception as e:
            raise e
    
    def iterate_dataset(self):
        try:
            self.train_data, self.test_data, SRC, TRG = self.create_dataset()
            self.train_iterator, self.test_iterator = BucketIterator.splits((self.train_data, self.test_data),batch_size=self.batch_size,sort_key = lambda x: len(x.src),sort_within_batch=True,device = self.device)
        
            return self.train_iterator, self.test_iterator, SRC, TRG, self.seq_data
        except Exception as e:
            raise e
```

### Wikipedia QA Data

Task: Anwer Questions

[Link to data](http://www.cs.cmu.edu/~ark/QA-data/)

#### Data Exploration

The data consist of: ['S09', 'S08', 'S10', 'LICENSE-S08,S09', 'README.v1.2']

There are 3 folders: S08, S09 and S10. The other 2 are files.

Within each of S08, S09, S10, there are:

> 1) data folder containing wikipedia articles

> 2) question answer pair text file

We need to consider this question answer text file.

```
gdrive/MyDrive/TSAI_END2/Session7/Assignment2/data/QuestionAnswerData/Question_Answer_Dataset_v1.2/S09
['data', 'question_answer_pairs.txt']


gdrive/MyDrive/TSAI_END2/Session7/Assignment2/data/QuestionAnswerData/Question_Answer_Dataset_v1.2/S08
['data', 'question_answer_pairs.txt']


gdrive/MyDrive/TSAI_END2/Session7/Assignment2/data/QuestionAnswerData/Question_Answer_Dataset_v1.2/S10
['data', 'question_answer_pairs.txt']


gdrive/MyDrive/TSAI_END2/Session7/Assignment2/data/QuestionAnswerData/Question_Answer_Dataset_v1.2/LICENSE-S08,S09


gdrive/MyDrive/TSAI_END2/Session7/Assignment2/data/QuestionAnswerData/Question_Answer_Dataset_v1.2/README.v1.2

```

All 3 'question_answer_pairs.txt' were read and combined. The result is below.

```
	ArticleTitle	Question	Answer	DifficultyFromQuestioner	DifficultyFromAnswerer	ArticleFile
0	Alessandro_Volta	Was Volta an Italian physicist?	yes	easy	easy	data/set4/a10
1	Alessandro_Volta	Was Volta an Italian physicist?	yes	easy	easy	data/set4/a10
2	Alessandro_Volta	Is Volta buried in the city of Pittsburgh?	no	easy	easy	data/set4/a10
3	Alessandro_Volta	Is Volta buried in the city of Pittsburgh?	no	easy	easy	data/set4/a10
4	Alessandro_Volta	Did Volta have a passion for the study of elec...	yes	easy	medium	data/set4/a10
...	...	...	...	...	...	...
3993	Zebra	What areas do the Grevy's Zebras inhabit?	NaN	hard	NaN	data/set1/a9
3994	Zebra	Which species of zebra is known as the common ...	Plains Zebra (Equus quagga, formerly Equus bur...	hard	medium	data/set1/a9
3995	Zebra	Which species of zebra is known as the common ...	Plains Zebra	hard	medium	data/set1/a9
3996	Zebra	At what age can a zebra breed?	five or six	hard	medium	data/set1/a9
3997	Zebra	At what age can a zebra breed?	5 or 6	hard	hard	data/set1/a9
3998 rows × 6 columns
```

<!--[Link to Github Code](https://github.com/garima-mahato/END2/blob/main/Session7-SecondHands-on/Assignment2/END2_Session7_Assignment2_QuestionAnswerDataset_v1_2.ipynb)-->

[Link to Colab Code](https://colab.research.google.com/drive/1DQdgNhJ1OXvtJKOvwL44fDFyQi5kiHBg?usp=sharing)

Code:

```
import torch
from torchtext.legacy import data
from torchtext.legacy.data import Field, BucketIterator

import os
import pandas as pd

from .SeqDataset import SeqDataset

class WikiDataset(SeqDataset):
    def __init__(self, data_path, seed, batch_size, device, split_ratio=[0.7, 0.3]):
        # super(WikiDataset, self).__init__(data_path, seed, batch_size, device, split_ratio)
        self.split_ratio = split_ratio
        self.data_path = data_path
        self.seed = seed
        self.device = device
        self.batch_size = batch_size
        self.seq_data = self.load_data(self.data_path)

    def load_data(self, data_path):
        seq_data = pd.DataFrame()
        for subdir in os.listdir(data_path):
            subdirectory = os.path.join(data_path, subdir)
            if os.path.isdir(subdirectory):
                for txt_file in os.listdir(subdirectory):
                    if '.txt' in txt_file:
                        df = pd.read_csv(os.path.join(subdirectory, txt_file), sep='\t', encoding=' ISO-8859-1')
                        seq_data = pd.concat([seq_data, df]).reset_index(drop=True)

        seq_data = seq_data[['Question','Answer']]
        seq_data = seq_data.dropna().reset_index(drop=True)
        seq_data.columns = ['src','trg']

        return seq_data
    
```

### Quora Data

Task: Generate duplicate question given a question

<!--[Link to Github Code](https://github.com/garima-mahato/END2/blob/main/Session7-SecondHands-on/Assignment2/END2_Session7_Assignment2_QuoraDataset.ipynb)-->

[Link to Colab Code](https://colab.research.google.com/drive/1nJdOw5nMuQz09YP_OFqLdP8emO_nDJ7E?usp=sharing)

Code:

```
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
    
```

## Additional Datasets
---

### AmbigNQ Light Data

Task: Answer a question

<!--[Link to Github Code](https://github.com/garima-mahato/END2/blob/main/Session7-SecondHands-on/Assignment2/END2_Session7_Assignment2_AmbigNQLightDataset.ipynb)-->

[Link to Colab Code](https://colab.research.google.com/drive/1jEeQQOcejQNy_JRPs6ZbBzRToT4dBAkq?usp=sharing)

```
import torch
from torchtext.legacy import data
from torchtext.legacy.data import Field, BucketIterator

from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
import json
import pandas as pd

from .SeqDataset import SeqDataset

class AmbigNqQADataset(SeqDataset):
    def __init__(self, data_path, seed, batch_size, device, split_ratio=[0.7, 0.3]):
        # super(QuoraDataset, self).__init__(data_path, seed, batch_size, device, split_ratio)
        self.split_ratio = split_ratio
        self.data_path = data_path
        self.seed = seed
        self.device = device
        self.batch_size = batch_size
        self.seq_data = self.load_data(self.data_path)

    def load_data(self, data_path):
        # download data
        resp = urlopen(data_path)
        zipfile = ZipFile(BytesIO(resp.read()))
        files = zipfile.namelist()
        for fs in files:
            if 'train' in fs:
                with zipfile.open(fs) as json_file:
                    train_json_data = json.load(json_file)
            elif 'dev' in fs:
                with zipfile.open(fs) as json_file:
                    test_json_data = json.load(json_file)
        
        # extract data
        seq_data = pd.DataFrame()
        for json_data in [train_json_data, test_json_data]:
            df = pd.json_normalize(json_data, record_path='annotations', meta=['question'])

            # for single answer
            sa_df = df[df['type']=='singleAnswer'][['question','answer']]
            sa_df = sa_df.explode('answer').reset_index(drop=True)

            # for multiple answer
            mqa_df = pd.concat([pd.DataFrame(x) for x in df[df['type']=='multipleQAs'].reset_index(drop=True)['qaPairs']], keys=df.index).reset_index(level=1, drop=True).reset_index(drop=True)
            mqa_df = mqa_df.explode('answer').reset_index(drop=True)

            if len(seq_data)==0:
                seq_data = pd.concat([sa_df,mqa_df]).reset_index(drop=True)
            else:
                seq_data = pd.concat([seq_data,sa_df,mqa_df]).reset_index(drop=True)

        seq_data.reset_index(drop=True,inplace=True)
        seq_data.columns = ['src','trg']

        return seq_data
    
```

### Commonsense QA Data

Task: Answer a question

<!--[Link to Github Code](https://github.com/garima-mahato/END2/blob/main/Session7-SecondHands-on/Assignment2/END2_Session7_Assignment2_AmbigNQLightDataset.ipynb)-->

[Link to Colab Code](https://colab.research.google.com/drive/1dsE_vbuoCurWxc6i15gMfjkSRDjOETcd?usp=sharing)

```
import torch
from torchtext.legacy import data
from torchtext.legacy.data import Field, BucketIterator

from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
import json
import pandas as pd

from .SeqDataset import SeqDataset

class CommonsenseQADataset(SeqDataset):
    def __init__(self, data_path, seed, batch_size, device, split_ratio=[0.7, 0.3]):
        # super(QuoraDataset, self).__init__(data_path, seed, batch_size, device, split_ratio)
        self.split_ratio = split_ratio
        self.data_path = data_path
        self.seed = seed
        self.device = device
        self.batch_size = batch_size
        self.seq_data = self.load_data(self.data_path)

    def load_data(self, data_path):
        # download data
        if not isinstance(data_path, list):
            data_path = [data_path]
        
        # extract data
        seq_data = pd.DataFrame()
        for url in data_path:
            resp = urlopen(url).read().decode()
            data = pd.read_json(resp,lines=True)
            df = pd.json_normalize(data.to_dict(orient='records'))

            df['trg'] = df.apply(lambda r: [x for x in r['question.choices'] if x['label']==r['answerKey']][0]['text'], axis=1)
            df = df[['question.stem','trg']]

            seq_data = pd.concat([seq_data,df]).reset_index(drop=True)

        seq_data.reset_index(drop=True,inplace=True)
        seq_data.columns = ['src','trg']

        return seq_data
    
```
