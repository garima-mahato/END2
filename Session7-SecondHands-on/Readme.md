
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

### 1) Wikipedia QA Data

**Task: Answer Questions**

**Description:**

There are three directories, one for each year of students: S08, S09, and S10.

The file "question_answer_pairs.txt" contains the questions and answers. The first line of the file contains 
column names for the tab-separated data fields in the file. This first line follows:

ArticleTitle    Question        Answer  DifficultyFromQuestioner        DifficultyFromAnswerer  ArticleFile

Field 1 is the name of the Wikipedia article from which questions and answers initially came.

Field 2 is the question.

Field 3 is the answer.

Field 4 is the prescribed difficulty rating for the question as given to the question-writer. 

Field 5 is a difficulty rating assigned by the individual who evaluated and answered the question, 
which may differ from the difficulty in field 4.

Field 6 is the relative path to the prefix of the article files. html files (.htm) and cleaned 
text (.txt) files are provided.

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

After that, only Question and Answer columns were extracted to form the below data:

```

	Question						Answer
0	Was Volta an Italian physicist?				yes
1	Was Volta an Italian physicist?				yes
2	Is Volta buried in the city of Pittsburgh?		no
3	Is Volta buried in the city of Pittsburgh?		no
4	Did Volta have a passion for the study of elec...	yes
...	...	...
3993	What areas do the Grevy's Zebras inhabit?		NaN
3994	Which species of zebra is known as the common ...	Plains Zebra (Equus quagga, formerly Equus bur...
3995	Which species of zebra is known as the common ...	Plains Zebra
3996	At what age can a zebra breed?				five or six
3997	At what age can a zebra breed?				5 or 6
3998 rows × 2 columns
```

There are some null values also. So, we need to remove those rows.

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 3998 entries, 0 to 3997
Data columns (total 2 columns):
 #   Column    Non-Null Count  Dtype 
---  ------    --------------  ----- 
 0   Question  3961 non-null   object
 1   Answer    3422 non-null   object
dtypes: object(2)
memory usage: 62.6+ KB
```

After removing, the final data has 3422 rows:

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 3422 entries, 0 to 3421
Data columns (total 2 columns):
 #   Column    Non-Null Count  Dtype 
---  ------    --------------  ----- 
 0   Question  3422 non-null   object
 1   Answer    3422 non-null   object
dtypes: object(2)
memory usage: 53.6+ KB

```

[Link to Github Code](https://github.com/garima-mahato/END2/blob/main/Session7-SecondHands-on/Assignment2/END2_Session7_Assignment2_QuestionAnswerDataset_v1_2.ipynb)

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

### 2) Quora Data

**Task: Generate Duplicate Question based on a question**

**Description:**

Quora dataset consists of over 400,000 lines of potential question duplicate pairs. Each line contains IDs for each question in the pair, the full text for each question, and a binary value that indicates whether the line truly contains a duplicate pair.

Sample Data:

```
	id	qid1	qid2	question1						question2						is_duplicate
0	0	1	2	What is the step by step guide to invest in sh...	What is the step by step guide to invest in sh...	0
1	1	3	4	What is the story of Kohinoor (Koh-i-Noor) Dia...	What would happen if the Indian government sto...	0
2	2	5	6	How can I increase the speed of my internet co...	How can Internet speed be increased by hacking...	0
3	3	7	8	Why am I mentally very lonely? How can I solve...	Find the remainder when [math]23^{24}[/math] i...	0
4	4	9	10	Which one dissolve in water quikly sugar, salt...	Which fish would survive in salt water?			0
```

id - unique questions pair id

qid1 - question 1 id

qid2 - question 2 id

question1 - full text of question 1

question2 - full text of question 2

is_duplicate - 0 indicates not a duplicate pair while 1 indicates dulicate questions

We need to consider question1 and question2 columns for our purpose. Among all rows, only those rows have to be considered where is_duplicate=1.

```
	question1						question2
0	Astrology: I am a Capricorn Sun Cap moon and c...	I'm a triple Capricorn (Sun, Moon and ascendan...
1	How can I be a good geologist?				What should I do to be a great geologist?
2	How do I read and find my YouTube comments?		How can I see all my Youtube comments?
3	What can make Physics easy to learn?			How can you make physics easy to learn?
4	What was your first sexual experience like?		What was your first sexual experience?
...	...	...
149258	What are some outfit ideas to wear to a frat p...	What are some outfit ideas wear to a frat them...
149259	Why is Manaphy childish in Pokémon Ranger and ...	Why is Manaphy annoying in Pokemon ranger and ...
149260	How does a long distance relationship work?		How are long distance relationships maintained?
149261	What does Jainism say about homosexuality?		What does Jainism say about Gays and Homosexua...
149262	Do you believe there is life after death?		Is it true that there is life after death?
```

Since there are no nulls in above data, we'll be using this data


[Link to Github Code](https://github.com/garima-mahato/END2/blob/main/Session7-SecondHands-on/Assignment2/END2_Session7_Assignment2_QuoraDataset.ipynb)

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

### 1) AmbigNQ Light Data

# Sequence to Sequence Prediction on AmbigQA Light Data
---

**Task: Answer Question**

**Description:**

**AmbigNQ** is a dataset covering 14,042 questions from NQ-open, an existing open-domain QA benchmark. We find that over half of the questions in NQ-open are ambiguous.

We provide two distributions of our new dataset AmbigNQ: a `full` version with all annotation metadata
and a `light` version with only inputs and outputs.

Here, Iam usig light version.

The light version contains
- train_light.json (3.3M)
- dev_light.json (977K)

`{train|dev}_light.json` files contains a list of dictionary that represents a single datapoint, with the following keys:

- `id` (string): an identifier for the question, consistent with the original NQ dataset.

- `question` (string): a question. This is identical to the question in the original NQ except we postprocess the string to start uppercase and end with a question mark.

- `annotations` (a list of dictionaries): a list of all acceptable outputs, where each output is a dictionary that represents either a single answer or multiple question-answer pairs.

    - `type`: `singleAnswer` or `multipleQAs`

    - (If `type` is `singleAnswer`) `answer`: a list of strings that are all acceptable answer texts

    - (If `type` is `multipleQAs`) `qaPairs`: a list of dictionaries with `question` and `answer`. `question` is a string, and `answer` is a list of strings that are all acceptable answer texts

Sample Data:

```
	annotations						id			question
0	[{'type': 'multipleQAs', 'qaPairs': [{'questio...	-4469503464110108318	When did the simpsons first air on television?
1	[{'type': 'singleAnswer', 'answer': ['David Mo...	4790842463458965203	Who played george washington in the john adams...
2	[{'type': 'multipleQAs', 'qaPairs': [{'questio...	-6631915997977101143	What is the legal age of marriage in usa?
3	[{'type': 'multipleQAs', 'qaPairs': [{'questio...	-3098213414945179817	Who starred in barefoot in the park on broadway?
4	[{'type': 'multipleQAs', 'qaPairs': [{'questio...	-927805218867163489	When did the manhattan project began and end?
...	...	...	...
10031	[{'type': 'multipleQAs', 'qaPairs': [{'questio...	8643122201694054820	When do the summer holidays start for schools?
10032	[{'type': 'multipleQAs', 'qaPairs': [{'questio...	9033094464364994905	Who is the band in the movie 10 things i hate ...
10033	[{'type': 'singleAnswer', 'answer': ['Gwynne E...	9101518012234561119	Who was the last person in the uk to be executed?
10034	[{'type': 'multipleQAs', 'qaPairs': [{'questio...	926954766593964346	Who does wonder woman end up with in the comics?
10035	[{'type': 'multipleQAs', 'qaPairs': [{'questio...	98262964342640738	When were the first pair of jordans released?
10036 rows × 3 columns
```

After expanding annotations column:

```
	type		qaPairs							answer	question
0	multipleQAs	[{'question': 'When did the Simpsons first air...	NaN	When did the simpsons first air on television?
1	singleAnswer	NaN	[David Morse]	Who played george washington in the john adams...
2	multipleQAs	[{'question': 'What is the legal age of marria...	NaN	What is the legal age of marriage in usa?
3	multipleQAs	[{'question': 'Who starred in barefoot in the ...	NaN	Who starred in barefoot in the park on broadway?
4	multipleQAs	[{'question': 'Based on the initial thoughts o...	NaN	When did the manhattan project began and end?
...	...	...	...	...
10246	multipleQAs	[{'question': 'Who is the band at Club Skunk i...	NaN	Who is the band in the movie 10 things i hate ...
10247	multipleQAs	[{'question': 'Who is the band that performs a...	NaN	Who is the band in the movie 10 things i hate ...
10248	singleAnswer	NaN	[Gwynne Evans and Peter Allen]	Who was the last person in the uk to be executed?
10249	multipleQAs	[{'question': 'Who does wonder woman end up wi...	NaN	Who does wonder woman end up with in the comics?
10250	multipleQAs	[{'question': 'When were the first pair of Air...	NaN	When were the first pair of jordans released?
10251 rows × 4 columns
```

Procedure to create question-answer pair:

1) The data was divided into 2 portions:

> 1) type = 'singleAnswer' : for this portion question column was taken and answer was taken from annotations column

```
	question						answer
0	Who played george washington in the john adams...	David Morse
1	When did the frozen ride open at epcot?			June 21, 2016
2	Name the landforms that form the boundaries of...	Aravali Range, Satpura Range, Vindhyan Range
3	When was the first airplane used in war?	Blériot XI
4	When was the first airplane used in war?	Nieuport IV
...	...	...
8346	Who played alotta fagina in austin powers movie?	Fabiana Udenio
8347	Who played alotta fagina in austin powers movie?	Fabiana Udenio
8348	Who played alotta fagina in austin powers movie?	Fabiana Udenio
8349	Who wrote make you feel my love song?	Bob Dylan
8350	Who was the last person in the uk to be executed?	Gwynne Evans and Peter Allen
8351 rows × 2 columns
```


> 2) type = 'multipleQAs' : for this portion, qaPairs is converted into question and answer columns expanding over rows.

```
	question						answer
0	When did the Simpsons first air on television ...	April 19, 1987
1	When did the Simpsons first air as a half-hour...	December 17, 1989
2	What is the legal age of marriage, without par...	18 years of age
3	What is the legal age of marriage, without par...	18
4	What is the legal age of marriage, without par...	19
...	...	...
19466	Who does wonder woman end up with in All Star ...	General Steven Rockwell Trevor
19467	Who does wonder woman end up with in All Star ...	Steve Trevor
19468	Who does wonder woman end up with in the new 52?	Superman and Batman
19469	When were the first pair of Air Jordans releas...	early 1984
19470	When were the first pair of Air Jordans releas...	November 17, 1984
19471 rows × 2 columns
```

2) Then, both these portion are combined vertically.

3) Steps 1-2 are repeated for both train and test data.

[Link to Github Code](https://github.com/garima-mahato/END2/blob/main/Session7-SecondHands-on/Assignment2/END2_Session7_Assignment2_AmbigNQLightDataset.ipynb)

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

### 2) Commonsense QA Data

**Task: Answer Common sense question**

**Description:**

CommonsenseQA is a new multiple-choice question answering dataset that requires different types of commonsense knowledge to predict the correct answers . It contains 12,102 questions with one correct answer and four distractor answers.  The dataset is provided in two major training/validation/testing set splits.

There are 3 JSON files for: train, validate, test.

We will consider train and validate files because test does not contain answers.

Each line in JSON file represents record. Each record consists of: 

1) answerKey:- It denotes the key  or label for correct option.

2) id:- uniques question id

3) question:- It is a dictionary of:

> i) question_concept - denotes the category to which question belong.

> ii) choices - denotes choices among which answer lies. It is a list of dictionary containing:

>> a) label: can be A or B or C or D

>> b) text

> iii) stem

Below is the structure when converted to Data Frame:

```
    answerKey	id			      question.question_concept	question.choices				question.stem
0	A	075e483d21c29a511267ef62bedc0461	punishing	[{'label': 'A', 'text': 'ignore'}, {'label': '...	The sanctions against the school were a punish...
1	B	61fe6e879ff18686d7552425a36344c8	people		[{'label': 'A', 'text': 'race track'}, {'label...	Sammy wanted to go to where the people were. ...
2	A	4c1cb0e95b99f72d55c068ba0255c54d	choker		[{'label': 'A', 'text': 'jewelry store'}, {'la...	To locate a choker not located in a jewelry bo...
3	D	02e821a3e53cb320790950aab4489e85	highway		[{'label': 'A', 'text': 'united states'}, {'la...	Google Maps and other highway and street GPS s...
4	C	23505889b94e880c3e89cff4ba119860	fox		[{'label': 'A', 'text': 'pretty flowers.'}, {'...	The fox walked from the city into the forest, ...
...	...	...	...	...	...
9736	E	f1b2a30a1facff543e055231c5f90dd0	going public	[{'label': 'A', 'text': 'consequences'}, {'lab...	What would someone need to do if he or she wan...
9737	D	a63b4d0c0b34d6e5f5ce7b2c2c08b825	chair		[{'label': 'A', 'text': 'stadium'}, {'label': ...	Where might you find a chair at an office?
9738	A	22d0eea15e10be56024fd00bb0e4f72f	jeans		[{'label': 'A', 'text': 'shopping mall'}, {'la...	Where would you buy jeans in a place with a la...
9739	A	7c55160a4630de9690eb328b57a18dc2	well		[{'label': 'A', 'text': 'fairytale'}, {'label'...	John fell down the well. he couldn't believe ...
9740	C	dd640927f9920930501fb8dc3efc196b	electricity	[{'label': 'A', 'text': 'put in to the water'}...	I forgot to pay the electricity bill, now what...
9741 rows × 5 columns
```

To create answer column, answerKey is matched with question.choices.

```
   answerKey	id				question.question_concept	question.choices				question.stem	answer
0	A	075e483d21c29a511267ef62bedc0461	punishing	[{'label': 'A', 'text': 'ignore'}, {'label': '...	The sanctions against the school were a punish...	ignore
1	B	61fe6e879ff18686d7552425a36344c8	people		[{'label': 'A', 'text': 'race track'}, {'label...	Sammy wanted to go to where the people were. ...	populated areas
2	A	4c1cb0e95b99f72d55c068ba0255c54d	choker		[{'label': 'A', 'text': 'jewelry store'}, {'la...	To locate a choker not located in a jewelry bo...	jewelry store
3	D	02e821a3e53cb320790950aab4489e85	highway		[{'label': 'A', 'text': 'united states'}, {'la...	Google Maps and other highway and street GPS s...	atlas
4	C	23505889b94e880c3e89cff4ba119860	fox		[{'label': 'A', 'text': 'pretty flowers.'}, {'...	The fox walked from the city into the forest, ...	natural habitat
...	...	...	...	...	...	...
9736	E	f1b2a30a1facff543e055231c5f90dd0	going public	[{'label': 'A', 'text': 'consequences'}, {'lab...	What would someone need to do if he or she wan...	telling all
9737	D	a63b4d0c0b34d6e5f5ce7b2c2c08b825	chair		[{'label': 'A', 'text': 'stadium'}, {'label': ...	Where might you find a chair at an office?	cubicle
9738	A	22d0eea15e10be56024fd00bb0e4f72f	jeans		[{'label': 'A', 'text': 'shopping mall'}, {'la...	Where would you buy jeans in a place with a la...	shopping mall
9739	A	7c55160a4630de9690eb328b57a18dc2	well		[{'label': 'A', 'text': 'fairytale'}, {'label'...	John fell down the well. he couldn't believe ...	fairytale
9740	C	dd640927f9920930501fb8dc3efc196b	electricity	[{'label': 'A', 'text': 'put in to the water'}...	I forgot to pay the electricity bill, now what...	produce heat
9741 rows × 6 columns
```

From above question.stem is selected as question column and answer column are selected.

[Link to Github Code](https://github.com/garima-mahato/END2/blob/main/Session7-SecondHands-on/Assignment2/END2_Session7_Assignment2_CommonsenseQA.ipynb)

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
