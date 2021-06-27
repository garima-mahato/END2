
# Session 7 - Second Hands-On (Assignment1)
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

