
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

[Link to GitHub Code]()

[Link to Colab Code]()

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

## 3) Model Building

**Model Code:**

```
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class classifier(nn.Module):
    
    # Define all the layers used in model
    def __init__(self, vocab_size, embedding_dim, hidden_dim1, hidden_dim2, output_dim, n_layers,
                 bidirectional, dropout, pad_index):
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

    def forward(self, text, text_lengths):
        embedded = self.embedding(text)

        # packed sequence
        packed_embedded = pack_padded_sequence(embedded, text_lengths.to('cpu'), batch_first=True) # unpad

        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        # concat the final forward and backward hidden state
        cat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

        rel = self.relu(cat)
        dense1 = self.fc1(rel)

        drop = self.dropout(dense1)
        preds = self.fc2(drop)
        
        return preds
```

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session7-SecondHands-on/Assignment1/assets/model.PNG)

### 4) Training and Testing

**Training Logs:**

```
Epoch: 01 | Epoch Time: 0m 3s
	Train Loss: 1.513 | Train Acc: 32.41%
	 Val. Loss: 1.450 |  Val. Acc: 37.15% 

Epoch: 02 | Epoch Time: 0m 3s
	Train Loss: 1.330 | Train Acc: 41.18%
	 Val. Loss: 1.354 |  Val. Acc: 40.87% 

Epoch: 03 | Epoch Time: 0m 3s
	Train Loss: 1.171 | Train Acc: 47.44%
	 Val. Loss: 1.342 |  Val. Acc: 41.35% 

Epoch: 04 | Epoch Time: 0m 3s
	Train Loss: 0.996 | Train Acc: 56.11%
	 Val. Loss: 1.425 |  Val. Acc: 40.87% 

Epoch: 05 | Epoch Time: 0m 3s
	Train Loss: 0.796 | Train Acc: 67.04%
	 Val. Loss: 1.660 |  Val. Acc: 37.96% 

Epoch: 06 | Epoch Time: 0m 3s
	Train Loss: 0.597 | Train Acc: 76.47%
	 Val. Loss: 2.035 |  Val. Acc: 39.62% 

Epoch: 07 | Epoch Time: 0m 3s
	Train Loss: 0.413 | Train Acc: 84.35%
	 Val. Loss: 2.526 |  Val. Acc: 38.83% 

Epoch: 08 | Epoch Time: 0m 3s
	Train Loss: 0.280 | Train Acc: 89.69%
	 Val. Loss: 2.906 |  Val. Acc: 38.91% 

Epoch: 09 | Epoch Time: 0m 3s
	Train Loss: 0.164 | Train Acc: 94.12%
	 Val. Loss: 3.431 |  Val. Acc: 38.06% 

Epoch: 10 | Epoch Time: 0m 3s
	Train Loss: 0.098 | Train Acc: 96.60%
	 Val. Loss: 4.340 |  Val. Acc: 39.41% 
```

#### Training aand Testing Visualization

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session7-SecondHands-on/Assignment1/assets/train_test_acc_loss_graph.PNG)

#### Train vs Test Accuracy

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session7-SecondHands-on/Assignment1/assets/train_test_acc.PNG)

#### Train vs Test Loss

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session7-SecondHands-on/Assignment1/assets/train_test_loss.PNG)

### 5) Prediction

#### 10 Correctly Classified Texts

```
****************************************
***** Correctly Classified Text: *******
****************************************
1) Text: Effective but too-tepid biopic
   Target Sentiment: neutral
   Predicted Sentiment: neutral

2) Text: The film provides some great insight into the neurotic mindset of all comics -- even those who have reached the absolute top of the game .
   Target Sentiment: neutral
   Predicted Sentiment: neutral

3) Text: Perhaps no picture ever made has more literally showed that the road to hell is paved with good intentions .
   Target Sentiment: positive
   Predicted Sentiment: positive

4) Text: Ultimately , it ponders the reasons we need stories so much .
   Target Sentiment: neutral
   Predicted Sentiment: neutral

5) Text: Illuminating if overly talky documentary .
   Target Sentiment: neutral
   Predicted Sentiment: neutral

6) Text: Light , cute and forgettable .
   Target Sentiment: neutral
   Predicted Sentiment: neutral

7) Text: Cantet perfectly captures the hotel lobbies , two-lane highways , and roadside cafes that permeate Vincent 's days
   Target Sentiment: positive
   Predicted Sentiment: positive

8) Text: A heavy reliance on CGI technology is beginning to creep into the series .
   Target Sentiment: neutral
   Predicted Sentiment: neutral

9) Text: Karmen moves like rhythm itself , her lips chanting to the beat , her long , braided hair doing little to wipe away the jeweled beads of sweat .
   Target Sentiment: neutral
   Predicted Sentiment: neutral

10) Text: Manages to be original , even though it rips off many of its ideas .
   Target Sentiment: neutral
   Predicted Sentiment: neutral
```


#### 10 Incorrectly Classified Texts

```
****************************************
***** Incorrectly Classified Text: *****
****************************************
1) Text: The Rock is destined to be the 21st Century 's new `` Conan '' and that he 's going to make a splash even greater than Arnold Schwarzenegger , Jean-Claud Van Damme or Steven Segal .
   Target Sentiment: positive
   Predicted Sentiment: negative

2) Text: The gorgeously elaborate continuation of `` The Lord of the Rings '' trilogy is so huge that a column of words can not adequately describe co-writer\/director Peter Jackson 's expanded vision of J.R.R. Tolkien 's Middle-earth .
   Target Sentiment: very positive
   Predicted Sentiment: positive

3) Text: If you sometimes like to go to the movies to have fun , Wasabi is a good place to start .
   Target Sentiment: positive
   Predicted Sentiment: negative

4) Text: Emerges as something rare , an issue movie that 's so honest and keenly observed that it does n't feel like one .
   Target Sentiment: very positive
   Predicted Sentiment: positive

5) Text: Offers that rare combination of entertainment and education .
   Target Sentiment: very positive
   Predicted Sentiment: positive

6) Text: Steers turns in a snappy screenplay that curls at the edges ; it 's so clever you want to hate it .
   Target Sentiment: positive
   Predicted Sentiment: negative

7) Text: But he somehow pulls it off .
   Target Sentiment: positive
   Predicted Sentiment: negative

8) Text: Take Care of My Cat offers a refreshingly different slice of Asian cinema .
   Target Sentiment: positive
   Predicted Sentiment: negative

9) Text: This is a film well worth seeing , talking and singing heads and all .
   Target Sentiment: very positive
   Predicted Sentiment: positive

10) Text: What really surprises about Wisegirls is its low-key quality and genuine tenderness .
   Target Sentiment: positive
   Predicted Sentiment: negative
```


### 6) Evaluation

#### Confusion Matrix on testing data

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session7-SecondHands-on/Assignment1/assets/conf_matrix.PNG)

#### Evaluation Metrics on Test Data

**F1 Macro Score: 0.37888155595873435**

**Accuracy: 39.42705256940343 %**

---
---

## Assignment 2

