# Session 5 - First Hands On
---

1) Look at this code (Links to an external site.) above. It has additional details on "Back Translate", i.e. using Google translate to convert the sentences. It has "random_swap" function, as well as "random_delete". 

2) Use "Back Translate", "random_swap" and "random_delete" to augment the data you are training on

3) Download the StanfordSentimentAnalysis Dataset from this link  (Links to an external site.)(it might be troubling to download it, so force download on chrome). Use "datasetSentences.txt" and "sentiment_labels.txt" files from the zip you just downloaded as your dataset. This dataset contains just over 10,000 pieces of Stanford data from HTML files of Rotten Tomatoes. The sentiments are rated between 1 and 25, where one is the most negative and 25 is the most positive.

4) Train your model and achieve 60%+ validation/text accuracy. Upload your collab file on GitHub with readme that contains details about your assignment/word (minimum 250 words), training logs showing final validation accuracy, and outcomes for 10 example inputs from the test/validation data.

## 1) Dataset

**Text**

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session5-FirstHands-on/assets/sst_sent_tree.PNG)

**Label**

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session5-FirstHands-on/assets/sentiments.jpg)

Sentiment:

| Label | Meaning |
|---|---|
| 1 | Very Negative |
| 2 | Negative |
| 3 | Neutral |
| 4 | Positive |
| 5 | Very Positive |

### Original Data

```
Sample training data:
  label                                               text
0     4  The Rock is destined to be the 21st Century 's...
1     5  The gorgeously elaborate continuation of `` Th...
2     4  Singer/composer Bryan Adams contributes a slew...
3     3  You 'd think by now America would have had eno...
4     4               Yet the act is still charming here .
 Data Size: 8544


Sample test data:
  label                                               text
0     3                     Effective but too-tepid biopic
1     4  If you sometimes like to go to the movies to h...
2     5  Emerges as something rare , an issue movie tha...
3     3  The film provides some great insight into the ...
4     5  Offers that rare combination of entertainment ...
 Data Size: 2210


Sample evaluation data:
  label                                               text
0     4  It 's a lovely film with lovely performances b...
1     3  No one goes unindicted here , which is probabl...
2     4  And if you 're not nearly moved to tears by a ...
3     5                   A warm , funny , engaging film .
4     5  Uses sharp humor and insight into human nature...
 Data Size: 1101
```

### Data Augmentation

```
Sample training data:
   label                                               text
0      4  The Rock is destined to be the 21st Century 's...
1      5  The gorgeously elaborate continuation of `` Th...
2      4  Singer/composer Bryan Adams contributes a slew...
3      3  You 'd think by now America would have had eno...
4      4               Yet the act is still charming here .
 Data Size: 27085


Sample test data:
  label                                               text
0     4  It 's a lovely film with lovely performances b...
1     3  No one goes unindicted here , which is probabl...
2     4  And if you 're not nearly moved to tears by a ...
3     5                   A warm , funny , engaging film .
4     5  Uses sharp humor and insight into human nature...
 Data Size: 1101
```

**Code to augment data:**
```
import re
class NLPDataAugmentor():
  def __init__(self, data, label, text, ratio=0.5):
    self.data = data
    self.label = label
    self.text = text
    self.ratio = int(ratio*len(self.data))
  
  #cleaning up text
  import re
  def get_only_chars(self,line):

      clean_line = ""

      line = line.replace("’", "")
      line = line.replace("'", "")
      line = line.replace("-", " ") #replace hyphens with spaces
      line = line.replace("\t", " ")
      line = line.replace("\n", " ")
      line = line.lower()

      for char in line:
          if char in 'qwertyuiopasdfghjklzxcvbnm ':
              clean_line += char
          else:
              clean_line += ' '

      clean_line = re.sub(' +',' ',clean_line) #delete extra spaces
      if clean_line[0] == ' ':
          clean_line = clean_line[1:]
      return clean_line

  def remove_stopwords(self,sentence):
    tokenized = sentence #custom_tokenize(sentence) #data['text'].apply(custom_tokenize) # Tokenize tweets
    lower_tokens = [t.lower() for t in tokenized] #tokenized.apply(lambda x: [t.lower() for t in x]) # Convert tokens into lower case
    alpha_only = [t for t in lower_tokens if t.isalpha()] #lower_tokens.apply(lambda x: [t for t in x if t.isalpha()]) # Remove punctuations
    no_stops = [t for t in alpha_only if t not in stopwords.words('english')] #alpha_only.apply(lambda x: [t for t in x if t not in stopwords.words('english')]) # remove stop words

    return no_stops

  def get_synonyms(self,word):
      import nltk
      from nltk.corpus import wordnet
      synonyms = []
        
      for syn in wordnet.synsets(word):
          for l in syn.lemmas():
              synonyms.append(l.name())
              # if l.antonyms():
              #     antonyms.append(l.antonyms()[0].name())
      synonyms = list(set(synonyms))
      if len(synonyms) > 0:
        new_synonym = random.choice(synonyms)
      else:
        new_synonym = word

      return new_synonym

  def random_insertion(self, sentence, n=5): 
      from random import randrange
      words = self.remove_stopwords(sentence) 
      if len(words)<=0:
        words = sentence
      for _ in range(n):
          word = random.choice(words)
          new_synonym = self.get_synonyms(word)
          sentence.insert(randrange(len(sentence)+1), new_synonym)
      return sentence
  
  # random deletion
  def random_deletion(self, words, p=0.5): 
    if len(words) == 1: # return if single word
        return words
    remaining = list(filter(lambda x: random.uniform(0,1) > p,words)) 
    if len(remaining) == 0: # if not left, sample a random word
        return [random.choice(words)] 
    else:
        return remaining
  
  # random swap
  def random_swap(self, sentence, n=5): 
    length = range(len(sentence)) 
    for _ in range(n):
        idx1, idx2 = random.sample(length, 2)
        sentence[idx1], sentence[idx2] = sentence[idx2], sentence[idx1] 
    return sentence
  
  import random
  import google_trans_new
  from google_trans_new import google_translator  

  def back_translation(self, sentence):
      translator = google_translator()

      available_langs = list(google_trans_new.LANGUAGES.keys()) 
      trans_lang = random.choice(available_langs) 

      translations = translator.translate(sentence, lang_tgt=trans_lang) 

      translations_en_random = translator.translate(translations, lang_src=trans_lang, lang_tgt='en') 

      return translations_en_random
  
  def clean_up(self, sentence):
    sentence = self.get_only_chars(sentence)
    words = sentence.split(' ')
    words = [word for word in words if word is not '']

    return words

  def execute(self):
    s1 = self.data.sample(self.ratio,random_state=4).reset_index(drop=True)
    s1[self.text] = s1[self.text].apply(self.clean_up).apply(self.random_insertion).map(lambda x: ' '.join(x))
    print('random insertion done')

    s2 = self.data.sample(self.ratio,random_state=1).reset_index(drop=True)
    s2[self.text] = s2[self.text].apply(self.clean_up).apply(self.random_deletion).map(lambda x: ' '.join(x))
    print('random deletion done')

    s3 = self.data[self.data[self.text].apply(self.clean_up).apply(len)>=3].sample(self.ratio,random_state=6).reset_index(drop=True)
    s3[self.text] = s3[self.text].apply(self.clean_up).apply(self.random_swap).map(lambda x: ' '.join(x))
    print('random swap done')

    s4 = self.data.sample(n=200,random_state=3).reset_index(drop=True)
    s4[self.text] = s4[self.text].apply(self.back_translation)
    print('back translation done')

    new_data = pd.concat([self.data,s1,s2,s3,s4])
    new_data.reset_index(inplace=True, drop=True)

    return new_data
```


## 2) EDA

### EDA - Original Dataset

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session5-FirstHands-on/assets/eda_sent_dist.PNG)

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session5-FirstHands-on/assets/eda_avg_sent_len_comp.png)

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session5-FirstHands-on/assets/eda_word_freq_comp.png)

### EDA - Augmented Dataset

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session5-FirstHands-on/assets/eda_aug1.PNG)

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session5-FirstHands-on/assets/eda_aug2.PNG)

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session5-FirstHands-on/assets/eda_aug3.PNG)

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session5-FirstHands-on/assets/eda_aug4.PNG)

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session5-FirstHands-on/assets/eda_aug5.PNG)

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session5-FirstHands-on/assets/eda_aug6.PNG)

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session5-FirstHands-on/assets/eda_aug7.PNG)

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session5-FirstHands-on/assets/eda_aug8.PNG)

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

## 4) Training and Testing

**Training Logs:**

```
Epoch: 01 | Epoch Time: 0m 5s
	Train Loss: 1.375 | Train Acc: 38.88%
	 Val. Loss: 1.308 |  Val. Acc: 40.18% 

Epoch: 02 | Epoch Time: 0m 5s
	Train Loss: 1.094 | Train Acc: 52.54%
	 Val. Loss: 1.386 |  Val. Acc: 40.34% 

Epoch: 03 | Epoch Time: 0m 5s
	Train Loss: 0.819 | Train Acc: 66.72%
	 Val. Loss: 1.605 |  Val. Acc: 38.51% 

Epoch: 04 | Epoch Time: 0m 5s
	Train Loss: 0.568 | Train Acc: 78.20%
	 Val. Loss: 1.957 |  Val. Acc: 39.29% 

Epoch: 05 | Epoch Time: 0m 5s
	Train Loss: 0.368 | Train Acc: 86.26%
	 Val. Loss: 2.394 |  Val. Acc: 40.07% 

Epoch: 06 | Epoch Time: 0m 5s
	Train Loss: 0.245 | Train Acc: 90.98%
	 Val. Loss: 2.866 |  Val. Acc: 38.60% 

Epoch: 07 | Epoch Time: 0m 5s
	Train Loss: 0.166 | Train Acc: 93.80%
	 Val. Loss: 3.697 |  Val. Acc: 38.94% 

Epoch: 08 | Epoch Time: 0m 5s
	Train Loss: 0.117 | Train Acc: 95.75%
	 Val. Loss: 4.880 |  Val. Acc: 37.29% 

Epoch: 09 | Epoch Time: 0m 5s
	Train Loss: 0.093 | Train Acc: 96.80%
	 Val. Loss: 5.191 |  Val. Acc: 38.06% 

Epoch: 10 | Epoch Time: 0m 5s
	Train Loss: 0.062 | Train Acc: 97.81%
	 Val. Loss: 5.869 |  Val. Acc: 37.47% 
```

#### Training aand Testing Visualization

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session5-FirstHands-on/assets/train_test_graph.PNG)

#### Train vs Test Accuracy

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session5-FirstHands-on/assets/train_test_acc.PNG)

#### Train vs Test Loss

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session5-FirstHands-on/assets/train_test_loss.PNG)

## 5) Prediction

#### 10 Correctly Classified Texts

```
****************************************
***** Correctly Classified Text: *******
****************************************
1) Text: No one goes unindicted here , which is probably for the best .
   
   Target Sentiment: Neutral
   
   Predicted Sentiment: Neutral


2) Text: There 's ... tremendous energy from the cast , a sense of playfulness and excitement that seems appropriate .
   
   Target Sentiment: Positive
   
   Predicted Sentiment: Positive


3) Text: Here 's yet another studio horror franchise mucking up its storyline with glitches casual fans could correct in their sleep .
   
   Target Sentiment: Very Negative
   
   Predicted Sentiment: Very Negative


4) Text: While the stoically delivered hokum of Hart 's War is never fun , it 's still a worthy addition to the growing canon of post-Saving Private Ryan tributes to the greatest generation .
   
   Target Sentiment: Neutral
   
   Predicted Sentiment: Neutral


5) Text: Building slowly and subtly , the film , sporting a breezy spontaneity and realistically drawn characterizations , develops into a significant character study that is both moving and wise .
   
   Target Sentiment: Positive
   
   Predicted Sentiment: Positive


6) Text: Ultimately feels empty and unsatisfying , like swallowing a Communion wafer without the wine .
   
   Target Sentiment: Very Negative
   
   Predicted Sentiment: Very Negative


7) Text: Chilling , well-acted , and finely directed : David Jacobson 's Dahmer .
   
   Target Sentiment: Positive
   
   Predicted Sentiment: Positive


8) Text: Against all odds in heaven and hell , it creeped me out just fine .
   
   Target Sentiment: Positive
   
   Predicted Sentiment: Positive


9) Text: A compelling Spanish film about the withering effects of jealousy in the life of a young monarch whose sexual passion for her husband becomes an obsession .
   
   Target Sentiment: Positive
   
   Predicted Sentiment: Positive


10) Text: It 's fascinating to see how Bettany and McDowell play off each other .
   
   Target Sentiment: Positive
   
   Predicted Sentiment: Positive

```


#### 10 Incorrectly Classified Texts

```
****************************************
***** Incorrectly Classified Text: *****
****************************************
1) Text: It 's a lovely film with lovely performances by Buy and Accorsi .
   Target Sentiment: Positive
   Predicted Sentiment: Negative

2) Text: And if you 're not nearly moved to tears by a couple of scenes , you 've got ice water in your veins .
   Target Sentiment: Positive
   Predicted Sentiment: Negative

3) Text: A warm , funny , engaging film .
   Target Sentiment: Very Positive
   Predicted Sentiment: Positive

4) Text: Uses sharp humor and insight into human nature to examine class conflict , adolescent yearning , the roots of friendship and sexual identity .
   Target Sentiment: Very Positive
   Predicted Sentiment: Negative

5) Text: Half Submarine flick , Half Ghost Story , All in one criminally neglected film
   Target Sentiment: Neutral
   Predicted Sentiment: Very Positive

6) Text: Entertains by providing good , lively company .
   Target Sentiment: Positive
   Predicted Sentiment: Negative

7) Text: Dazzles with its fully-written characters , its determined stylishness ( which always relates to characters and story ) and Johnny Dankworth 's best soundtrack in years .
   Target Sentiment: Very Positive
   Predicted Sentiment: Negative

8) Text: Visually imaginative , thematically instructive and thoroughly delightful , it takes us on a roller-coaster ride from innocence to experience without even a hint of that typical kiddie-flick sentimentality .
   Target Sentiment: Very Positive
   Predicted Sentiment: Positive

9) Text: Nothing 's at stake , just a twisty double-cross you can smell a mile away -- still , the derivative Nine Queens is lots of fun .
   Target Sentiment: Positive
   Predicted Sentiment: Negative

10) Text: Unlike the speedy wham-bam effect of most Hollywood offerings , character development -- and more importantly , character empathy -- is at the heart of Italian for Beginners .
   Target Sentiment: Very Positive
   Predicted Sentiment: Negative
```


## 6) Evaluation

**Accuracy on Testing data: 37.47 %**


#### Confusion Matrix on testing data

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session5-FirstHands-on/assets/test_conf_matrix.PNG)

