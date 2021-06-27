
# Session 7 - Second Hands-On (Assignment 2)
---

## Assignment
---

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
