# BERT and BART
---

## Assignment

1) TASK 1: Train BERT using the code mentioned here (Links to an external site.) on the Squad Dataset for 20% overall samples (1/5 Epochs). Show results on 5 samples. 

2) TASK 2: Reproductive these (Links to an external site.) results, and show output on 5 samples.

3) TASK 3: Reproduce the training explained in this blog (Links to an external site.). You can decide to pick fewer datasets. 

4) Proceed to Session 14 - Assignment Solutions page and:

> 1) Submit README link for Task 1 (training log snippets and 5 sample results along with BERT description must be available) - 750

> 2) Submit README link for Task 2 (training log snippets and 5 sample results) - 250

> 3) Submit README link for Task 3 (training log snippets and 5 sample results along with BART description must be available) - 1000

## Solution

### TASK 1

#### BERT

#### Training

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session14-BERTandBART/assets/BERT_training.png)

```
***** Running training *****
  Num examples = 28080
  Num Epochs = 1
  Batch size = 16
  Total optimization steps = 1755
Epoch:   0%|          | 0/1 [00:00<?, ?it/s]
Iteration:   0%|          | 0/1755 [00:00<?, ?it/s]
Train loss: 1.7061200550198554
```

#### Sample Predictions

```
Sample Answers predicted by BERT
****************************************
1) Question: In what country is Normandy located?
Predicted Answer: France
----------------------------------------
2) Question: When were the Normans in Normandy?
Predicted Answer: 10th and 11th centuries
----------------------------------------
3) Question: From which countries did the Norse originate?
Predicted Answer: Denmark, Iceland and Norway
----------------------------------------
4) Question: Who was the Norse leader?
Predicted Answer: Rollo
----------------------------------------
5) Question: What century did the Normans first gain their separate identity?
Predicted Answer: 10th century
----------------------------------------
6) Question: Who gave their name to Normandy in the 1000's and 1100's
Predicted Answer: The Normans
----------------------------------------
7) Question: What is France a region of?
Predicted Answer: Normandy
----------------------------------------
8) Question: Who did King Charles III swear fealty to?
Predicted Answer: West Francia
----------------------------------------
9) Question: When did the Frankish identity emerge?
Predicted Answer: first half of the 10th century
----------------------------------------
10) Question: Who was the duke in the battle of Hastings?
Predicted Answer: William the Conqueror
----------------------------------------
```