# Assignment
---

Pick any of your past code and:

1) Implement the following metrics (either on separate models or same, your choice):

> 1) Recall, Precision, and F1 Score

> 2) BLEU 

> 3) Perplexity (explain whether you are using bigram, trigram, or something else, what does your PPL score represent?)

> 4) BERTScore (here are [1](https://colab.research.google.com/drive/1kpL8Y_AnUUiCxFjhxSrxCsc6-sDMNb_Q) [2](https://huggingface.co/metrics/bertscore) )

2) Once done, proceed to answer questions in the Assignment-Submission Page. 

Questions asked are:

> 1) Share the link to the readme file where you have explained all 4 metrics. 

> 2) Share the link(s) where we can find the code and training logs for all of your 4 metrics

> 3) Share the last 2-3 epochs/stage logs for all of your 4 metrics separately (A, B, C, D) and describe your understanding about the numbers you're seeing, are they good/bad? Why?

# Solution
---

## Evaluation Metrics
---

### 1) Recall, Precision, and F1 Score

[GitHub Notebook Link](https://github.com/garima-mahato/END2/blob/main/Session9-LearningRatesandEvaluationMetricsPart1/END2_Session9_Assignment_SentimentClassification_On_SSTDataset_Precision%2CRecall%2CF1Metric.ipynb)
[Google Colab Link](https://githubtocolab.com/garima-mahato/END2/blob/main/Session9-LearningRatesandEvaluationMetricsPart1/END2_Session9_Assignment_SentimentClassification_On_SSTDataset_Precision%2CRecall%2CF1Metric.ipynb)

**Code**

```
import numpy as np

class precision_recall_f1score:
    def __init__(self, labels=None):
        # if labels is not None:
        #     assert isinstance(labels, list), 'labels must be list of integers starting with 0'
        self.labels = labels
        self.classwise_correct_pred = {c: 0 for c in self.labels}
        self.classwise_pred = {c: 0 for c in self.labels}
        self.classwise_target = {c: 0 for c in self.labels}
        self.classwise_prec_recall_f1 = {c: {'precision': 0, 'recall': 0, 'f1': 0} for c in self.labels}
        self.global_tp = 0
        self.global_fp = 0
        self.global_fn = 0
        self.micro_avg_f1, self.micro_prec, self.micro_recall, self.macro_f1, self.macro_precision, self.macro_recall, self.wgt_f1, self.wgt_precision, self.wgt_recall = 0,0,0,0,0,0,0,0,0

    def update(self, preds, targets):
        try:
            # if len(targets.shape)!=2:
            #     targets = targets.unsqueeze(1)
            top_pred = preds.argmax(1, keepdim = True)
            self.global_tp += top_pred.eq(targets.view_as(top_pred)).sum().item()
            for c in self.classwise_correct_pred.keys():
                self.classwise_correct_pred[c] += (top_pred[top_pred.eq(targets.view_as(top_pred))]==c).sum().float()
                self.classwise_pred[c] += (top_pred==c).sum().float()
                self.classwise_target[c] += (targets==c).sum().float()
        except Exception as e:
            raise e
    
    def calculate(self):
        try:
            for k in self.labels:
                self.classwise_prec_recall_f1[k]['precision'] = (self.classwise_correct_pred[k]/self.classwise_pred[k]).item()
                self.classwise_prec_recall_f1[k]['recall'] = (self.classwise_correct_pred[k]/self.classwise_target[k]).item()
                self.classwise_prec_recall_f1[k]['f1'] = (self.classwise_prec_recall_f1[k]['precision'] * self.classwise_prec_recall_f1[k]['recall'])/(self.classwise_prec_recall_f1[k]['precision'] + self.classwise_prec_recall_f1[k]['recall'] + 1e-20)

                self.global_fp += self.classwise_pred[k] - self.classwise_correct_pred[k]
                self.global_fn += self.classwise_target[k] - self.classwise_correct_pred[k]

            self.micro_prec = self.global_tp/(self.global_tp+self.global_fp.item())
            self.micro_recall = self.global_tp/(self.global_tp+self.global_fn.item())
            self.micro_avg_f1 = (self.micro_prec * self.micro_recall) / (self.micro_prec + self.micro_recall + 1e-20)

            self.macro_precision = np.average([self.classwise_prec_recall_f1[k]['precision'] for k in self.classwise_correct_pred.keys()])
            self.macro_recall = np.average([self.classwise_prec_recall_f1[k]['recall'] for k in self.classwise_correct_pred.keys()])
            self.macro_f1 = np.average([self.classwise_prec_recall_f1[k]['f1'] for k in self.classwise_correct_pred.keys()])

            weights = [v.item() for v in self.classwise_target.values()]
            self.wgt_precision = np.average([self.classwise_prec_recall_f1[k]['precision'] for k in self.classwise_correct_pred.keys()], weights=weights)
            self.wgt_recall = np.average([self.classwise_prec_recall_f1[k]['recall'] for k in self.classwise_correct_pred.keys()], weights=weights)
            self.wgt_f1 = np.average([self.classwise_prec_recall_f1[k]['f1'] for k in self.classwise_correct_pred.keys()], weights=weights)

            return self.classwise_prec_recall_f1, self.micro_avg_f1, self.micro_prec, self.micro_recall, self.macro_f1, self.macro_precision, self.macro_recall, self.wgt_f1, self.wgt_precision, self.wgt_recall
        except Exception as e:
            raise e
```

### 2) BLEU Score

[GitHub Notebook Link](https://github.com/garima-mahato/END2/blob/main/Session9-LearningRatesandEvaluationMetricsPart1/END2_Session9_BLEU_PPL_BertScore.ipynb)
[Google Colab Link](https://githubtocolab.com/garima-mahato/END2/blob/main/Session9-LearningRatesandEvaluationMetricsPart1/END2_Session9_BLEU_PPL_BertScore.ipynb)

**Code**

Base Code

```

```

Implementation

```
from torchtext.data.metrics import bleu_score

def calculate_bleu(model, max_n=4):
    
    trgs = []
    pred_trgs = []

    data = Multi30k(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    
    for datum in data:
        
        src, trg = datum
        
        pred_trg = translate_sentence(model, src)
        
        #cut off <eos> token
        pred_trg = pred_trg[:-1]
        
        pred_trgs.append(pred_trg.strip().split(' '))
        trgs.append([token_transform[TGT_LANGUAGE](trg.strip())])
        
    return bleu_score(pred_trgs, trgs, max_n=max_n)
```

### 3) Perplexity

[GitHub Notebook Link](https://github.com/garima-mahato/END2/blob/main/Session9-LearningRatesandEvaluationMetricsPart1/END2_Session9_BLEU_PPL_BertScore.ipynb)
[Google Colab Link](https://githubtocolab.com/garima-mahato/END2/blob/main/Session9-LearningRatesandEvaluationMetricsPart1/END2_Session9_BLEU_PPL_BertScore.ipynb)

Perplexity means agitated/entangled state. Agitation or Randomness of a system is measured by entropy of the system. 

**Code**

```
model.eval()
losses = 0

val_iter = Multi30k(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

for src, tgt in val_dataloader:
    src = src.to(DEVICE)
    tgt = tgt.to(DEVICE)

    tgt_input = tgt[:-1, :]

    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

    logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)
    
    tgt_out = tgt[1:, :]
    loss = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
    losses += loss.item()

val_loss = losses / len(val_dataloader)

perplexity = np.exp(val_loss)
```

### 4) BERT Score

[GitHub Notebook Link](https://github.com/garima-mahato/END2/blob/main/Session9-LearningRatesandEvaluationMetricsPart1/END2_Session9_BLEU_PPL_BertScore.ipynb)
[Google Colab Link](https://githubtocolab.com/garima-mahato/END2/blob/main/Session9-LearningRatesandEvaluationMetricsPart1/END2_Session9_BLEU_PPL_BertScore.ipynb)

**Code**

Base Code

```

```

Implementation

```
from bert_score import score
from transformers import logging

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

def calculate_bert_score(model):
    
    trgs = []
    pred_trgs = []

    data = Multi30k(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    
    for datum in data:
        
        src, trg = datum
        
        pred_trg = translate_sentence(model, src)
        
        #cut off <eos> token
        pred_trg = pred_trg[:-1]
        
        pred_trgs.append(pred_trg.strip())
        trgs.append([trg.strip()])

    P, R, F1 = score(pred_trgs, trgs, lang="en", verbose=False, batch_size=BATCH_SIZE)
    P, R, F1 = P.mean(), R.mean(), F1.mean()
        
    return P, R, F1
```