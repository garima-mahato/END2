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

|Terms|Meaning|
|---|---|
|Reference translation/sentence/word/token | Human translation/sentence/word/token |
|Candidate translation/sentence/word/token | Machine/Model translation/sentence/word/token |

### 1) Recall, Precision, and F1 Score

[GitHub Notebook Link](https://github.com/garima-mahato/END2/blob/main/Session9-LearningRatesandEvaluationMetricsPart1/END2_Session9_Assignment_SentimentClassification_On_SSTDataset_Precision%2CRecall%2CF1Metric.ipynb)

[Google Colab Link](https://githubtocolab.com/garima-mahato/END2/blob/main/Session9-LearningRatesandEvaluationMetricsPart1/END2_Session9_Assignment_SentimentClassification_On_SSTDataset_Precision%2CRecall%2CF1Metric.ipynb)

#### Description



#### Code

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

#### Training Logs of Precision, Recall, F1 Score with explanation

```
Epoch: 08 | Epoch Time: 0m 1s
	 Train Loss: 0.512 | Train Acc: 80.45%
	 Val. Loss: 2.123 
	 Val Metric: Precision, Recall, F1
	 ======================================
	 Class | Precision | Recall | F1
	 ======================================
	 positive | 0.3987538814544678 | 0.436860054731369 | 0.20846904884291376
	 negative | 0.43919509649276733 | 0.5516483783721924 | 0.24452021829408901
	 neutral | 0.2540540397167206 | 0.2326732724905014 | 0.1214470265542746
	 very positive | 0.49799197912216187 | 0.4542124569416046 | 0.23754789602675389
	 very negative | 0.40528634190559387 | 0.2067415714263916 | 0.13690476100518295
	 ======================================
	 Micro Average F1 Score: 0.2018606024808033
	 Macro Average F1 Score: 0.18977779014464283
	 Weighted Average F1 Score: 0.19786720630547747
Epoch: 09 | Epoch Time: 0m 1s
	 Train Loss: 0.367 | Train Acc: 86.41%
	 Val. Loss: 2.589 
	 Val Metric: Precision, Recall, F1
	 ======================================
	 Class | Precision | Recall | F1
	 ======================================
	 positive | 0.42064371705055237 | 0.4311717748641968 | 0.21292134245934963
	 negative | 0.43939393758773804 | 0.38241758942604065 | 0.20446533651249416
	 neutral | 0.2233918160200119 | 0.31518152356147766 | 0.13073237709661886
	 very positive | 0.49900200963020325 | 0.45787546038627625 | 0.23877746320975812
	 very negative | 0.36201781034469604 | 0.27415731549263 | 0.1560102352539963
	 ======================================
	 Micro Average F1 Score: 0.1904902539870053
	 Macro Average F1 Score: 0.1885813509064434
	 Weighted Average F1 Score: 0.19262911587987164
Epoch: 10 | Epoch Time: 0m 1s
	 Train Loss: 0.270 | Train Acc: 89.91%
	 Val. Loss: 3.245 
	 Val Metric: Precision, Recall, F1
	 ======================================
	 Class | Precision | Recall | F1
	 ======================================
	 positive | 0.42147117853164673 | 0.48236632347106934 | 0.22493368817608342
	 negative | 0.4451901614665985 | 0.4373626410961151 | 0.22062084471733523
	 neutral | 0.24074074625968933 | 0.3217821717262268 | 0.13771186502373733
	 very positive | 0.5513513684272766 | 0.37362638115882874 | 0.22270742904316226
	 very negative | 0.38562092185020447 | 0.26516854763031006 | 0.15712383893443077
	 ======================================
	 Micro Average F1 Score: 0.19772593030124036
	 Macro Average F1 Score: 0.1926195331789498
	 Weighted Average F1 Score: 0.1988935131090743
```

As we can see Classwise Precision and Recall and F1 is increasing along with Micro, Macro,Weighted Avg F1 score. This means that the model is improving as it is able to predict more accurately and recalling is improved.

----

### 2) BLEU Score

[GitHub Notebook Link](https://github.com/garima-mahato/END2/blob/main/Session9-LearningRatesandEvaluationMetricsPart1/END2_Session9_BLEU_PPL_BertScore.ipynb)

[Google Colab Link](https://githubtocolab.com/garima-mahato/END2/blob/main/Session9-LearningRatesandEvaluationMetricsPart1/END2_Session9_BLEU_PPL_BertScore.ipynb)

#### Description



#### Code

**Base Code**

[Reference](https://pytorch.org/text/_modules/torchtext/data/metrics.html)

```
def bleu_score(candidate_corpus, references_corpus, max_n=4, weights=[0.25] * 4):
    """Computes the BLEU score between a candidate translation corpus and a references
    translation corpus. Based on https://www.aclweb.org/anthology/P02-1040.pdf

    Arguments:
        candidate_corpus: an iterable of candidate translations. Each translation is an
            iterable of tokens
        references_corpus: an iterable of iterables of reference translations. Each
            translation is an iterable of tokens
        max_n: the maximum n-gram we want to use. E.g. if max_n=3, we will use unigrams,
            bigrams and trigrams
        weights: a list of weights used for each n-gram category (uniform by default)

    Examples:
        >>> from torchtext.data.metrics import bleu_score
        >>> candidate_corpus = [['My', 'full', 'pytorch', 'test'], ['Another', 'Sentence']]
        >>> references_corpus = [[['My', 'full', 'pytorch', 'test'], ['Completely', 'Different']], [['No', 'Match']]]
        >>> bleu_score(candidate_corpus, references_corpus)
            0.8408964276313782
    """

    assert max_n == len(weights), 'Length of the "weights" list has be equal to max_n'
    assert len(candidate_corpus) == len(references_corpus),\
        'The length of candidate and reference corpus should be the same'

    clipped_counts = torch.zeros(max_n)
    total_counts = torch.zeros(max_n)
    weights = torch.tensor(weights)

    candidate_len = 0.0
    refs_len = 0.0

    for (candidate, refs) in zip(candidate_corpus, references_corpus):
        candidate_len += len(candidate)

        # Get the length of the reference that's closest in length to the candidate
        refs_len_list = [float(len(ref)) for ref in refs]
        refs_len += min(refs_len_list, key=lambda x: abs(len(candidate) - x))

        reference_counters = _compute_ngram_counter(refs[0], max_n)
        for ref in refs[1:]:
            reference_counters = reference_counters | _compute_ngram_counter(ref, max_n)

        candidate_counter = _compute_ngram_counter(candidate, max_n)

        clipped_counter = candidate_counter & reference_counters

        for ngram in clipped_counter:
            clipped_counts[len(ngram) - 1] += clipped_counter[ngram]

        for ngram in candidate_counter:  # TODO: no need to loop through the whole counter
            total_counts[len(ngram) - 1] += candidate_counter[ngram]

    if min(clipped_counts) == 0:
        return 0.0
    else:
        pn = clipped_counts / total_counts
        log_pn = weights * torch.log(pn)
        score = torch.exp(sum(log_pn))

        bp = math.exp(min(1 - refs_len / candidate_len, 0))

        return bp * score.item()

```

**Implementation**

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

#### Code

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

#### Description

*Input*:

> Target/Reference Sentence (x): A man sleeping in a green room on a couch.

> Predicted/Candidate Sentence (x_hat): A man is sleeping on a green room on a couch .

*Procedure*

To calculate BERT Score,

1) Convert Target/Reference Sentence and Predicted/Candidate Sentence to Contextual Embeddings:

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session9-LearningRatesandEvaluationMetricsPart1/assets/bert_score_img2.png)

> Both the sentences are tokenized.

>> Target/Reference Tokens: ['A', 'man', 'sleeping', 'in', 'a', 'green', 'room', 'on', 'a', 'couch', '.']

>> Predicted/Candidate Tokens: ['A', 'man', 'is', 'sleeping', 'on', 'a', 'green', 'room', 'on', 'a', 'couch', '.']

> Both of these are then passed through BERT/ELMo or similar embedding model to obtain contextual embeddings represented by vectors. BERT model is used to perform tokenization and contextual embedding generation task. BERT, which tokenizes the input text into a sequence of word pieces, where unknown words are split into several commonly observed sequences of characters. The representation for each word piece is computed with a Transformer encoder by repeatedly applying self-attention and nonlinear transformations in an alternating fashion. The output generated by BERT model is embeddings repesented by below matrices. Each embedding of a sentence is a collection of vectors where each vector is contextual representation of each token in sentence.

>> Target/Reference Contextual Embeddings: *X*

>> Predicted/Candidate Contextual Embeddings: *X_hat*

2) Calculate Pairwise Cosine Similarity among Contextual Embeddings Vector:

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session9-LearningRatesandEvaluationMetricsPart1/assets/bert_score_pairwise_cosine_sim.png)

> Each vector in embedding is represented as below.

>> Target/Reference Token Vector: X<sub>i</sub>

>> Predicted/Candidate Token Vector: X_hat<sub>j</sub>

> For each vector X<sub>i</sub> in Reference Tokens, each X_hat<sub>j</sub> vector from candidate tokens is considered one by one. For each pair of (X<sub>i</sub> , X_hat<sub>j</sub>), cosine similarity is calculated as below.

> **Cosine similarity = (X<sub>i</sub><sup>T</sup> * X_hat<sub>j</sub>) / (||X<sub>i</sub>|| * ||X_hat<sub>j</sub>||)**

> Since the vectors X<sub>i</sub> and X_hat<sub>j</sub> are normalized, the calculation of cosine similarity reduces to **(X<sub>i</sub><sup>T</sup> * X_hat<sub>j</sub>)**

3) Calculate BERT SCore:

> i) Calculate Precision:

>> For each token vector X_hat<sub>j</sub> in candidate tokens, maximum of cosine similarity score of X_hat<sub>j</sub> with all X<sub>i</sub> reference tokens is taken as the similarity measure for that X_hat<sub>j</sub> token vector. To calculate precision for the entire sentence:

>>> a) With Importance Weighing: Previous work on similarity measures demonstrated that rare words can be more indicative for sentence similarity than common words. Inverse document frequency(IDF) is used to measure the importance of words. IDF scores are computed on corpus. Then weighted average of all cosine similarity of X_hat<sub>j</sub> with their IDF score is calculated. This average is the precision for the sentence.

>>> b) Without Importance Weighing: Average of all cosine similarity of X_hat<sub>j</sub> is calculated. This average is the precision for the sentence.

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session9-LearningRatesandEvaluationMetricsPart1/assets/bert_score_precision_formula.PNG)

> ii) Calculate Recall:

>> For each token vector X<sub>i</sub> in refernce tokens, maximum of cosine similarity score of X<sub>i</sub> with all X_hat<sub>j</sub> reference tokens is taken as the similarity measure for that X<sub>i</sub> token vector. To calculate recall for the entire sentence:

>>> a) With Importance Weighing: Previous work on similarity measures demonstrated that rare words can be more indicative for sentence similarity than common words. Inverse document frequency(IDF) is used to measure the importance of words. IDF scores are computed on corpus. Then weighted average of all cosine similarity of X<sub>i</sub> with their IDF score is calculated. This average is the recall for the sentence.

>>> b) Without Importance Weighing: Average of all cosine similarity of X<sub>i</sub> is calculated. This average is the recall for the sentence.

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session9-LearningRatesandEvaluationMetricsPart1/assets/bert_score_recall_formula.PNG)

> iii) Calculate F1 Score: F1 score is the harmonic ean of above calculated precision and recall.

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session9-LearningRatesandEvaluationMetricsPart1/assets/bert_score_fscore_formula.PNG)


>> Target/Reference Contextual Embeddings: *X*

>> Predicted/Candidate Contextual Embeddings: *X_hat*

#### Code

[Reference](https://github.com/Tiiiger/bert_score/blob/master/bert_score/score.py)

**Base Code**

```
def score(
    cands,
    refs,
    model_type=None,
    num_layers=None,
    verbose=False,
    idf=False,
    device=None,
    batch_size=64,
    nthreads=4,
    all_layers=False,
    lang=None,
    return_hash=False,
    rescale_with_baseline=False,
    baseline_path=None,
):
    """
    BERTScore metric.
    Args:
        - :param: `cands` (list of str): candidate sentences
        - :param: `refs` (list of str or list of list of str): reference sentences
        - :param: `model_type` (str): bert specification, default using the suggested
                  model for the target langauge; has to specify at least one of
                  `model_type` or `lang`
        - :param: `num_layers` (int): the layer of representation to use.
                  default using the number of layer tuned on WMT16 correlation data
        - :param: `verbose` (bool): turn on intermediate status update
        - :param: `idf` (bool or dict): use idf weighting, can also be a precomputed idf_dict
        - :param: `device` (str): on which the contextual embedding model will be allocated on.
                  If this argument is None, the model lives on cuda:0 if cuda is available.
        - :param: `nthreads` (int): number of threads
        - :param: `batch_size` (int): bert score processing batch size
        - :param: `lang` (str): language of the sentences; has to specify
                  at least one of `model_type` or `lang`. `lang` needs to be
                  specified when `rescale_with_baseline` is True.
        - :param: `return_hash` (bool): return hash code of the setting
        - :param: `rescale_with_baseline` (bool): rescale bertscore with pre-computed baseline
        - :param: `baseline_path` (str): customized baseline file
    Return:
        - :param: `(P, R, F)`: each is of shape (N); N = number of input
                  candidate reference pairs. if returning hashcode, the
                  output will be ((P, R, F), hashcode). If a candidate have 
                  multiple references, the returned score of this candidate is 
                  the *best* score among all references.
    """
    assert len(cands) == len(refs), "Different number of candidates and references"

    assert lang is not None or model_type is not None, "Either lang or model_type should be specified"

    ref_group_boundaries = None
    if not isinstance(refs[0], str):
        ref_group_boundaries = []
        ori_cands, ori_refs = cands, refs
        cands, refs = [], []
        count = 0
        for cand, ref_group in zip(ori_cands, ori_refs):
            cands += [cand] * len(ref_group)
            refs += ref_group
            ref_group_boundaries.append((count, count + len(ref_group)))
            count += len(ref_group)

    if rescale_with_baseline:
        assert lang is not None, "Need to specify Language when rescaling with baseline"

    if model_type is None:
        lang = lang.lower()
        model_type = lang2model[lang]
    if num_layers is None:
        num_layers = model2layers[model_type]

    tokenizer = get_tokenizer(model_type)
    model = get_model(model_type, num_layers, all_layers)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    if not idf:
        idf_dict = defaultdict(lambda: 1.0)
        # set idf for [SEP] and [CLS] to 0
        idf_dict[tokenizer.sep_token_id] = 0
        idf_dict[tokenizer.cls_token_id] = 0
    elif isinstance(idf, dict):
        if verbose:
            print("using predefined IDF dict...")
        idf_dict = idf
    else:
        if verbose:
            print("preparing IDF dict...")
        start = time.perf_counter()
        idf_dict = get_idf_dict(refs, tokenizer, nthreads=nthreads)
        if verbose:
            print("done in {:.2f} seconds".format(time.perf_counter() - start))

    if verbose:
        print("calculating scores...")
    start = time.perf_counter()
    all_preds = bert_cos_score_idf(
        model,
        refs,
        cands,
        tokenizer,
        idf_dict,
        verbose=verbose,
        device=device,
        batch_size=batch_size,
        all_layers=all_layers,
    ).cpu()

    if ref_group_boundaries is not None:
        max_preds = []
        for beg, end in ref_group_boundaries:
            max_preds.append(all_preds[beg:end].max(dim=0)[0])
        all_preds = torch.stack(max_preds, dim=0)

    use_custom_baseline = baseline_path is not None
    if rescale_with_baseline:
        if baseline_path is None:
            baseline_path = os.path.join(os.path.dirname(__file__), f"rescale_baseline/{lang}/{model_type}.tsv")
        if os.path.isfile(baseline_path):
            if not all_layers:
                baselines = torch.from_numpy(pd.read_csv(baseline_path).iloc[num_layers].to_numpy())[1:].float()
            else:
                baselines = torch.from_numpy(pd.read_csv(baseline_path).to_numpy())[:, 1:].unsqueeze(1).float()

            all_preds = (all_preds - baselines) / (1 - baselines)
        else:
            print(
                f"Warning: Baseline not Found for {model_type} on {lang} at {baseline_path}", file=sys.stderr,
            )

    out = all_preds[..., 0], all_preds[..., 1], all_preds[..., 2]  # P, R, F

    if verbose:
        time_diff = time.perf_counter() - start
        print(f"done in {time_diff:.2f} seconds, {len(refs) / time_diff:.2f} sentences/sec")

    if return_hash:
        return tuple(
            [
                out,
                get_hash(model_type, num_layers, idf, rescale_with_baseline, use_custom_baseline=use_custom_baseline,),
            ]
        )

    return out

```

**Implementation**

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
---

## Training Logs of BLEU Score, Perplexity and Bert Score with Explanation

### Training Logs

```
Epoch: 1:- Train loss: 5.321 | Val PPL: 61.466 | Val BLEU Score: 0.036 | Val BERT Score: Precision - 0.844, Recall - 0.871, F1 Score - 0.857 || Epoch time = 41.566s
Epoch: 2:- Train loss: 3.768 | Val PPL: 28.040 | Val BLEU Score: 0.107 | Val BERT Score: Precision - 0.882, Recall - 0.896, F1 Score - 0.889 || Epoch time = 44.592s
Epoch: 3:- Train loss: 3.163 | Val PPL: 18.227 | Val BLEU Score: 0.166 | Val BERT Score: Precision - 0.898, Recall - 0.908, F1 Score - 0.903 || Epoch time = 43.619s
Epoch: 4:- Train loss: 2.771 | Val PPL: 13.789 | Val BLEU Score: 0.204 | Val BERT Score: Precision - 0.906, Recall - 0.917, F1 Score - 0.911 || Epoch time = 44.338s
Epoch: 5:- Train loss: 2.481 | Val PPL: 11.615 | Val BLEU Score: 0.234 | Val BERT Score: Precision - 0.913, Recall - 0.924, F1 Score - 0.918 || Epoch time = 44.738s
```

### Explanation

**Note: Reference sentence means Target/Human annotated text. Candidate Sentence means Model generated text.**

#### 1) Perplexity

```
Epoch: 1:- Train loss: 5.321 | Val PPL: 61.466 || Epoch time = 41.566s
Epoch: 2:- Train loss: 3.768 | Val PPL: 28.040 || Epoch time = 44.592s
Epoch: 3:- Train loss: 3.163 | Val PPL: 18.227 || Epoch time = 43.619s
Epoch: 4:- Train loss: 2.771 | Val PPL: 13.789 || Epoch time = 44.338s
Epoch: 5:- Train loss: 2.481 | Val PPL: 11.615 || Epoch time = 44.738s
```

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session9-LearningRatesandEvaluationMetricsPart1/assets/ppl_graph.PNG)

We can see a diminishing perplexity trend which indicates that the model is learning and is becoming and lesser confused. Perplexity is exponentiation of entropy. Lesser the perplexity, lesser the entropy(randomness) of the model. Thus, a better model at predicting.

#### 2) BLEU Score

```
Epoch: 1:- Train loss: 5.321 | Val BLEU Score: 0.036 || Epoch time = 41.566s
Epoch: 2:- Train loss: 3.768 | Val BLEU Score: 0.107 || Epoch time = 44.592s
Epoch: 3:- Train loss: 3.163 | Val BLEU Score: 0.166 || Epoch time = 43.619s
Epoch: 4:- Train loss: 2.771 | Val BLEU Score: 0.204 || Epoch time = 44.338s
Epoch: 5:- Train loss: 2.481 | Val BLEU Score: 0.234 || Epoch time = 44.738s
```

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session9-LearningRatesandEvaluationMetricsPart1/assets/bleu_graph.PNG)

We can see an increasing BLEU Score trend. BLEU score is product of brevity penality(BP),which is best match length, and exponentiation of summation of weighted 1-gram, 2-gram, 3-gram, 4-gram presence count(cout of n-grams present in both reference and candidate sentence) known as modified precision. As BLEU score increases, BP and precsion increases. Higher BP means candidate translation will match the reference translations in length, in word choice, and word order. Higher precision means 1-gram, 2-gram, 3-gram and 4-gram of candidate are occurring more in number in reference sentences. Thus, predicted and target sentences are becoming similar and hence model is improving.

#### 3) BERT Score

```
Epoch: 1:- Train loss: 5.321 | Val BERT Score: Precision - 0.844, Recall - 0.871, F1 Score - 0.857 || Epoch time = 41.566s
Epoch: 2:- Train loss: 3.768 | Val BERT Score: Precision - 0.882, Recall - 0.896, F1 Score - 0.889 || Epoch time = 44.592s
Epoch: 3:- Train loss: 3.163 | Val BERT Score: Precision - 0.898, Recall - 0.908, F1 Score - 0.903 || Epoch time = 43.619s
Epoch: 4:- Train loss: 2.771 | Val BERT Score: Precision - 0.906, Recall - 0.917, F1 Score - 0.911 || Epoch time = 44.338s
Epoch: 5:- Train loss: 2.481 | Val BERT Score: Precision - 0.913, Recall - 0.924, F1 Score - 0.918 || Epoch time = 44.738s
```

We can see an increasing precision, recall and f1 score of BERT Score trend. BERTScore leverages the pre-trained contextual embeddings from BERT and matches words in candidate and reference sentences by cosine similarity to calculate precision, recall and F1 score. 

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session9-LearningRatesandEvaluationMetricsPart1/assets/bert_score_precision_graph.PNG)

For precision, each token in candidate sentence is matched by cosine similarity with all the tokens of reference sentence pairwise and the pair with maximum similarity becomes the precision score for that token. Mean of all the precision scores becomes precision of the model for that sentence. Precision scores for dataset is calculated in batches and their mean is the Precision of the model shown here. Increasing precision means increasing individual precision scores which in turn means increasing cosine similarity among reference and candidate tokens. Increasing cosine similarity means predicted and target sentences have words which are becoming more similar in nature. Thus, model is becoming better at generating text similar in nature to target text.

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session9-LearningRatesandEvaluationMetricsPart1/assets/bert_score_recall_graph.PNG)

For recall, each token in reference sentence is matched by cosine similarity with all the tokens of candidate sentence pairwise and the pair with maximum similarity becomes the recall score for that token. Mean of all the recall scores becomes precision of the model for that sentence. Recall scores for dataset is calculated in batches and their mean is the recall of the model shown here. Increasing recall means increasing individual recall scores which in turn means increasing cosine similarity among reference and candidate tokens. Increasing cosine similarity means predicted and target sentences have words which are becoming more similar in nature. Thus, model is becoming better at remembering context from target text.

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session9-LearningRatesandEvaluationMetricsPart1/assets/bert_score_f1_graph.PNG)

For F1 Score, harmonic mean of precion score and recall score for each batch is calculated. F1 scores for model,which is shown here, is mean of the F1 score of all the batches in the dataset. Increasing f1 means better precision and recall of the model. Thus, model is becoming not only better at generating text similar to target text but also becoming better at remembering context from target text.

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session9-LearningRatesandEvaluationMetricsPart1/assets/perplexity_bleu_bert_score_comp_graph.PNG)