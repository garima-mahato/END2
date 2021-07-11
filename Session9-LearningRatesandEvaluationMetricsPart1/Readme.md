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

### 2) BLEU Score

[GitHub Notebook Link](https://github.com/garima-mahato/END2/blob/main/Session9-LearningRatesandEvaluationMetricsPart1/END2_Session9_BLEU_PPL_BertScore.ipynb)

[Google Colab Link](https://githubtocolab.com/garima-mahato/END2/blob/main/Session9-LearningRatesandEvaluationMetricsPart1/END2_Session9_BLEU_PPL_BertScore.ipynb)

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