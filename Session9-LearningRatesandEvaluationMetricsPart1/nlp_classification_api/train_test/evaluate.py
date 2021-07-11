import torch

from ..metrics import *

from ..utils import categorical_accuracy

def evaluate(model, iterator, criterion, eval_metric=None, labels=None):
    
    # initialize every epoch
    epoch_loss = 0
    epoch_acc = 0
    metric = []

    # deactivating dropout layers
    model.eval()

    if eval_metric == 'prec_recall_f1':
        pcf = precision_recall_f1score(labels)
    
    # deactivates autograd
    with torch.no_grad():
    
        for batch in iterator:
        
            # retrieve text and no. of words
            text, text_lengths = batch.src   
            
            # convert to 1D tensor
            predictions = model(text, text_lengths).squeeze(1)
            
            # compute loss and accuracy
            loss = criterion(predictions, batch.trg)
            if eval_metric == 'prec_recall_f1':
                pcf.update(predictions, batch.trg)
            acc = categorical_accuracy(predictions, batch.trg)
            
            # keep track of loss and accuracy
            epoch_loss += loss.item()
            if eval_metric is None:
                epoch_acc += acc.item()

    metric.append(epoch_acc / len(iterator)) 
        
    if eval_metric == 'prec_recall_f1':
        metric.append(pcf.calculate())
    
    return epoch_loss / len(iterator), metric