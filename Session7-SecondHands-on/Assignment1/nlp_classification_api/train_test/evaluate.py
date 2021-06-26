import torch

from ..utils import categorical_accuracy

def evaluate(model, iterator, criterion):
    
    # initialize every epoch
    epoch_loss = 0
    epoch_acc = 0

    # deactivating dropout layers
    model.eval()
    
    # deactivates autograd
    with torch.no_grad():
    
        for batch in iterator:
        
            # retrieve text and no. of words
            text, text_lengths = batch.src   
            
            # convert to 1D tensor
            predictions = model(text, text_lengths).squeeze(1)
            
            # compute loss and accuracy
            loss = criterion(predictions, batch.trg)
            acc = categorical_accuracy(predictions, batch.trg)
            
            # keep track of loss and accuracy
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)