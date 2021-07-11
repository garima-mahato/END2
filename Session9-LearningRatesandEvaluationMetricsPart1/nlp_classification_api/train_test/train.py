import torch

from ..utils import categorical_accuracy

def train(model, iterator, optimizer, criterion):
    
    # initialize every epoch 
    epoch_loss = 0
    epoch_acc = 0
    
    # set the model in training phase
    model.train()  
    
    for batch in iterator:
        
        # resets the gradients after every batch
        optimizer.zero_grad()   
        
        # retrieve text and no. of words
        text, text_lengths = batch.src   
        
        # convert to 1D tensor
        predictions = model(text, text_lengths)
        # print(predictions.shape)
        # print(batch.label.shape)
        # compute the loss
        loss = criterion(predictions, batch.trg)        
        
        # compute the categorical accuracy
        acc = categorical_accuracy(predictions, batch.trg)   
        
        # backpropage the loss and compute the gradients
        loss.backward()       
        
        # update the weights
        optimizer.step()      
        
        # loss and accuracy
        epoch_loss += loss.item()  
        epoch_acc += acc.item()    
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)