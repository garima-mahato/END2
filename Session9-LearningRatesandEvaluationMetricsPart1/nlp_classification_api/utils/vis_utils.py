from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, cmap=plt.cm.YlOrBr):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    (Adapted from scikit-learn docs).
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', origin='lower', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # Show all ticks
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # Label with respective list entries
           xticklabels=classes, yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label')

    # Set alignment of tick labels
    plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    return fig, ax

# visualize accuracy and loss graph
def visualize_graph(train_losses, train_acc, test_losses, test_acc):
  fig, axs = plt.subplots(2,2,figsize=(15,10))
  axs[0, 0].plot(train_losses)
  axs[0, 0].set_title("Training Loss")
  axs[1, 0].plot(train_acc)
  axs[1, 0].set_title("Training Accuracy")
  axs[0, 1].plot(test_losses)
  axs[0, 1].set_title("Test Loss")
  axs[1, 1].plot(test_acc)
  axs[1, 1].set_title("Test Accuracy")

def visualize_save_train_vs_test_graph(EPOCHS, dict_list, title, xlabel, ylabel, PATH, name="fig"):
  plt.figure(figsize=(20,10))
  #epochs = range(1,EPOCHS+1)
  for label, item in dict_list.items():
    x = np.linspace(1, EPOCHS+1, len(item))
    plt.plot(x, item, label=label)
  
  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.legend()
  plt.savefig(PATH+"/"+name+".png")

def evaluation_pred(model, iterator, itos=None, tokenizer=None):
  # deactivating dropout layers
  model.eval()
  if itos is not None:
    eval_df = pd.DataFrame(columns=['src','trg','pred'])
  else:
    eval_df = pd.DataFrame(columns=['trg','pred'])
  # deactivates autograd
  with torch.no_grad():
    for batch in iterator:
      # retrieve text and no. of words
      text, text_lengths = batch.src 
      label = batch.trg.cpu().numpy()
      
      # convert to 1D tensor
      predictions = model(text, text_lengths)
      top_pred = predictions.argmax(1, keepdim = True).cpu().numpy()
      batch_df = pd.DataFrame(top_pred, columns=['pred'])
      if itos is not None:
        src = [" ".join([tokenizer[ind] for ind in ex]) for ex in text.cpu().numpy().tolist()]
        batch_df['src'] = src
        batch_df['src'] = batch_df['src'].str.replace('<unk>','')
        batch_df['src'] = batch_df['src'].str.replace('<pad>','')
      batch_df['trg'] = label
      batch_df['pred'] = batch_df['pred'].astype(int)
      batch_df['trg'] = batch_df['trg'].astype(int)
      eval_df = pd.concat([eval_df, batch_df])
      
  return eval_df