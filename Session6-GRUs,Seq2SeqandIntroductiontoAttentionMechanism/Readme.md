# Session 6: Encoder-Decoder

## Assignment

1) Take the last code  (+tweet dataset) and convert that in such a war that:

> 1)encoder: an RNN/LSTM layer takes the words in a sentence one by one and finally converts them into a single vector. VERY IMPORTANT TO MAKE THIS SINGLE VECTOR

> 2) this single vector is then sent to another RNN/LSTM that also takes the last prediction as its second input. Then we take the final vector from this Cell
and send this final vector to a Linear Layer and make the final prediction. 

> 3) This is how it will look:

>> 1) embedding

>> 2) word from a sentence +last hidden vector -> encoder -> single vector

>> 3) single vector + last hidden vector -> decoder -> single vector

>> 4) single vector -> FC layer -> Prediction

2) Your code will be checked for plagiarism, and if we find that you have copied from the internet, then -100%. 

3) The code needs to look as simple as possible, the focus is on making encoder/decoder classes and how to link objects together

4) Getting good accuracy is NOT the target, but must achieve at least 45% or more

5) Once the model is trained, take one sentence, "print the outputs" of the encoder for each step and "print the outputs" for each step of the decoder. ← THIS IS THE ACTUAL ASSIGNMENT

## Solution:

### Dataset

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session6-GRUs%2CSeq2SeqandIntroductiontoAttentionMechanism/assets/dataset.PNG)

### EDA

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session6-GRUs%2CSeq2SeqandIntroductiontoAttentionMechanism/assets/eda1.PNG)

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session6-GRUs%2CSeq2SeqandIntroductiontoAttentionMechanism/assets/eda2.PNG)

### Model Building

[Link to Code in GitHub](https://github.com/garima-mahato/END2/blob/main/Session6-GRUs,Seq2SeqandIntroductiontoAttentionMechanism/Session6_LSTM_Encoder_Decoder_TweetsDataset.ipynb)

[Link to colab code](https://githubtocolab.com/garima-mahato/END2/blob/main/Session6-GRUs,Seq2SeqandIntroductiontoAttentionMechanism/Session6_LSTM_Encoder_Decoder_TweetsDataset.ipynb)

#### Encoder

```
class EncoderLSTM(nn.Module):
  def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=1):
    super().__init__()
    self.hidden_dim = hidden_dim
    self.n_layers = n_layers
    # embedding layer
    self.embedding = nn.Embedding(vocab_size, embedding_dim)

    # lstm layer
    self.lstm = nn.LSTM(embedding_dim, 
                        hidden_dim,
                        num_layers=n_layers,
                        batch_first=True)
    
    self.enc_context = nn.Linear(hidden_dim, output_dim)
  
  def initHidden(self, batch_size, device):
    return (torch.zeros(1, batch_size, self.hidden_dim, device=device), torch.zeros(1, batch_size, self.hidden_dim, device=device))

  def forward(self, text, enc_hidden, visualize=False, verbose=False):
    # hidden, cell = enc_hidden
    embedded = self.embedding(text)
    
    lstm_output, (hidden, cell) = self.lstm(embedded, enc_hidden)
    
    output = self.enc_context(hidden.squeeze(0))
    
    if verbose:
      print('inside encoder:-')
      print(f'shape of text input to encoder: {text.shape}')
      print(f'shape of Embedding layer output: {embedded.shape}')
      print(f'shape of lstm layer output: {hidden.shape}')
      print(f'shape of fc layer output: {output.shape}')
      print(f'shape of encoder output: {output.shape}')

    if visualize:
      enc_op = torch.cat((enc_hidden[0],lstm_output),dim=1)[0].detach().cpu().numpy()
      fig, ax = plt.subplots(figsize=(20,10)) 
      sns.heatmap(enc_op, fmt=".2f", vmin=-1, vmax=1, annot=True, cmap="YlGnBu", ax=ax)
      plt.title('Hidden State in each time step of Encoder', fontsize = 20) # title with fontsize 20
      plt.xlabel('Hidden state', fontsize = 15) # x-axis label with fontsize 15
      plt.ylabel('Time Step', fontsize = 15) # y-axis label with fontsize 15
      plt.show()

      enc_op = output.detach().cpu().numpy()
      fig, ax = plt.subplots(figsize=(20,4)) 
      sns.heatmap(enc_op, fmt=".2f", vmin=-1, vmax=1, annot=True, cmap="YlGnBu", ax=ax, annot_kws={"size": 20})
      plt.title('Encoded Representation from Encoder', fontsize = 20) # title with fontsize 20
      plt.show()

    return output, (hidden, cell)
```

#### Decoder

```
class DecoderLSTM(nn.Module):
  def __init__(self, enc_dim, hidden_dim, output_dim, n_layers=1):
    super().__init__()

    # lstm layer
    self.lstm = nn.LSTMCell(enc_dim, 
                        hidden_dim,
                        bias=False)
                        # num_layers=n_layers,
                        # batch_first=True)
    
    self.decoded_op = nn.Linear(hidden_dim, output_dim)
  
  def forward(self, enc_context, enc_hidden, dec_steps=2, visualize=False, verbose=False):
    dec_input = enc_context.unsqueeze(1)
    hidden, cell = enc_hidden
    hidden = hidden.squeeze(0)
    cell = cell.squeeze(0)
    dec_outputs = []
    for i in range(dec_steps):
      hidden, cell = self.lstm(enc_context, (hidden, cell))
      dec_outputs.append(hidden)
    dec_output = torch.stack(dec_outputs, dim=1)
        
    output = self.decoded_op(hidden)

    if verbose:
      print('inside decoder:-')
      print(f'shape of output from encoder which goes as input to decoder: {enc_context.shape}')
      print(f'shape of lstm layer output: {hidden.shape}')
      print(f'shape of fc layer output: {output.shape}')
      print(f'shape of decoder output: {output.shape}')

    if visualize:
      enc_op = dec_output[0].detach().cpu().numpy()
      fig, ax = plt.subplots(figsize=(50,4)) 
      sns.heatmap(enc_op, fmt=".2f", vmin=-1, vmax=1, annot=True, cmap="YlGnBu", ax=ax, annot_kws={"size": 20})
      plt.title('Hidden State in each time step of Decoder', fontsize = 20) # title with fontsize 20
      plt.xlabel('Hidden State', fontsize = 15) # x-axis label with fontsize 15
      plt.ylabel('Time Step', fontsize = 15) # y-axis label with fontsize 15
      plt.show()

      enc_op = output.detach().cpu().numpy()
      fig, ax = plt.subplots(figsize=(20,4)) 
      sns.heatmap(enc_op, fmt=".2f", vmin=-1, vmax=1, annot=True, cmap="YlGnBu", ax=ax, annot_kws={"size": 20})
      plt.title('Decoded Representation from Decoder', fontsize = 20) # title with fontsize 20
      plt.show()
    return output
```

#### Model

```
class LSTMEncoderDecoderClassifier(nn.Module):
    
    # Define all the layers used in model
    def __init__(self, device, vocab_size, embedding_dim, hidden_enc_dim, hidden_dec_dim, context_dim, output_dim, n_classes, n_enc_layers=1, n_dec_layers=1):
        # Constructor
        super().__init__()
        self.device = device

        # encoder layer
        self.encoder = EncoderLSTM(vocab_size, embedding_dim, hidden_enc_dim, context_dim, n_enc_layers)

        # decoder layer
        self.decoder = DecoderLSTM(context_dim, hidden_dec_dim, output_dim, n_dec_layers)

        # output layer
        self.linear_output = nn.Linear(output_dim, n_classes)

    def forward(self, text, dec_steps=2, visualize=False, verbose=False): #, text_lengths):
        # text = [batch size,sent_length]
        enc_h = self.encoder.initHidden(text.shape[0], self.device)
        
        encoded_context, encoded_hidden = self.encoder(text, enc_h, visualize, verbose)#, text_lengths)
        decoded = self.decoder(encoded_context, encoded_hidden, dec_steps, visualize, verbose)
        prediction = self.linear_output(decoded)

        if verbose:
          print(f'shape of final output: {prediction.shape}')

        if visualize:
          enc_op = prediction.detach().cpu().numpy()
          fig, ax = plt.subplots(figsize=(20,4)) 
          sns.heatmap(enc_op, fmt=".2f", vmin=-1, vmax=1, annot=True, cmap="YlGnBu", ax=ax)#.set(title=f"Encoded Representation from Encoder")
          plt.title('Final Prediction', fontsize = 20) # title with fontsize 20
          plt.show()
        
        return prediction
```

### Training and Testing

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session6-GRUs%2CSeq2SeqandIntroductiontoAttentionMechanism/assets/training.PNG)

### Visualization

#### Train and Test Accuracy/Loss

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session6-GRUs%2CSeq2SeqandIntroductiontoAttentionMechanism/assets/train_test_acc_loss_comp.PNG)

#### Train vs Test Accuracy

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session6-GRUs%2CSeq2SeqandIntroductiontoAttentionMechanism/assets/train_test_acc.PNG)

#### Train vs Test Loss

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session6-GRUs%2CSeq2SeqandIntroductiontoAttentionMechanism/assets/train_test_loss.PNG)


### Evaluation

#### Confusion Matrix

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session6-GRUs%2CSeq2SeqandIntroductiontoAttentionMechanism/assets/conf_matrix.PNG)

#### Evaluation Metrics Result

F1 Macro Score: 0.4941398028137809
Accuracy: 78.04878048780488 %

#### Model Visualization on sample result

```
**Sample Sentence**: Obama has called the GOP budget social Darwinism. Nice try, but they believe in social creationism.

**Target Label**: 1

**Predicted Label**: 1
```

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session6-GRUs%2CSeq2SeqandIntroductiontoAttentionMechanism/assets/enc1.PNG)

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session6-GRUs%2CSeq2SeqandIntroductiontoAttentionMechanism/assets/enc2.PNG)

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session6-GRUs%2CSeq2SeqandIntroductiontoAttentionMechanism/assets/dec1.PNG)

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session6-GRUs%2CSeq2SeqandIntroductiontoAttentionMechanism/assets/dec2.PNG)

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session6-GRUs%2CSeq2SeqandIntroductiontoAttentionMechanism/assets/pred.PNG)

