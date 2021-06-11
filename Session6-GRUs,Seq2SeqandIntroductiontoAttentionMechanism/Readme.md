<!--# Session 6: Encoder-Decoder

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

	tweets	labels
0	Obama has called the GOP budget social Darwini...	1
1	In his teen years, Obama has been known to use...	0
2	IPA Congratulates President Barack Obama for L...	0
3	RT @Professor_Why: #WhatsRomneyHiding - his co...	0
4	RT @wardollarshome: Obama has approved more ta...	1
...	...	...
1359	@liberalminds Its trending idiot.. Did you loo...	0
1360	RT @AstoldByBass: #KimKardashiansNextBoyfriend...	0
1361	RT @GatorNation41: gas was $1.92 when Obama to...	1
1362	@xShwag haha i know im just so smart, i mean y...	1
1363	#OBAMA: DICTATOR IN TRAINING. If he passes t...	0
1364 rows × 2 columns

### EDA

### Model Building

#### Encoder

```
class EncoderLSTM(nn.Module):
  def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=1):
    super().__init__()
    self.hidden_dim = hidden_dim
    self.n_layers = n_layers
    # embedding layer
    self.embedding = nn.Embedding(vocab_size, embedding_dim) #, padding_idx=pad_index)

    # lstm layer
    self.lstm = nn.LSTM(embedding_dim, 
                        hidden_dim,
                        num_layers=n_layers,
                        batch_first=True)
    
    self.enc_context = nn.Linear(hidden_dim, output_dim)

  def forward(self, text, visualize=False): #, text_lengths):
    embedded = self.embedding(text)
    
    lstm_output, (hidden, cell) = self.lstm(embedded)
    
    output = self.enc_context(hidden.squeeze(0))
    
    if visualize:
      enc_op = lstm_output[0].detach().cpu().numpy()
      fig, ax = plt.subplots(figsize=(20,10)) 
      sns.heatmap(enc_op, fmt=".2f", vmin=-1, vmax=1, annot=True, cmap="YlGnBu", ax=ax)#.set(title=f"Encoded Representation from Encoder")
      plt.title('Hidden State in each time step of Encoder', fontsize = 20) # title with fontsize 20
      plt.xlabel('Hidden state', fontsize = 15) # x-axis label with fontsize 15
      plt.ylabel('Time Step', fontsize = 15) # y-axis label with fontsize 15
      plt.show()

      enc_op = output.detach().cpu().numpy()
      fig, ax = plt.subplots(figsize=(20,4)) 
      sns.heatmap(enc_op, fmt=".2f", vmin=-1, vmax=1, annot=True, cmap="YlGnBu", ax=ax, annot_kws={"size": 20})#.set(title=f"Encoded Representation from Encoder")
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
    self.lstm = nn.LSTM(enc_dim, 
                        hidden_dim,
                        num_layers=n_layers,
                        batch_first=True)
    
    self.decoded_op = nn.Linear(hidden_dim, output_dim)
  
  def forward(self, enc_context, enc_hidden, visualize=False): 
    packed_output, (hidden, cell) = self.lstm(enc_context.unsqueeze(1), enc_hidden)
    
    output = self.decoded_op(hidden).squeeze(0)
    if visualize:
      enc_op = packed_output[0].detach().cpu().numpy()
      fig, ax = plt.subplots(figsize=(50,4)) 
      sns.heatmap(enc_op, fmt=".2f", vmin=-1, vmax=1, annot=True, cmap="YlGnBu", ax=ax, annot_kws={"size": 20})#.set(title=f"Decoded Representation from Decoder")
      plt.title('Hidden State in each time step of  Decoder', fontsize = 20) # title with fontsize 20
      plt.xlabel('Hidden State', fontsize = 15) # x-axis label with fontsize 15
      plt.ylabel('Time Step', fontsize = 15) # y-axis label with fontsize 15
      plt.show()

      enc_op = output.detach().cpu().numpy()
      fig, ax = plt.subplots(figsize=(20,4)) 
      sns.heatmap(enc_op, fmt=".2f", vmin=-1, vmax=1, annot=True, cmap="YlGnBu", ax=ax, annot_kws={"size": 20})#.set(title=f"Encoded Representation from Encoder")
      plt.title('Decoded Representation from Encoder', fontsize = 20) # title with fontsize 20
      plt.show()
    return output
```

#### Model

```
class LSTMEncoderDecoderClassifier(nn.Module):
    
    # Define all the layers used in model
    def __init__(self, vocab_size, embedding_dim, hidden_enc_dim, hidden_dec_dim, context_dim, output_dim, n_classes, n_enc_layers=1, n_dec_layers=1):
        # Constructor
        super().__init__()

        # encoder layer
        self.encoder = EncoderLSTM(vocab_size, embedding_dim, hidden_enc_dim, context_dim, n_enc_layers)

        # decoder layer
        self.decoder = DecoderLSTM(context_dim, hidden_dec_dim, output_dim, n_dec_layers)

        # output layer
        self.linear_output = nn.Linear(output_dim, n_classes)

    def forward(self, text, visualize=False): #, text_lengths):
        # text = [batch size,sent_length]
        encoded_context, encoded_hidden = self.encoder(text, visualize)#, text_lengths)
        decoded = self.decoder(encoded_context, encoded_hidden, visualize)
        prediction = self.linear_output(decoded)

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

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session6-GRUs%2CSeq2SeqandIntroductiontoAttentionMechanism/assets/enc1.PNG)

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session6-GRUs%2CSeq2SeqandIntroductiontoAttentionMechanism/assets/enc2.PNG)

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session6-GRUs%2CSeq2SeqandIntroductiontoAttentionMechanism/assets/dec1.PNG)

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session6-GRUs%2CSeq2SeqandIntroductiontoAttentionMechanism/assets/dec2.PNG)

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session6-GRUs%2CSeq2SeqandIntroductiontoAttentionMechanism/assets/pred.PNG)
-->
