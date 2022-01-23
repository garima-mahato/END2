# Sequence to Sequence for Sentiment Classification

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

**Model Architecture**

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session6-GRUs%2CSeq2SeqandIntroductiontoAttentionMechanism/assets/model.PNG)

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

**F1 Macro Score: 0.5041695303550974**

**Accuracy: 79.02439024390245 %**

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

#### 10 Correctly Classified Images

```
****************************************
***** Correctly Classified Text: *******
****************************************
1) Text: Obama has called the GOP budget social Darwinism. Nice try, but they believe in social creationism.
   Target Sentiment: Positive
   Predicted Sentiment: Positive

2) Text: In his teen years, Obama has been known to use marijuana and cocaine.
   Target Sentiment: Negative
   Predicted Sentiment: Negative

3) Text: RT @Professor_Why: #WhatsRomneyHiding - his connection to supporters of Critical Race Theory.... Oh wait, that was Obama, not Romney...
   Target Sentiment: Negative
   Predicted Sentiment: Negative

4) Text: RT @wardollarshome: Obama has approved more targeted assassinations than any modern US prez; READ & RT: http://t.co/bfC4gbBW
   Target Sentiment: Positive
   Predicted Sentiment: Positive

5) Text: Video shows federal officials joking about cost of lavish conference http://t.co/2i4SmoPM #obama #crime #p2 #news #tcot #teaparty
   Target Sentiment: Negative
   Predicted Sentiment: Negative

6) Text: one Chicago kid who says "Obama is my man" tells Jesse Watters that the gun violence in Chicago is like "World War 17"
   Target Sentiment: Negative
   Predicted Sentiment: Negative

7) Text: RT @ohgirlphrase: American kid "You're from the UK? Ohhh cool, So do you have tea with the Queen?". British kid: "Do you like, go to Mcdonalds with Obama?
   Target Sentiment: Negative
   Predicted Sentiment: Negative

8) Text: President Obama &lt; Lindsay Lohan RUMORS beginning cross shape lights on ST &lt; 1987 Analyst64 DC bicycle courier &lt; Video changes to scramble.
   Target Sentiment: Negative
   Predicted Sentiment: Negative

9) Text: Obama's Gender Advantage Extends to the States - 2012 Decoded: New detail on recent swing state polling further ... http://t.co/8iSanDGS
   Target Sentiment: Positive
   Predicted Sentiment: Positive

10) Text: Here's How Obama and the Democrats Will Win in 2012: Let's start by going back to the assorted polls questioning... http://t.co/zpg0TVm3
   Target Sentiment: Positive
   Predicted Sentiment: Positive
```

#### 10 Incorrectly Classified Images

```
****************************************
***** Incorrectly Classified Text: *****
****************************************
1) Text: IPA Congratulates President Barack Obama for Leadership Regarding JOBS Act: WASHINGTON, Apr 05, 2012 (BUSINESS W... http://t.co/8le3DC8E
   Target Sentiment: Negative
   Predicted Sentiment: Positive

2) Text: A valid explanation for why Obama won't let women on the golf course.   #WhatsRomneyHiding
   Target Sentiment: Positive
   Predicted Sentiment: Negative

3) Text: #WhatsRomneyHiding? Obama's dignity and sense of humor? #p2 #tcot
   Target Sentiment: Neutral
   Predicted Sentiment: Negative

4) Text: RealClearPolitics - Obama's Organizational Advantage on Full ...: As a small but electorally significant state t... http://t.co/3Ax22aBB
   Target Sentiment: Neutral
   Predicted Sentiment: Positive

5) Text: of course..I blame HAARP, sinners, aliens, Bush and Obama for all this.  :P....&gt;&gt; http://t.co/7eq8nebt
   Target Sentiment: Negative
   Predicted Sentiment: Positive

6) Text: RT @wilycyotee Pres. Obama's ongoing support of women is another reason I am so proud he is my President!  @edshow #Obama2012
   Target Sentiment: Neutral
   Predicted Sentiment: Negative

7) Text: If Obama win 2012 Election wait til 2016 he will have full white hair! just like Bill clinton!
   Target Sentiment: Neutral
   Predicted Sentiment: Negative

8) Text: Even CBS won't buy bogus WH explanation of Obama Supreme Court comments - at http://t.co/rkNdEmIy #withnewt #tpn #tcot #tlot #tpp #sgp
   Target Sentiment: Positive
   Predicted Sentiment: Negative

9) Text: RT @BunkerBlast: RT @teacherspets Obama's Budget: 'Interest Payments Will Exceed Defense Budget' in 2019  http://t.co/uddCXCjt
   Target Sentiment: Neutral
   Predicted Sentiment: Positive

10) Text: Barack Obama President Ronald Reagan's Initial Actions Project: President Ronald Reagan was also facing an econo... http://t.co/8Go8oCpf
   Target Sentiment: Negative
   Predicted Sentiment: Positive
```
