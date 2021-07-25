# Assignment

1) Follow the similar strategy as we did in our [baby-steps-code](https://colab.research.google.com/drive/1IlorkvXhZgmd_sayOVx4bC_I5Qpdzxk_?usp=sharing) , but replace GRU with LSTM. In your code you must:

> 1) Perform 1 full feed forward step for the encoder manually

> 2) Perform 1 full feed forward step for the decoder manually.

> 3) You can use any of the 3 attention mechanisms that we discussed. 

2) Explain your steps in the readme file and

3) Submit the assignment asking for these things:

> 1) Link to the readme file that must explain Encoder/Decoder Feed-forward manual steps and the attention mechanism that you have used - 500 pts

> 2) Copy-paste (don't redirect to github), the Encoder Feed Forward steps for 2 words - 250 pts

> 3) Copy-paste (don't redirect to github), the Decoder Feed Forward steps for 2 words - 250 pts

---

# Solution

[Link to GitHub Code](https://github.com/garima-mahato/END2/blob/main/Session11-AdvancedConceptsAnd4thHandsOn/END2_Session11_4thHandsOn.ipynb)

[Link to Colab Code](https://githubtocolab.com/garima-mahato/END2/blob/main/Session11-AdvancedConceptsAnd4thHandsOn/END2_Session11_4thHandsOn.ipynb)

**Input Sentence: 'elles sont tres grosses .'**

**Target Sentence: 'they are very big .'**


## Data Preparation

#### Step 1: Add <EOS> special token at the end of both sentences.

*Input Sentence: 'elles sont tres grosses . \<EOS\>'*

*Target Sentence: 'they are very big . \<EOS\>'*

#### Step 2: Convert both sentences into tokens.

*Input Tokens: ['elles', 'sont', 'tres', 'grosses', '.', '\<EOS\>']*

*Target Tokens: ['they', 'are', 'very', 'big', '.', '\<EOS\>']*

#### Step 3: Convert tokens of both sentences into numbers.

*Input Tensor: [351, 349, 121, 1062, 5, 1]*

*Target Tensor: [221, 124, 303, 131, 4, 1]*

#### Step 4: Add batch dimension into tensors of both sentences.

*Input Tensor: [[351, 349, 121, 1062, 5, 1]]*

*Target Tensor: [[221, 124, 303, 131, 4, 1]]*


## Architecture

### Encoder Architecture

The encoder of a seq2seq network is a RNN that outputs some value for every word from the input sentence. For every input word the encoder outputs a vector and a hidden state, and uses the hidden state for the next input word.

![image](https://pytorch.org/tutorials/_images/encoder-network.png)

Encoder consist of:

> 1) **Embedding Layer** to convert each token in target tensor to embedding.

> 2) **LSTM Layer** to analyse the embedding and generate encodings to be utilised by the decoder for translation.


### Attention Decoder Architecture

If only the context vector is passed between the encoder and decoder, that single vector carries the burden of encoding the entire sentence.

Attention allows the decoder network to "focus" on a different part of the encoder's outputs for every step of the decoder's own outputs. First we calculate a set of *attention weights*. These will be multiplied by the encoder output vectors to create a weighted combination. The result (called ``attn_applied`` in the code) should contain information about that specific part of the input sequence, and thus help the decoder choose the right output words.

![image](https://i.imgur.com/1152PYf.png)

Calculating the attention weights is done with another feed-forward layer ``attn``, using the decoder's input and hidden state as inputs. Because there are sentences of all sizes in the training data, to actually create and train this layer we have to choose a maximum sentence length (input length, for encoder outputs) that it can apply to. Sentences of the maximum length will use all the attention weights, while shorter sentences will only use the first few.

![image](https://pytorch.org/tutorials/_images/attention-decoder-network.png)


## Encoder Feed-Forward Steps

#### 1) Feed Forward Step 1

Tensor corresponding to 1st word/token('elles') [351] is taken from input tensor. It is then reshaped to add batch dimension and passed to the embedding layer.

```
embedded_input = embedding(input_tensor[i].view(-1, 1))
```

Output of embedding layer is passed into LSTM Layer.

```
output, (encoder_hidden, encoder_cell_state) = lstm(embedded_input, (encoder_hidden, encoder_cell_state))
encoder_outputs[i] += output[0,0]
```

Below is the view of Encoder Output. The encoding learnt by LSTM with respect to the first token or word in 256 dimensions is being represented.

```
plot_matrix(encoder_outputs)
```

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session11-AdvancedConceptsAnd4thHandsOn/assets/enc1.PNG)

#### 2) Feed Forward Step 2

Tensor corresponding to 2nd word/token('sont') [349] is taken from input tensor. It is then reshaped to add batch dimension and passed to the embedding layer.

```
embedded_input = embedding(input_tensor[i].view(-1, 1))
```

Output of embedding layer is passed into LSTM Layer.

```
output, (encoder_hidden, encoder_cell_state) = lstm(embedded_input, (encoder_hidden, encoder_cell_state))
encoder_outputs[i] += output[0,0]
```

Below is the view of Encoder Output. The encoding learnt by LSTM with respect to the 1st to 2nd tokens or words in 256 dimensions is being represented.

```
plot_matrix(encoder_outputs)
```

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session11-AdvancedConceptsAnd4thHandsOn/assets/enc2.PNG)

#### 3) Feed Forward Step 3

Tensor corresponding to 3rd word/token('tres') [121] is taken from input tensor. It is then reshaped to add batch dimension and passed to the embedding layer.

```
embedded_input = embedding(input_tensor[i].view(-1, 1))
```

Output of embedding layer is passed into LSTM Layer.

```
output, (encoder_hidden, encoder_cell_state) = lstm(embedded_input, (encoder_hidden, encoder_cell_state))
encoder_outputs[i] += output[0,0]
```

Below is the view of Encoder Output. The encoding learnt by LSTM with respect to the 1st to 3rd tokens or words in 256 dimensions is being represented.

```
plot_matrix(encoder_outputs)
```

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session11-AdvancedConceptsAnd4thHandsOn/assets/enc3.PNG)

#### 4) Feed Forward Step 4

Tensor corresponding to 4th word/token('grosses') [1062] is taken from input tensor. It is then reshaped to add batch dimension and passed to the embedding layer.

```
embedded_input = embedding(input_tensor[i].view(-1, 1))
```

Output of embedding layer is passed into LSTM Layer.

```
output, (encoder_hidden, encoder_cell_state) = lstm(embedded_input, (encoder_hidden, encoder_cell_state))
encoder_outputs[i] += output[0,0]
```

Below is the view of Encoder Output. The encoding learnt by LSTM with respect to the 1st to 4th tokens or words in 256 dimensions is being represented.

```
plot_matrix(encoder_outputs)
```

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session11-AdvancedConceptsAnd4thHandsOn/assets/enc4.PNG)

#### 5) Feed Forward Step 5

Tensor corresponding to 5th word/token('.') [5] is taken from input tensor. It is then reshaped to add batch dimension and passed to the embedding layer.

```
embedded_input = embedding(input_tensor[i].view(-1, 1))
```

Output of embedding layer is passed into LSTM Layer.

```
output, (encoder_hidden, encoder_cell_state) = lstm(embedded_input, (encoder_hidden, encoder_cell_state))
encoder_outputs[i] += output[0,0]
```

Below is the view of Encoder Output. The encoding learnt by LSTM with respect to the 1st to 5th tokens or words in 256 dimensions is being represented.

```
plot_matrix(encoder_outputs)
```

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session11-AdvancedConceptsAnd4thHandsOn/assets/enc5.PNG)

#### 6) Feed Forward Step 6

Tensor corresponding to 6th word/token('<EOS>') [1] is taken from input tensor. It is then reshaped to add batch dimension and passed to the embedding layer.

```
embedded_input = embedding(input_tensor[i].view(-1, 1))
```

Output of embedding layer is passed into LSTM Layer.

```
output, (encoder_hidden, encoder_cell_state) = lstm(embedded_input, (encoder_hidden, encoder_cell_state))
encoder_outputs[i] += output[0,0]
```

Below is the view of Encoder Output. The encoding learnt by LSTM with respect to the 1st to 6th tokens or words in 256 dimensions is being represented.

```
plot_matrix(encoder_outputs)
```

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session11-AdvancedConceptsAnd4thHandsOn/assets/enc6.PNG)


## Decoder Feed-Forward Steps

#### 1) Feed Forward Step 1

After all words in the input sentence have been encoded, then the task of decoding, or more specifically translating, is started. 

Teacher forcing is used with 50% probability. If 1 comes, then teacher forcing is used and first token from target tensor [221] is passed else \<SOS\> is passed as decoder's input.

```
teacher_forcing = 0.5
decoder_input = torch.tensor([[target_indices[i] if np.random.binomial(1,teacher_forcing) == 1 else SOS_token]], device=device)
```

The last hidden and cell states of the encoder is sent to the decoder.Since the last hidden and cell state are by product of encoder afer having seen the entire sentence, they capture the essence and context of the input sentence.

```
decoder_hidden = encoder_hidden
decoder_cell_state = encoder_cell_state
```

The numerical token of *decoder_input* must be converted into embedding of 256 dimension for standard contextual represention.

```
embedded = embedding(decoder_input)
```

The embeddings of *decoder_input* and encoder's last hidden state are appended vertically. This appended/concatenated tensor is passed through an FC Layer which maps the 512 dimensions to 10 dimensions where 10 represents the max length of sentences in the dataset to calculate the weights/importance that should be given to each token of input while predicting a particular word. The weeights are then passed through softmax layer to normalize the weights. These weights reperesnt the attention that must be given to each token for a particular output. So, these weights are multiplied with the encoder's outputs. The result is a tensor representing the encoder tokens which must be focused on to generate English token.
```
attn_weight_layer = nn.Linear(256 * 2, 10).to(device)
input_to_lstm_layer = nn.Linear(256 * 2, 256).to(device)

attn_weights = attn_weight_layer(torch.cat((embedded[0], decoder_hidden[0]), 1))
attn_weights = F.softmax(attn_weights, dim = 1)
attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
```

The embedded input and focused state are concatenated and passed through FC layer to convert into 256 dimensions and then batch is added.

```
input_to_lstm = input_to_lstm_layer(torch.cat((embedded[0], attn_applied[0]), 1))
input_to_lstm = input_to_lstm.unsqueeze(0)
```

The above transformed input is then passed into LSTM layer for learning translation to English.

```
output, (decoder_hidden, decoder_cell_state) = dec_lstm(input_to_lstm, (decoder_hidden, decoder_cell_state))
```

The output of LSTM layer is passed to ReLU and Softmax for selecting one of the words from the English vocabulary formed from the dataset.

```
output = F.relu(output)
output = F.softmax(output_word_layer(output[0]), dim = 1)
```

The word having the maximum softmax value is chosen as the predicted word.

```
top_value, top_index = output.data.topk(1)
pred_op.append(top_index.item())
```

Below is the attention map for predicting 1st English word.

```
decoder_attentions[i] = attn_weights.data.detach().cpu()
print(f"Target: {target_sentence.split(' ')[i]} \nPredicted: {output_lang.index2word[top_index.item()]}")#, attn_weights)
print("Attention Map")
showAttention(input_sentence, [output_lang.index2word[pred] for pred in pred_op], decoder_attentions)
```

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session11-AdvancedConceptsAnd4thHandsOn/assets/dec1.PNG)

#### 2) Feed Forward Step 2

Teacher forcing is used with 50% probability. If 1 comes, then teacher forcing is used and first token from target tensor [224] is passed else previously predicted word is passed as decoder's input.

```
teacher_forcing = 0.5
decoder_input = torch.tensor([[target_indices[i] if np.random.binomial(1,teacher_forcing) == 1 else SOS_token]], device=device)
```

The last hidden and cell states of the encoder is sent to the decoder.Since the last hidden and cell state are by product of encoder afer having seen the entire sentence, they capture the essence and context of the input sentence.

```
decoder_hidden = encoder_hidden
decoder_cell_state = encoder_cell_state
```

The numerical token of *decoder_input* must be converted into embedding of 256 dimension for standard contextual represention.

```
embedded = embedding(decoder_input)
```

The embeddings of *decoder_input* and encoder's last hidden state are appended vertically. This appended/concatenated tensor is passed through an FC Layer which maps the 512 dimensions to 10 dimensions where 10 represents the max length of sentences in the dataset to calculate the weights/importance that should be given to each token of input while predicting a particular word. The weeights are then passed through softmax layer to normalize the weights. These weights reperesnt the attention that must be given to each token for a particular output. So, these weights are multiplied with the encoder's outputs. The result is a tensor representing the encoder tokens which must be focused on to generate English token.
```
attn_weight_layer = nn.Linear(256 * 2, 10).to(device)
input_to_lstm_layer = nn.Linear(256 * 2, 256).to(device)

attn_weights = attn_weight_layer(torch.cat((embedded[0], decoder_hidden[0]), 1))
attn_weights = F.softmax(attn_weights, dim = 1)
attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
```

The embedded input and focused state are concatenated and passed through FC layer to convert into 256 dimensions and then batch is added.

```
input_to_lstm = input_to_lstm_layer(torch.cat((embedded[0], attn_applied[0]), 1))
input_to_lstm = input_to_lstm.unsqueeze(0)
```

The above transformed input is then passed into LSTM layer for learning translation to English.

```
output, (decoder_hidden, decoder_cell_state) = dec_lstm(input_to_lstm, (decoder_hidden, decoder_cell_state))
```

The output of LSTM layer is passed to ReLU and Softmax for selecting one of the words from the English vocabulary formed from the dataset.

```
output = F.relu(output)
output = F.softmax(output_word_layer(output[0]), dim = 1)
```

The word having the maximum softmax value is chosen as the predicted word.

```
top_value, top_index = output.data.topk(1)
pred_op.append(top_index.item())
```

Below is the attention map for predicting 2nd English word.

```
decoder_attentions[i] = attn_weights.data.detach().cpu()
print(f"Target: {target_sentence.split(' ')[i]} \nPredicted: {output_lang.index2word[top_index.item()]}")#, attn_weights)
print("Attention Map")
showAttention(input_sentence, [output_lang.index2word[pred] for pred in pred_op], decoder_attentions)
```

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session11-AdvancedConceptsAnd4thHandsOn/assets/dec2.PNG)

#### 3) Feed Forward Step 3

Teacher forcing is used with 50% probability. If 1 comes, then teacher forcing is used and first token from target tensor [303] is passed else previously predicted word is passed as decoder's input.

```
teacher_forcing = 0.5
decoder_input = torch.tensor([[target_indices[i] if np.random.binomial(1,teacher_forcing) == 1 else SOS_token]], device=device)
```

The last hidden and cell states of the encoder is sent to the decoder.Since the last hidden and cell state are by product of encoder afer having seen the entire sentence, they capture the essence and context of the input sentence.

```
decoder_hidden = encoder_hidden
decoder_cell_state = encoder_cell_state
```

The numerical token of *decoder_input* must be converted into embedding of 256 dimension for standard contextual represention.

```
embedded = embedding(decoder_input)
```

The embeddings of *decoder_input* and encoder's last hidden state are appended vertically. This appended/concatenated tensor is passed through an FC Layer which maps the 512 dimensions to 10 dimensions where 10 represents the max length of sentences in the dataset to calculate the weights/importance that should be given to each token of input while predicting a particular word. The weeights are then passed through softmax layer to normalize the weights. These weights reperesnt the attention that must be given to each token for a particular output. So, these weights are multiplied with the encoder's outputs. The result is a tensor representing the encoder tokens which must be focused on to generate English token.
```
attn_weight_layer = nn.Linear(256 * 2, 10).to(device)
input_to_lstm_layer = nn.Linear(256 * 2, 256).to(device)

attn_weights = attn_weight_layer(torch.cat((embedded[0], decoder_hidden[0]), 1))
attn_weights = F.softmax(attn_weights, dim = 1)
attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
```

The embedded input and focused state are concatenated and passed through FC layer to convert into 256 dimensions and then batch is added.

```
input_to_lstm = input_to_lstm_layer(torch.cat((embedded[0], attn_applied[0]), 1))
input_to_lstm = input_to_lstm.unsqueeze(0)
```

The above transformed input is then passed into LSTM layer for learning translation to English.

```
output, (decoder_hidden, decoder_cell_state) = dec_lstm(input_to_lstm, (decoder_hidden, decoder_cell_state))
```

The output of LSTM layer is passed to ReLU and Softmax for selecting one of the words from the English vocabulary formed from the dataset.

```
output = F.relu(output)
output = F.softmax(output_word_layer(output[0]), dim = 1)
```

The word having the maximum softmax value is chosen as the predicted word.

```
top_value, top_index = output.data.topk(1)
pred_op.append(top_index.item())
```

Below is the attention map for predicting 3rd English word.

```
decoder_attentions[i] = attn_weights.data.detach().cpu()
print(f"Target: {target_sentence.split(' ')[i]} \nPredicted: {output_lang.index2word[top_index.item()]}")#, attn_weights)
print("Attention Map")
showAttention(input_sentence, [output_lang.index2word[pred] for pred in pred_op], decoder_attentions)
```

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session11-AdvancedConceptsAnd4thHandsOn/assets/dec3.PNG)

#### 4) Feed Forward Step 4

Teacher forcing is used with 50% probability. If 1 comes, then teacher forcing is used and first token from target tensor [131] is passed else previously predicted word is passed as decoder's input.


```
teacher_forcing = 0.5
decoder_input = torch.tensor([[target_indices[i] if np.random.binomial(1,teacher_forcing) == 1 else SOS_token]], device=device)
```

The last hidden and cell states of the encoder is sent to the decoder.Since the last hidden and cell state are by product of encoder afer having seen the entire sentence, they capture the essence and context of the input sentence.

```
decoder_hidden = encoder_hidden
decoder_cell_state = encoder_cell_state
```

The numerical token of *decoder_input* must be converted into embedding of 256 dimension for standard contextual represention.

```
embedded = embedding(decoder_input)
```

The embeddings of *decoder_input* and encoder's last hidden state are appended vertically. This appended/concatenated tensor is passed through an FC Layer which maps the 512 dimensions to 10 dimensions where 10 represents the max length of sentences in the dataset to calculate the weights/importance that should be given to each token of input while predicting a particular word. The weeights are then passed through softmax layer to normalize the weights. These weights reperesnt the attention that must be given to each token for a particular output. So, these weights are multiplied with the encoder's outputs. The result is a tensor representing the encoder tokens which must be focused on to generate English token.
```
attn_weight_layer = nn.Linear(256 * 2, 10).to(device)
input_to_lstm_layer = nn.Linear(256 * 2, 256).to(device)

attn_weights = attn_weight_layer(torch.cat((embedded[0], decoder_hidden[0]), 1))
attn_weights = F.softmax(attn_weights, dim = 1)
attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
```

The embedded input and focused state are concatenated and passed through FC layer to convert into 256 dimensions and then batch is added.

```
input_to_lstm = input_to_lstm_layer(torch.cat((embedded[0], attn_applied[0]), 1))
input_to_lstm = input_to_lstm.unsqueeze(0)
```

The above transformed input is then passed into LSTM layer for learning translation to English.

```
output, (decoder_hidden, decoder_cell_state) = dec_lstm(input_to_lstm, (decoder_hidden, decoder_cell_state))
```

The output of LSTM layer is passed to ReLU and Softmax for selecting one of the words from the English vocabulary formed from the dataset.

```
output = F.relu(output)
output = F.softmax(output_word_layer(output[0]), dim = 1)
```

The word having the maximum softmax value is chosen as the predicted word.

```
top_value, top_index = output.data.topk(1)
pred_op.append(top_index.item())
```

Below is the attention map for predicting 4th English word.

```
decoder_attentions[i] = attn_weights.data.detach().cpu()
print(f"Target: {target_sentence.split(' ')[i]} \nPredicted: {output_lang.index2word[top_index.item()]}")#, attn_weights)
print("Attention Map")
showAttention(input_sentence, [output_lang.index2word[pred] for pred in pred_op], decoder_attentions)
```

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session11-AdvancedConceptsAnd4thHandsOn/assets/dec4.PNG)

#### 5) Feed Forward Step 5

Teacher forcing is used with 50% probability. If 1 comes, then teacher forcing is used and first token from target tensor [4] is passed else previously predicted word is passed as decoder's input.

```
teacher_forcing = 0.5
decoder_input = torch.tensor([[target_indices[i] if np.random.binomial(1,teacher_forcing) == 1 else SOS_token]], device=device)
```

The last hidden and cell states of the encoder is sent to the decoder.Since the last hidden and cell state are by product of encoder afer having seen the entire sentence, they capture the essence and context of the input sentence.

```
decoder_hidden = encoder_hidden
decoder_cell_state = encoder_cell_state
```

The numerical token of *decoder_input* must be converted into embedding of 256 dimension for standard contextual represention.

```
embedded = embedding(decoder_input)
```

The embeddings of *decoder_input* and encoder's last hidden state are appended vertically. This appended/concatenated tensor is passed through an FC Layer which maps the 512 dimensions to 10 dimensions where 10 represents the max length of sentences in the dataset to calculate the weights/importance that should be given to each token of input while predicting a particular word. The weeights are then passed through softmax layer to normalize the weights. These weights reperesnt the attention that must be given to each token for a particular output. So, these weights are multiplied with the encoder's outputs. The result is a tensor representing the encoder tokens which must be focused on to generate English token.
```
attn_weight_layer = nn.Linear(256 * 2, 10).to(device)
input_to_lstm_layer = nn.Linear(256 * 2, 256).to(device)

attn_weights = attn_weight_layer(torch.cat((embedded[0], decoder_hidden[0]), 1))
attn_weights = F.softmax(attn_weights, dim = 1)
attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
```

The embedded input and focused state are concatenated and passed through FC layer to convert into 256 dimensions and then batch is added.

```
input_to_lstm = input_to_lstm_layer(torch.cat((embedded[0], attn_applied[0]), 1))
input_to_lstm = input_to_lstm.unsqueeze(0)
```

The above transformed input is then passed into LSTM layer for learning translation to English.

```
output, (decoder_hidden, decoder_cell_state) = dec_lstm(input_to_lstm, (decoder_hidden, decoder_cell_state))
```

The output of LSTM layer is passed to ReLU and Softmax for selecting one of the words from the English vocabulary formed from the dataset.

```
output = F.relu(output)
output = F.softmax(output_word_layer(output[0]), dim = 1)
```

The word having the maximum softmax value is chosen as the predicted word.

```
top_value, top_index = output.data.topk(1)
pred_op.append(top_index.item())
```

Below is the attention map for predicting 5th English word.

```
decoder_attentions[i] = attn_weights.data.detach().cpu()
print(f"Target: {target_sentence.split(' ')[i]} \nPredicted: {output_lang.index2word[top_index.item()]}")#, attn_weights)
print("Attention Map")
showAttention(input_sentence, [output_lang.index2word[pred] for pred in pred_op], decoder_attentions)
```

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session11-AdvancedConceptsAnd4thHandsOn/assets/dec5.PNG)

#### 6) Feed Forward Step 6

Teacher forcing is used with 50% probability. If 1 comes, then teacher forcing is used and first token from target tensor [1] is passed else previously predicted word is passed as decoder's input.

```
teacher_forcing = 0.5
decoder_input = torch.tensor([[target_indices[i] if np.random.binomial(1,teacher_forcing) == 1 else SOS_token]], device=device)
```

The last hidden and cell states of the encoder is sent to the decoder.Since the last hidden and cell state are by product of encoder afer having seen the entire sentence, they capture the essence and context of the input sentence.

```
decoder_hidden = encoder_hidden
decoder_cell_state = encoder_cell_state
```

The numerical token of *decoder_input* must be converted into embedding of 256 dimension for standard contextual represention.

```
embedded = embedding(decoder_input)
```

The embeddings of *decoder_input* and encoder's last hidden state are appended vertically. This appended/concatenated tensor is passed through an FC Layer which maps the 512 dimensions to 10 dimensions where 10 represents the max length of sentences in the dataset to calculate the weights/importance that should be given to each token of input while predicting a particular word. The weeights are then passed through softmax layer to normalize the weights. These weights reperesnt the attention that must be given to each token for a particular output. So, these weights are multiplied with the encoder's outputs. The result is a tensor representing the encoder tokens which must be focused on to generate English token.
```
attn_weight_layer = nn.Linear(256 * 2, 10).to(device)
input_to_lstm_layer = nn.Linear(256 * 2, 256).to(device)

attn_weights = attn_weight_layer(torch.cat((embedded[0], decoder_hidden[0]), 1))
attn_weights = F.softmax(attn_weights, dim = 1)
attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
```

The embedded input and focused state are concatenated and passed through FC layer to convert into 256 dimensions and then batch is added.

```
input_to_lstm = input_to_lstm_layer(torch.cat((embedded[0], attn_applied[0]), 1))
input_to_lstm = input_to_lstm.unsqueeze(0)
```

The above transformed input is then passed into LSTM layer for learning translation to English.

```
output, (decoder_hidden, decoder_cell_state) = dec_lstm(input_to_lstm, (decoder_hidden, decoder_cell_state))
```

The output of LSTM layer is passed to ReLU and Softmax for selecting one of the words from the English vocabulary formed from the dataset.

```
output = F.relu(output)
output = F.softmax(output_word_layer(output[0]), dim = 1)
```

The word having the maximum softmax value is chosen as the predicted word.

```
top_value, top_index = output.data.topk(1)
pred_op.append(top_index.item())
```

Below is the attention map for predicting 6th English word.

```
decoder_attentions[i] = attn_weights.data.detach().cpu()
print(f"Target: {target_sentence.split(' ')[i]} \nPredicted: {output_lang.index2word[top_index.item()]}")#, attn_weights)
print("Attention Map")
showAttention(input_sentence, [output_lang.index2word[pred] for pred in pred_op], decoder_attentions)
```

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session11-AdvancedConceptsAnd4thHandsOn/assets/dec6.PNG)



**Input Sentence: 'elles sont tres grosses . \<EOS\>'**

**Target Sentence: 'they are very big . \<EOS\>'**

**Predicted Sentence: 'french impossible foolish foolish foolish handkerchiefs'**