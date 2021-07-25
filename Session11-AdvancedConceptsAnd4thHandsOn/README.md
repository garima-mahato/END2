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
```embedded_input = embedding(input_tensor[i].view(-1, 1))```

Output of embedding layer is passed into LSTM Layer

```
output, (encoder_hidden, encoder_cell_state) = lstm(embedded_input, (encoder_hidden, encoder_cell_state))
encoder_outputs[i] += output[0,0]
```

# View of Encoder Ouput
plot_matrix(encoder_outputs)

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session11-AdvancedConceptsAnd4thHandsOn/assets/enc1.PNG)

#### 2) Feed Forward Step 2

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session11-AdvancedConceptsAnd4thHandsOn/assets/enc2.PNG)

#### 3) Feed Forward Step 3

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session11-AdvancedConceptsAnd4thHandsOn/assets/enc3.PNG)

#### 4) Feed Forward Step 4

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session11-AdvancedConceptsAnd4thHandsOn/assets/enc4.PNG)

#### 5) Feed Forward Step 5

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session11-AdvancedConceptsAnd4thHandsOn/assets/enc5.PNG)

#### 6) Feed Forward Step 6

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session11-AdvancedConceptsAnd4thHandsOn/assets/enc6.PNG)


## Decoder Feed-Forward Steps

#### 1) Feed Forward Step 1

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session11-AdvancedConceptsAnd4thHandsOn/assets/dec1.PNG)

#### 2) Feed Forward Step 2

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session11-AdvancedConceptsAnd4thHandsOn/assets/dec2.PNG)

#### 3) Feed Forward Step 3

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session11-AdvancedConceptsAnd4thHandsOn/assets/dec3.PNG)

#### 4) Feed Forward Step 4

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session11-AdvancedConceptsAnd4thHandsOn/assets/dec4.PNG)

#### 5) Feed Forward Step 5

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session11-AdvancedConceptsAnd4thHandsOn/assets/dec5.PNG)

#### 6) Feed Forward Step 6

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session11-AdvancedConceptsAnd4thHandsOn/assets/dec6.PNG)



**Input Sentence: 'elles sont tres grosses . \<EOS\>'**

**Target Sentence: 'they are very big . \<EOS\>'**

**Predicted Sentence: 'french impossible foolish foolish foolish handkerchiefs'**