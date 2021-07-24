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

[Link to GitHub Code]()

[Link to Colab Code]()

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

#### Step 3: Add batch dimension into tensors of both sentences.

*Input Tensor: [[351, 349, 121, 1062, 5, 1]]*

*Target Tensor: [[221, 124, 303, 131, 4, 1]]*


## Encoder Feed-Forward