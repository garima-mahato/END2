# Assignment
---

Refer to this [Repo](https://github.com/bentrevett/pytorch-seq2seq). 

1) You are going to refactor this repo in the next 3 sessions. In the current assignment, change the 2 and 3 (optional 4, 500 additional points such that:

> 1) is uses none of the legacy stuff

> 2) It MUST use Multi30k dataset from torchtext 

> 3) uses yield_token, and other code that we wrote

2) Once done, proceed to answer questions in the Assignment-Submission Page. 

# Solution
---

## 2 - Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation 

[GitHub Link]()

[Colab Link]()

Now we have the basic workflow covered, this tutorial will focus on improving our results. Building on our knowledge of PyTorch and torchtext gained from the previous tutorial, we'll cover a second second model, which helps with the information compression problem faced by encoder-decoder models. This model will be based off an implementation of Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation, which uses GRUs.

## 3 - Neural Machine Translation by Jointly Learning to Align and Translate 

[GitHub Link]()

[Colab Link]()

Next, we learn about attention by implementing Neural Machine Translation by Jointly Learning to Align and Translate. This further allievates the information compression problem by allowing the decoder to "look back" at the input sentence by creating context vectors that are weighted sums of the encoder hidden states. The weights for this weighted sum are calculated via an attention mechanism, where the decoder learns to pay attention to the most relevant words in the input sentence.

## 4 - Packed Padded Sequences, Masking, Inference and BLEU

[GitHub Link]()

[Colab Link]()

In this notebook, we will improve the previous model architecture by adding packed padded sequences and masking. These are two methods commonly used in NLP. Packed padded sequences allow us to only process the non-padded elements of our input sentence with our RNN. Masking is used to force the model to ignore certain elements we do not want it to look at, such as attention over padded elements. Together, these give us a small performance boost. We also cover a very basic way of using the model for inference, allowing us to get translations for any sentence we want to give to the model and how we can view the attention values over the source sequence for those translations. Finally, we show how to calculate the BLEU metric from our translations.

