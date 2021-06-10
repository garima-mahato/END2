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
