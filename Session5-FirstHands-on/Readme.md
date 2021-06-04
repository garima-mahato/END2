# Session 5 - First Hands On
---

1) Look at this code (Links to an external site.) above. It has additional details on "Back Translate", i.e. using Google translate to convert the sentences. It has "random_swap" function, as well as "random_delete". 

2) Use "Back Translate", "random_swap" and "random_delete" to augment the data you are training on

3) Download the StanfordSentimentAnalysis Dataset from this link  (Links to an external site.)(it might be troubling to download it, so force download on chrome). Use "datasetSentences.txt" and "sentiment_labels.txt" files from the zip you just downloaded as your dataset. This dataset contains just over 10,000 pieces of Stanford data from HTML files of Rotten Tomatoes. The sentiments are rated between 1 and 25, where one is the most negative and 25 is the most positive.

4) Train your model and achieve 60%+ validation/text accuracy. Upload your collab file on GitHub with readme that contains details about your assignment/word (minimum 250 words), training logs showing final validation accuracy, and outcomes for 10 example inputs from the test/validation data.

## Dataset

## EDA - Original Dataset

## Data Augmentation

## EDA - Augmented Dataset

## Model Building

## Training

## Testing

## Prediction

#### 10 Correctly Classified Texts

```
****************************************
***** Correctly Classified Text: *******
****************************************
1) Text: No one goes unindicted here , which is probably for the best .
   
   Target Sentiment: Neutral
   
   Predicted Sentiment: Neutral


2) Text: There 's ... tremendous energy from the cast , a sense of playfulness and excitement that seems appropriate .
   
   Target Sentiment: Positive
   
   Predicted Sentiment: Positive


3) Text: Here 's yet another studio horror franchise mucking up its storyline with glitches casual fans could correct in their sleep .
   
   Target Sentiment: Very Negative
   
   Predicted Sentiment: Very Negative


4) Text: While the stoically delivered hokum of Hart 's War is never fun , it 's still a worthy addition to the growing canon of post-Saving Private Ryan tributes to the greatest generation .
   
   Target Sentiment: Neutral
   
   Predicted Sentiment: Neutral


5) Text: Building slowly and subtly , the film , sporting a breezy spontaneity and realistically drawn characterizations , develops into a significant character study that is both moving and wise .
   
   Target Sentiment: Positive
   
   Predicted Sentiment: Positive


6) Text: Ultimately feels empty and unsatisfying , like swallowing a Communion wafer without the wine .
   
   Target Sentiment: Very Negative
   
   Predicted Sentiment: Very Negative


7) Text: Chilling , well-acted , and finely directed : David Jacobson 's Dahmer .
   
   Target Sentiment: Positive
   
   Predicted Sentiment: Positive


8) Text: Against all odds in heaven and hell , it creeped me out just fine .
   
   Target Sentiment: Positive
   
   Predicted Sentiment: Positive


9) Text: A compelling Spanish film about the withering effects of jealousy in the life of a young monarch whose sexual passion for her husband becomes an obsession .
   
   Target Sentiment: Positive
   
   Predicted Sentiment: Positive


10) Text: It 's fascinating to see how Bettany and McDowell play off each other .
   
   Target Sentiment: Positive
   
   Predicted Sentiment: Positive

```
