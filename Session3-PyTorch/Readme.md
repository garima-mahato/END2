# Identifying MNIST image and generating sum

[GitHub Link to jupyter notebook](https://github.com/garima-mahato/END2/blob/main/Session3-PyTorch/END2_Session3_PytorchAssignment.ipynb)

[Colab Link to jupyter notebook](https://githubtocolab.com/garima-mahato/END2/blob/main/Session3-PyTorch/END2_Session3_PytorchAssignment.ipynb)


## Dataset

#### Data Representation

**Inputs**:

> 1) MNIST Image of dimension 28x28x1
> 2) Random number between 0 and 9. It is represented as one-hot encoded vector of dimension 1x10.

**Outputs**:

> 1) Number as shown in MNIST Image input. It is represented as one-hot encoded vector of dimension 1x10.
> 2) Sum of MNIST number and random number input. It is represented as one-hot encoded vector of dimension 1x19.


#### Data Generation Strategy

1) Read MNIST data from torchvision datasets. It gives image and the number corresponding to it.

```
mnist_data = torchvision.datasets.MNIST(root="data", train=train, download=True, transform=None)
img, out_number = mnist_data[idx]
```

2) For each MNIST image, a random number between 0 and 9 is generated 

```
random_input = randrange(10)
```

3) Calculate sum by adding MNIST number from step 1 and random number from step 2.

```
sum = number + random_input
```

Thus, dataset will contain (img, random_input) as input and (out_number, sum) as output.

**This dataset is further divided into training, test and evaluation datasets.** The **60,000 images of training MNIST dataset** is used to create training and test dataset. **80% of 60,000 data** form *training dataset*. **20% of 60,000 data** form *test dataset*. The **10,000 images of test MNIST dataset** is used to create *evaluation dataset*.

Training Dataset size: 48000
Validation Dataset size: 12000
Evaluation Dataset size: 10000


**Training Dataset**

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session3-PyTorch/assets/train_data.PNG)


**Test Dataset**

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session3-PyTorch/assets/test_data.PNG)

**Evaluation Dataset**

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session3-PyTorch/assets/eval_data.PNG)

## Model

**IdentityAdderModel** - a neural network that can:
I) take 2 inputs:

  1) an image from MNIST dataset, and
  
  2) a random number between 0 and 9
  
II) and gives two outputs:

  1) the "number" that was represented by the MNIST image, and
  2) the "sum" of this number with the random number that was generated and sent as the input to the network


#### Model Structure:

The model consist of 3 parts:

**a) CNN network to identify number from MNIST image:** It consist of 2 convolution blocks. Each convolution block consist of 2 convolution sequentials and Maxpooling layer. Each convolution sequential consist of convolution layer followed by ReLU, batch normalization and dropout layers. After the 2 convolution blocks, another convolution lock is used. By now, a good enough receptive field of 32x32 is reached. So, then a global average pooling is applied. The result of average pooling is combined with fully connected layers to get an embedding of the number identified which has size 20.

**b) Fully connected network to learn both number and addition of numbers:** The 20 size embedding from above CNN and one hot encoding of random number input with size 10 are concatenated and then passed through 2 fully connected layers.

**c) Output layers:** There are 2 output layers. Each of them is a fully connected layer with 10 neurons for number prediction but 19(as max sum result=18) neurons for sum prediction.



![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session3-PyTorch/assets/onnx_identity_adder_model.onnx.png)

#### Summary of model used:

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session3-PyTorch/assets/model_summary.png)

## Loss Function used is:

```
loss = nn.CrossEntropyLoss(out1, num) + nn.CrossEntropyLoss(out2, sum)
```

Since predicting both MNIST number and sum are classification problems with 10 and 19 classes respectively.

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session3-PyTorch/assets/loss_fn.jpeg)


## Training and Testing of model

**Maximum Training Accuracy: 97.84 %**
**Test Accuracy: 98.84 %**

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session3-PyTorch/assets/lr_training_log.PNG)

#### Accuracy and Loss of model during training and testing

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session3-PyTorch/assets/lr_train_test_acc_loss.PNG)


#### Training vs Testing Accuracy

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session3-PyTorch/assets/lr_train_test_acc_graph.PNG)



## Model Evaluation

Model is evaluated on evaluation dataset. Accuracy is in percentage.

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session3-PyTorch/assets/lr_model_eval.PNG)

## Model Prediction

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session3-PyTorch/assets/lr_model_pred.PNG)
