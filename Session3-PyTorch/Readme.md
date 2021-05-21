# Identifying MNIST image and generating sum

[Link to Colab file](https://github.com/garima-mahato/END2/blob/main/Session3-PyTorch/END2_Session3_PytorchAssignment.ipynb)


## Dataset

#### Data Representation

**Inputs**

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session3-PyTorch/assets/train_data.PNG)

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session3-PyTorch/assets/test_data.PNG)

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session3-PyTorch/assets/eval_data.PNG)

## Model

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session3-PyTorch/assets/onnx_identity_adder_model.onnx.png)

**IdentityAdderModel** - a neural network that can:
I) take 2 inputs:

  1) an image from MNIST dataset, and
  
  2) a random number between 0 and 9
  
II) and gives two outputs:

  1) the "number" that was represented by the MNIST image, and
  2) the "sum" of this number with the random number that was generated and sent as the input to the network

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session3-PyTorch/assets/model_summary.png)

## Experimentation

