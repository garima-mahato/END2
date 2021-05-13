# Neural Network Training in Excel
---

[Link to training excel sheet](https://github.com/garima-mahato/END2/blob/main/Session%202-BackProp_Embeddings_and_Language_Models/END2_S2_Backpropagation.xlsx)

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session%202-BackProp_Embeddings_and_Language_Models/assets/training.PNG)

## Major Steps in NN Training

Suppose we have the below network:

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session%202-BackProp_Embeddings_and_Language_Models/assets/nn.PNG)


#### 1) Initialization of Neural Network
Randomly initializing weights [w1,....,w8] and Learning Rate lr.

#### 2) Forward Propagation

For the above network

'''
h1 = w1*i1 + w2*i2				
h2 = w3*i1 + w4*i2				
a_h1 = σ(h1) = 1/(1+exp(-h1))				
a_h2 = σ(h2) = 1/(1+exp(-h2))				
o1 = w5*a_h1 + w6*a_h2				
o2 = w7*a_h1 + w8*a_h2				
a_o1 = σ(o1) = 1/(1+exp(-o1))				
a_o2 = σ(o2) = 1/(1+exp(-o2))				
E1 = ½*(t1-a_o1)²				
E2 = ½*(t2-a_o2)²				
E_total = E1 + E2		
'''

#### 3) Backpropagation


# Error Graphs for various learning rates

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session%202-BackProp_Embeddings_and_Language_Models/assets/err_lr.PNG)
