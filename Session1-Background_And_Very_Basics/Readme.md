# Assignment

## Part 1

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session1-Background_And_Very_Basics/assets/assignment_part1.PNG)

## Part 2

### 1) What is a neural network neuron?

A neural network mimics the neurological system of human body. The basic building block of neural network is a neuron. Each neuron takes some input, performs some computation and generates the output. 

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session1-Background_And_Very_Basics/assets/neuron3.jpg)

This is how a neuron looks like. Let's dissect it and see what happens inside.

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session1-Background_And_Very_Basics/assets/neuron4.png)

So, there are 3 operations going on inside a single neuron. A lot is going on inside a tiny circle. Let's unpack it.

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session1-Background_And_Very_Basics/assets/neuron2.PNG)

Each input(*x<sub>i</sub>*) to the neuron is assigned a weight(*w<sub>i</sub>*), where subscript *i* denotes the input index. Below computations happen inside the neuron:

> i) Multiplication of input and weights to give weighted inputs: **a<sub>i</sub> = x<sub>i</sub> x w<sub>i</sub>** . 

> ii) Summation of weighted inputs. Sometimes bias (*b*) is also added. **z = a<sub>1</sub> + a<sub>2</sub> + ... + + a<sub>i</sub> + b** 

> iii) The summation result *z* is passed through an activation function f to give the result y: **y = f(z)**

This y is the output of a neuron. Thus, a neuron can be thought of as a function with weights and biases as parameters.


### 2) What is the use of the learning rate?

Suppose you are walking on a hill and you need to reach the bottom most of the hill. You are unable to see the bottom most point because of very abrupt rise and fall in the hilly region. What would you do? You would look in the directions from where you are standing to see which direction will take you downwards. Then you will move in that direction and you will take a small step in that direction so that you do not miss the bottom most point. If it takes you upwards, you would take a larger step so that you quickly overcome that area.

**How much step size** to take is determined by **learning rate**. Let's try to understand with respect to neuron. A neural network is combination of neurons, i.e., functions. In other terms, its a function. In order to find the best function which explains a particular dataset, we need to find the parameters,i.e. weights and biases, for which output is as close as possible to the expected output,i.e. error is minimum. For each set of parameters i.e. weights and biases, the error/loss would be different. Thus, loss is a function of parameters. When we plot a graph of parameters versus loss, we get the below graph.

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session1-Background_And_Very_Basics/assets/lr2.png)

This seems similar to the hill you were walking. Let's draw parallel from your hilly experience. The bottom most point in the graph is where loss is minimum. The red arrows denote those points. As the learning algorithm is unaware of these minimas, it will try to get a sense of the direction by finding slope of tangent. It will make steps down the loss function in the direction with the steepest descent. The size of each step is determined by the parameter called as the learning rate, α. For example, the distance between each 'star' in the graph above represents a step determined by α. A smaller α would result in a smaller step and a larger α results in a larger step. 

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session1-Background_And_Very_Basics/assets/lr1.jpg)

If the learning rate is very big, the loss will keep on increasing. If it is very small, then we will ake forever to reach the minimum loss. Learning rate gives us control as to how much change we want in step.

#### 3) How are weights initialized?

Proper initial values must be given to the network otherwise it will lead to problems like vanishing or exploding gradients. There are different techniques to initialize weights. You can visualize and play with it in [deeplearning.ai: Weight Initialization](https://www.deeplearning.ai/ai-notes/initialization/). 

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session1-Background_And_Very_Basics/assets/wi1.gif)

The different techniques are:

**i) Zero/Ones/Constant Initialization**

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session1-Background_And_Very_Basics/assets/wi2.png)

In this technique, all weights are initialized with zero/one/constant value. The derivative with respect to loss function becomes same for all of the weights which in turn updates the weights to the same value in each subsequent iteration. Thus, hidden units become symmetric and it behaves like a linear model.

To observe this, we'll take an example of a neural network with three hidden layers with ReLU activation function in hidden layers and sigmoid for the output layer.
Using the above neural network on the dataset “make circles” from sklearn.datasets and zero weight initialization, the result obtained as the following :
for 15000 iterations, loss = 0.6931471805599453, accuracy = 50 %

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session1-Background_And_Very_Basics/assets/wi3.png)

**ii) Random Initialization**

In this, random weights are assigned to each neuron connection. It is based on Break Symmetry in which:

> i) If two hidden units have the same inputs and same activation function, then they must have different initial parameters

> ii) It’s desirable to initialize each unit differently to compute a different function

If we randomly, initialize weights without knowing the underlying distribution, 2 issues might occur:

> i) If the weights are initialized with too small random values, then the gradient diminishes as it propagates to the deeper layers.

> ii) If the weights are initialized with too large values, then the gradient increases(explodes) as it propagates to the deeper layers.

To observe this, we'll take the above example of a neural network with three hidden layers with ReLU activation function in hidden layers and sigmoid for the output layer.
Using the above neural network on the dataset “make circles” from sklearn.datasets and zero weight initialization, the result obtained as the following :
for 15000 iterations, loss = 0.38278397192120406, accuracy = 86 %

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session1-Background_And_Very_Basics/assets/wi4.png)

So, while using random weights intialization, we use normal distribution.

**The normal random weight initialization does not work well for very deep network, especially with non-linear activation functions like ReLU. So, Xaxier and He initialization consider into account both size of the network and activation function.**

**iii) He Normal Initialization**



**iv) Xavier/Glorot Initialization**

In this, network weights are initialized by drawing samples from truncated normal distribution with:
```mean = 0, and 
standard deviation = sqrt(1/fan_in), where fan_in = number of input units to weight```


### 4) What is "loss" in a neural network?

### 5) What is the "chain rule" in gradient flow?
