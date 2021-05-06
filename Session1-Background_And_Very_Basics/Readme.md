# Assignment

## Part 1

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session1-Background_And_Very_Basics/assets/assignment_part1.PNG)

## Part 2

#### 1) What is a neural network neuron?

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


#### 2) What is the use of the learning rate?

Suppose you are walking on a hill and you need to reach the bottom most of the hill. You are unable to see the bottom most point because of very abrupt rise and fall in the hilly region. What would you do? You would look in the directions from where you are standing to see which direction will take you downwards. Then you will move in that direction and you will take a small step in that direction so that you do not miss the bottom most point. If it takes you upwards, you would take a larger step so that you quickly overcome that area.

**How much step size** to take is determined by **learning rate**. Let's try to understand with respect to neuron. A neural network is combination of neurons, i.e., functions. In other terms, its a function. In order to find the best function which explains a particular dataset, we need to find the parameters,i.e. weights and biases, for which output is as close as possible to the expected output,i.e. error is minimum. For each set of parameters i.e. weights and biases, the error/loss would be different. Thus, loss is a function of parameters. When we plot a graph of parameters versus loss, we get the below graph.

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session1-Background_And_Very_Basics/assets/lr2.png)

This seems similar to the hill you were walking. Let's draw parallel from your hilly experience. The bottom most point in the graph is where loss is minimum. The red arrows denote those points. As the learning algorithm is unaware of these minimas, it will try to get a sense of the direction by finding slope of tangent. It will make steps down the loss function in the direction with the steepest descent. The size of each step is determined by the parameter called as the learning rate, α. For example, the distance between each 'star' in the graph above represents a step determined by α. A smaller α would result in a smaller step and a larger α results in a larger step. 

![](https://raw.githubusercontent.com/garima-mahato/END2/main/Session1-Background_And_Very_Basics/assets/lr1.jpg)

#### 3) How are weights initialized?

#### 4) What is "loss" in a neural network?

#### 5) What is the "chain rule" in gradient flow?
