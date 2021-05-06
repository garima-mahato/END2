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

This y is the output of a neuron. Thus, we get a neuron.


#### 2) What is the use of the learning rate?



#### 3) How are weights initialized?

#### 4) What is "loss" in a neural network?

#### 5) What is the "chain rule" in gradient flow?
