# Session 4 - RNN are dead, and their renewed relevance
---

## Assignment: 

1) Refer to the file we wrote in the [class](https://colab.research.google.com/drive/1-xwX32O0WYOqcCROJnnJiSdzScPCudAM?usp=sharing): Rewrite this code, but this time remove RNN and add LSTM. 

2) Refer to this [file](https://colab.research.google.com/drive/12Pciev6dvYBJ7KxwSHruG-XMwcoj0SfJ). 
> 1) The questions this time are already mentioned in the file. Take as much time as you want (but less than 7 days), to solve the file. Once you are done, then write your solutions in the quiz. 
> 2) Please note that the Session 4 Assignment Solution will time out after 15 minutes, as you just have to copy-paste your answers. 

## Solution:

### **1) Solution for LSTM Rewrite of RNN:**

| Google Colab | GitHub |
|---|---|
| [END2_Session4_AssignmentSolution1.ipynb](https://colab.research.google.com/drive/10UkWvdJ3IMPeMwDLz5ADy-_FDJb7W8Ej?usp=sharing) | [END2_Session4_AssignmentSolution1.ipynb](https://github.com/garima-mahato/END2/blob/main/Session4-RNN_are_dead%2C_and_their_renewed_relevance/END2_Session4_AssignmentSolution1.ipynb) |

### **2) Solution for LSTM Questions:**

| Google Colab | GitHub |
|---|---|
| [END2_Session4_AssignmentSolution2.ipynb](https://githubtocolab.com/garima-mahato/END2/blob/main/Session4-RNN_are_dead%2C_and_their_renewed_relevance/END2_Session4_AssignmentSolution2.ipynb) | [END2_Session4_AssignmentSolution2.ipynb](https://github.com/garima-mahato/END2/blob/main/Session4-RNN_are_dead%2C_and_their_renewed_relevance/END2_Session4_AssignmentSolution2.ipynb) |


**Quiz Question 1:** 
What is the value of sigmoid(0) calculated from your code? (Answer up to 1 decimal point, e.g. 4.2 and NOT 4.29999999, no rounding off).

**Ans 1: 0.5**

**Quiz Question 2:** 
What is the value of dsigmoid(sigmoid(0)) calculated from your code?? (Answer up to 2 decimal point, e.g. 4.29 and NOT 4.29999999, no rounding off).

**Ans 2: 0.25**

**Quiz Question 3:**
What is the value of tanh(dsigmoid(sigmoid(0))) calculated from your code?? (Answer up to 5 decimal point, e.g. 4.29999 and NOT 4.29999999, no rounding off).

**Ans 3: 0.24491**

**Quiz Question 4: ** 
What is the value of dtanh(tanh(dsigmoid(sigmoid(0)))) calculated from your code?? (Answer up to 5 decimal point, e.g. 4.29999 and NOT 4.29999999, no rounding off).

**Ans 4: 0.94001**

**Quiz Question 5: **
In the class definition below, what should be size_a, size_b, and size_c? ONLY use the variables defined above.

**Ans:**
```
size_a = Hidden_Layer_size

size_b = z_size

size_c = X_size
```

**Quiz Question 6: **
What is the output of 'print(len(forward(np.zeros((X_size, 1)), np.zeros((Hidden_Layer_size, 1)), np.zeros((Hidden_Layer_size, 1)), parameters)))'?

**Ans: 9**

**Quiz Question 7: **
Assuming you have fixed the forward function, run this command: z, f, i, C_bar, C, o, h, v, y = forward(np.zeros((X_size, 1)), np.zeros((Hidden_Layer_size, 1)), np.zeros((Hidden_Layer_size, 1)))

Now, find these values:

1) print(z.shape)

2) print(np.sum(z))

3) print(np.sum(f))

**Ans:**

1) (85, 1)

2) 3.0

3) 5.0

**Quiz Question 7:**
Run the above code for 50000 iterations making sure that you have 100 hidden layers and time_steps is 40. What is the loss value you're seeing?

**Ans:**

After running once for 50000 iterations: loss 5.822303

After running twice for 50000 iterations: loss 1.940444
