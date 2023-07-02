



## Introduction to Neural Network

A neural network has it’s basic building block that is called as neurons. Several neurons
interacts with next and previous layer neurons for their state computations. These neurons
are highly interconnected with a weight associated with each connection. The architecture
of a multi-layer perceptron consists of input layers, hidden layers and output layer(s).

Types of Layers in Neural Network

1. Input Layer This layer contains the input data that is feed to it. There is basically
no computation done on the input layer. It contains number of nodes equal to the
number of features.
2. Hidden Layers These are the layers where actual computation is done via change
in the weights. In MLP there must be at least one hidden layer.
3. Output Layers This layers give the output which we generate through the forward
propagation. This is matched with the actual output and we then we computed how
much changes we need to make in our neural network to make it predict things with
less loss.


## Activation Functions
An activation functions takes a value as an input and produced another value as it’s
output. There are various activation functions that one can use. The activation function
which are used in this projects are Sigmod, Relu and Tanh.

## Back Propagation Algorithm

Implemented 
- Stochastic Gradient Descent
- Batch Gradient Descent (BGD)
- Mini Batch Gradient Descent

## Achieved 

Dynamic deep neural network library in C
- Implemented a module capable of generating arbitrary dense neural networks with several tunable parameters
- Achieved MSE of 42 in house price prediction over Boston house prices dataset and a classification accuracy of 98%
over breast cancer Winsconsin dataset


## Software Requirements
- GNU Compiler Collections (GCC).
- Git Version Control System.

## References
1. http://deeplearning.stanford.edu/tutorial/supervised/MultiLayerNeuralNetworks/
2. https://towardsdatascience.com/how-does-back-propagation-in-artificial-neural-networks-work-c7cad873ea7
3. https://towardsdatascience.com/machine-learning-for-beginners-an-introduction-to-neural-networks-d49f22d238f9
4. https://builtin.com/data-science/gradient-descent
5. https://towardsdatascience.com/stochastic-gradient-descent-clearly-explained-53d239905d31
6. https://towardsdatascience.com/the-vanishing-gradient-problem-69bf08b15484