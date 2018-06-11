# ANN - Multilayer Perceptron Introduction

A multilayer perceptron (MLP) is a class of feedforward artificial neural network.
An MLP consists of at least three layers of nodes. Except for the input nodes, each 
node is a neuron that uses a nonlinear activation function. MLP utilizes a supervised learning
technique called backpropagation for training.Its multiple layers and non-linear activation 
distinguish MLP from a linear perceptron. It can distinguish data that is not linearly separable.

## Steps 

- Data Preprocessing
  - Read the dataset
  - Define features and labels
  - Encode the dependent Variable
  - Divide the dataset into training and testing
- Training
  - Tensorflow data structure for holding features and label
  - implement the model
  - train the model
  - Reduce MSE(mean square error) by (actual o/p - desired o/p)
  - Repeat training to decrease loss/cost function
- Prediction
  - Make predicton on the dataset

## Getting Started

In this example :
* Description: Predict metal or rock returns from sonar return data.
* Type: Binary Classification
* Dimensions: 208 instances, 61 attributes
* Inputs: Numeric
* Output: Categorical, 2 class labels
* UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Sonar,+Mines+vs.+Rocks)

## Prerequisites
* Tensorflow

## Notes

```ruby
from sklearn.cross_validation import train_test_split
```

is updated in new versions as :

```ruby
from sklearn.model_selection import train_test_split
```
