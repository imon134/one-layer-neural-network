# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 14:36:27 2022

@author: imon
"""
import numpy as np


def sigmoid(z):
    """Does the sigmoid activation function"""
    s = 1/(1 + np.exp(-z))
    return s


def layer_sizes(X, Y):
    """Gets the sizes for each layer in the neural network"""
    n_x = X.shape[0]  # size of the input layer
    n_y = Y.shape[0]  # size of the output layer
    n_h = 4      # size of the hidden layer or no of hidden units in that layer
    return (n_x, n_h, n_y)


def initialize_parameters(n_x, n_h, n_y):
    """Initializes weights and biases in the neural network"""
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros(shape=(n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros(shape=(n_y, 1))
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}  # stores the weights and biases in a dictionary
    return parameters       # to be returned by this function


def forward_propagation(X, parameters):
    """performs forward propagation  algorithm using the weights and biases"""
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    Z1 = np.dot(W1, X) + b1
    #    below is the activation unit for the hidden layer
    A1 = np.tanh(Z1)       # uses tanh activation function for the hidden layer
    Z2 = np.dot(W2, A1) + b2
    #     below is the activation unit for the output layer
    A2 = sigmoid(Z2)  # uses sigmoid for 0 or 1 classification

    assert(A2.shape == (1, X.shape[1]))

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache


def compute_cost(A2, Y):
    """computes the cost function"""
    m = Y.shape[1]
    logprobs = -1 * (np.multiply(np.log(A2), Y) + np.multiply(
        (1 - Y), np.log(1 - A2)))
    cost = np.sum(logprobs) / m
    cost = float(np.squeeze(cost))  # makes sure cost is the dimension we want

    return cost


def backward_propagation(parameters, cache, X, Y):
    """Performs the back propagation algorithm"""
    m = Y.shape[1]

    W2 = parameters['W2']

    A1 = cache['A1']
    A2 = cache['A2']
    # YOUR CODE ENDS HERE

    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T)/m
    db2 = np.sum(dZ2, axis=1, keepdims=True)/m
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = np.dot(dZ1, X.T)/m
    db1 = np.sum(dZ1, axis=1, keepdims=True)/m

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads


def update_parameters(parameters, grads, learning_rate=0.3):
    """reduces or increases the weights and biases to predict better """
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def nn_model(X, Y, n_h, num_iterations=10000, print_cost=True):
    """The model to be called to use the one layer neural network"""
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    parameters = initialize_parameters(n_x, n_h, n_y)

    for i in range(0, num_iterations):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads)

        if print_cost and i % 1 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    return parameters


def predict(parameters, X):

    A2, cache = A2, cache = forward_propagation(X, parameters)
    predictions = np.round(A2)

    return predictions
