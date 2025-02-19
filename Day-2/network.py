import random
import numpy as np 

class Netword(object):
    def __init__(self, sizes):
        # 'sizes' -> number of neurons in a layer 
        # biases and weights are initialized randomly using a Gaussian distribution with mean 0
        # and variannce 1 
        self.num_layers = len(sizes)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a 
    ## Backprop Algo 
    def backprop(self, x, y):
        """Return a tuple `(nabla_b, nabla_w)` representing the gradient for the cost function 
        C_x. 
        'nable_b' & 'nabla_w' being the layer-by-layer lists of numpy arrays, similar to 'self.biases', 'self.weights'"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward 
        activation = x 
        activation = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b 
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass 
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta 
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp 
            nabla_b[-l] = delta 
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

def sigmoid(z):
    return 1.0 / (1.0+np.exp(-z))

def sigmoid_prime(z):
    ## Derivative of sigmoid 
    return sigmoid(z)*(1-sigmoid(z))
