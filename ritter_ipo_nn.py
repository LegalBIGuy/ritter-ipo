# With acknowledgement to deeplearning.ai
# Coursera courses by Andrew Ng
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
import h5py
import scipy
import matplotlib.pyplot as plt


class NeuralNetwork:

    def __init__(self, layer_dims, learning_rate = 0.0075):
        self.layer_dims = layer_dims
        self.L = len(self.layer_dims) - 1
        self.model_caches = []
        self.learning_rate = learning_rate
        np.random.seed(42)

    def sigmoid(self, Z):
        return 1/(1+np.exp(-Z))

    def sigmoid_backward(self, dA, Z):
        s = 1/(1+np.exp(-Z))
        dZ = dA * s * (1-s)
        return dZ

    def relu(self, Z):
        return np.maximum(0,Z)

    def relu_backward(self, dA, Z):
        dZ = np.array(dA, copy=True)
        # When z <= 0, set dz to 0 as well. 
        dZ[Z <= 0] = 0
        return dZ

    def initialize_parameters_he(self):
        parameters = {}
        for l in range(1, self.L + 1):
            parameters['W' + str(l)] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * np.sqrt(2/self.layer_dims[l-1])
            parameters['b' + str(l)] = np.zeros((self.layer_dims[l], 1))

        return parameters

    def linear_activation_forward(self, A_prev, W, b, activation):
        linear_cache = (A_prev, W, b)
        Z = np.dot(W,A_prev) + b

        if activation == "sigmoid":
            A = self.sigmoid(Z)
        elif activation == "relu":
            A = self.relu(Z)

        activation_cache = Z
        cache = (linear_cache, activation_cache)
        return A, cache

    def model_forward(self, X):
        A = X

        # Implement [LINEAR -> RELU]*(L-1)
        for l in range(1, self.L):
            A_prev = A 
            A, cache = self.linear_activation_forward(A_prev, self.parameters['W' + str(l)], self.parameters['b' + str(l)], 'relu')
            self.model_caches.append(cache)

        # Implement LINEAR -> SIGMOID
        AL, cache = self.linear_activation_forward(A, self.parameters['W' + str(self.L)], self.parameters['b' + str(self.L)], 'sigmoid')
        self.model_caches.append(cache)
     
        return AL

    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        cost = (-1/m)*(np.sum(np.multiply(np.log(AL),Y) + np.multiply((1-Y),np.log(1-AL))))
        cost = np.squeeze(cost)
        return cost

    def linear_backward(self, dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = 1 / m * (np.dot(dZ, A_prev.T))
        db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = (np.dot(W.T, dZ))

        return dA_prev, dW, db

    def linear_activation_backward(self, dA, cache, activation):
        linear_cache, activation_cache = cache
        
        if activation == "relu":
            dZ = self.relu_backward(dA, activation_cache)

        elif activation == "sigmoid":
            dZ = self.sigmoid_backward(dA, activation_cache)

        dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        return dA_prev, dW, db

    def model_backward(self, AL, Y):
        grads = {}
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)
        
        # Initializing the backpropagation
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        
        # (SIGMOID -> LINEAR) gradients. 
        current_cache = self.model_caches[self.L-1]
        grads["dA" + str(self.L)], grads["dW" + str(self.L)], grads["db" + str(self.L)] = self.linear_activation_backward(dAL, current_cache, activation = "sigmoid")
        
        for l in reversed(range(self.L-1)):
            # (RELU -> LINEAR) gradients.
            current_cache = self.model_caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads["dA" + str(l + 2)], current_cache, activation = "relu")
            grads["dA" + str(l + 1)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads

    def update_parameters(self, grads):

        for l in range(self.L):
            self.parameters["W" + str(l+1)] = self.parameters["W" + str(l+1)] - self.learning_rate * grads["dW" + str(l+1)]
            self.parameters["b" + str(l+1)] = self.parameters["b" + str(l+1)] - self.learning_rate * grads["db" + str(l+1)]

    def initialize_momentum(self):

        self.momentum_v = {}
        for l in range(self.L):
            self.momentum_v["dW" + str(l + 1)] = np.zeros_like(self.parameters["W" + str(l+1)])
            self.momentum_v["db" + str(l + 1)] = np.zeros_like(self.parameters["b" + str(l+1)])

    def update_parameters_momentum(self, grads, beta = 0.9):

        for l in range(self.L):
            self.momentum_v["dW" + str(l + 1)] = beta * self.momentum_v["dW" + str(l + 1)] + (1 - beta) * grads['dW' + str(l + 1)]
            self.momentum_v["db" + str(l + 1)] = beta * self.momentum_v["db" + str(l + 1)] + (1 - beta) * grads['db' + str(l + 1)]
            self.parameters["W" + str(l + 1)] = self.parameters["W" + str(l + 1)] - self.learning_rate * self.momentum_v["dW" + str(l + 1)]
            self.parameters["b" + str(l + 1)] = self.parameters["b" + str(l + 1)] - self.learning_rate * self.momentum_v["db" + str(l + 1)]

    def train(self, X, Y, use_momentum, num_iterations = 1000, print_cost_n = 100):
        
        # initialize parameters
        self.parameters = self.initialize_parameters_he()
        
        self.costs = []

        if use_momentum:
            self.initialize_momentum()

        for i in range(0, num_iterations):
            AL = self.model_forward(X)
            cost = self.compute_cost(AL, Y)
            grads = self.model_backward(AL, Y)
            if use_momentum:
                self.update_parameters_momentum(grads)
            else:
                self.update_parameters(grads)
            self.model_caches = []

            # Print the cost every print_cost_n training example
            if i % print_cost_n == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
                self.costs.append(cost)

    def plot_costs(self):        

        plt.plot(np.squeeze(self.costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(self.learning_rate))
        plt.show()

    def predict(self, X, y):
        m = X.shape[1]
        n = len(self.parameters) // 2
        p = np.zeros((1,m))

        probs = self.model_forward(X)

        for i in range(0, probs.shape[1]):
            if probs[0,i] > 0.5:
                p[0,i] = 1
            else:
                p[0,i] = 0

        print("Accuracy: "  + str(np.sum((p == y)/m)))   
        return p



if __name__ == "__main__":

    df = pd.read_csv('IPO2609FeatureEngineering.csv')
    ipo_train, ipo_test = train_test_split(df, random_state=42)

    # Columns for training (X), remove label (y)
    features = df.columns.drop(["underpriced"])

    train_x = ipo_train[features.values.tolist()].T.astype(float) 
    train_x = normalize(train_x)
    train_y = np.ravel(ipo_train["underpriced"])
    train_y = train_y.reshape(1, train_y.shape[0])

    test_x = ipo_test[features.values.tolist()].T.astype(float) 
    test_x = normalize(test_x)
    test_y = np.ravel(ipo_test["underpriced"])
    test_y = test_y.reshape(1, test_y.shape[0])

    ipo_layer_dims = [train_x.shape[0], 15, 7, 5, 1]

    # Train/Predict with Gradient Descent
    nn = NeuralNetwork(ipo_layer_dims, .05)
    nn.train(train_x, train_y, False, 2800)
    nn.plot_costs()
    p = nn.predict(test_x, test_y)
        
    # Train / Predict with Momentum
    nn = NeuralNetwork(ipo_layer_dims, 0.03)
    nn.train(train_x, train_y, True, 7500)
    nn.plot_costs()
    p = nn.predict(test_x, test_y)


