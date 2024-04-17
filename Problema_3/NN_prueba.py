import numpy as np
import random
import pandas as pd
from tqdm import tqdm

def dataset(data):
    random.seed(32)
    data_ = list(zip( data['idx'], data['forecasted_performance'],data['hs_sleep'], data['hs_study'], data['prev_scores'], data['practice'], data['extracurricular_activities']))
    random.shuffle(data_)
    idx, y, x1, x2, x3, x4, x5 = zip(*data_)
    x1 = list(x1)
    x2 = list(x2)
    x3 = list(x3)
    x4 = list(x4)
    x5= list(x5)
    y = list(y)
    for i in range(len(x5)):
        if x5[i] == False:
            x5[i] = 0
        else:
            x5[i] = 1

    X = np.zeros((len(x1), 5))
    for i in range(len(X)):
        X[i][0] = x1[i]
        X[i][1] = x2[i]
        X[i][2] = x3[i]
        X[i][3] = x4[i]
        X[i][4] = x5[i]
    
    Y = np.zeros((len(x1), 1))
    for i in range(len(Y)):
        Y[i] = y[i]

    return X, Y

class NeuralNetwork:
    def __init__(self, layers_, activation_functions, activation_grads_):
        # Initialize weights with small random values
        np.random.seed(32)
        self.layers = layers_
        self.W = [np.random.randn(layers_[i], layers_[i+1]) * 0.01 for i in range(len(layers_)-1)]
        self.B = [np.zeros((1, layers_[i+1])) for i in range(len(layers_)-1)]
        self.activations = activation_functions
        self.activation_grads = activation_grads_
    

    def forward(self, X):
        self.z = [X]
        self.a = [X]

        #self.z.append(np.dot(X, self.W[0]) + self.B[0])
        #self.a.append(self.activations[0](self.z[0]))
        for i in range(0, len(self.layers)-1): #hacer el for hasta len(self.layers) -1
            self.z.append(np.dot(self.a[i], self.W[i]) + self.B[i])
            self.a.append(self.activations[i](self.z[i+1]))
        return self.a[-1]

    def compute_loss(self, Y, Y_hat):
        return np.mean((Y - Y_hat) ** 2)
    
    def backprop(self, X, Y):
        delta = []
        self.dW = []
        self.dB = []

        delta.insert(0, (self.a[-1] - Y) * self.activation_grads[-1](self.a[-1])) #habria que reemplazar la resta por funcion de error?
        self.dW.insert(0, np.dot(self.a[-2].T, delta[-1]))
        self.dB.insert(0, np.sum(delta[-1], axis=0, keepdims=True))
        
        for i in range(len(self.layers)-2, 0, -1):
            delta.insert(0, np.dot(delta[0], self.W[i].T) * self.activation_grads[i-1](self.a[i]))
            self.dW.insert(0, np.dot(self.a[i-1].T, delta[0]))
            self.dB.insert(0, np.sum(delta[0], axis=0))

    def update_weights(self, learning_rate):
        # Update weights and biases using backpropagation deltas
        for i in range(len(self.W)):

            self.W[i] -= learning_rate * self.dW[i]
            self.B[i] -= learning_rate * self.dB[i]
            

    def train(self, X, Y, epochs, learning_rate):
        for epoch in tqdm(range(epochs)):
            for x, y in zip(X, Y):
                x = x.reshape((self.layers[0], 1)).T
                y = y.reshape((self.layers[-1], 1)).T
                Y_hat = self.forward(x)
                loss = self.compute_loss(y, Y_hat)
                self.backprop(x, y)
                self.update_weights(learning_rate)
                if epoch % 100 == 0:
                    print(f'Epoch {epoch}, Loss: {loss}')

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_grad(z):
    return z * (1 - z)

def relu(z):
    return np.maximum(0, z)

def relu_grad(z):
    return (z > 0).astype(float)

def linear(z):
    return z

def linear_grad(z):
    return 1

# Example usage
if __name__ == "__main__":
    path = 'datasets/Student_Performance_DEV.csv'
    y_predict = []
    data = pd.read_csv(path)
    X, Y = dataset(data)
    print(X[0])
    # Assuming input features are of size 2, hidden layer size 3, output size 1
    nn = NeuralNetwork([5, 3, 1], [relu, linear], [relu_grad, linear_grad])
    #X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # XOR input
    #Y = np.array([[0], [1], [1], [0]])  # XOR output
    nn.train(X, Y, epochs=500, learning_rate=0.01)
    print("prediccion: ",nn.forward(np.array([8.28329602, 4.08333045, 4, 12, 1])))

