import numpy as np
from ast import literal_eval

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

def loss_fn(actual, predicted):
    squared_diff = (actual - predicted)**2
    return squared_diff.mean()

def dloss_fn(actual, predicted):
    return 2 * (actual - predicted)

class NN:
    def __init__(self, x, y):
        self.input = x

        self.w1 = np.random.rand(self.input.shape[1], 4)
        self.w2 = np.random.rand(4, 1)

        self.y = y
        self.output = np.zeros(y.shape)
    
    def feed_forward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.w1))
        self.output = sigmoid(np.dot(self.layer1, self.w2))
    
    def back_prop(self):
        chain1 = dloss_fn(self.y, self.output) * sigmoid_derivative(self.output)
        chain2 = np.dot(chain1, self.w2.T) * sigmoid_derivative(self.layer1)

        dw2 = np.dot(self.layer1.T, chain1)
        dw1 = np.dot(self.input.T, chain2)

        self.w1 += dw1
        self.w2 += dw2

def train():
    x = np.array([[0,0,1],
                  [0,1,1],
                  [1,0,1],
                  [1,1,1]])
    y = np.array([[0],[1],[1],[0]])
    nn = NN(x,y)

    for i in range(1500):
        nn.feed_forward()
        nn.back_prop()
    
    print(nn.output)

def save(nn, fname):
    o = open(fname, 'w+')
    for col in nn.w1:
        o.write((','.join(str(list(col)).split(','))) + '\n')
    o.write('--\n')
    for col in nn.w2:
        o.write((','.join(str(list(col)).split(','))) + '\n')
    o.close()

def load(fname):
    with open(fname) as f:
        lines = f.read()
    w1, w2 = lines.split('--')
    interpret = lambda x: literal_eval('['+ ','.join(x.splitlines()).strip(',') + ']')
    w1 = np.array(interpret(w1))
    w2 = np.array(interpret(w2))
    nn = NN(np.array([[0,0,0]]), np.array([[0]]))
    nn.w1 = w1
    nn.w2 = w2
    return nn

def predict(nn, x):
    nn.input = x
    nn.feed_forward()
    print(nn.output)

if __name__ == '__main__':
    nn = load('pickle.txt')
    predict(nn, np.array([[1,0,1]]))
