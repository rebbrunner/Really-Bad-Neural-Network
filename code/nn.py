import numpy as np
from collections import Counter

# Simple NN from James Loy's "Neural Networks with Python text"

PUNCTUATION =','

tokenize = lambda x: [word.strip(PUNCTUATION) for word in x.split(' ')]
classification_map = {'up': 0, 'same': 1, 'down': 2}

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

def dloss_fn(actual, predicted):
    return 2 * (actual - predicted)

class NN:
    def __init__(self, x, y):
        self.input = x

        self.w1 = np.random.rand(self.input.shape[len(self.input.shape)-1], 4)
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

def gen_dict(fname):
    with open(fname) as f:
        lines = f.read().splitlines()
    x = [line.split('|')[0] for line in lines]
    y = [line.split('|')[1] for line in lines]
    unique_words = list(Counter([w for w in x for w in tokenize(w)]).keys())
    word2vec = {word: i for (i, word) in enumerate(unique_words)}
    return x, y, word2vec

def preprocess(fname):
    x, y, word2vec = gen_dict(fname)
    temp = []
    for sent in x:
        words = tokenize(sent)
        t = [0]*len(word2vec)
        for word in words:
            i = word2vec[word]
            t[i] = 1
        temp.append(t)
    x = np.array(temp)
    y = np.array([[classification_map[word]*.1] for word in y])
    return x, y

def train(fname):
    x, y = preprocess(fname)

    print(x)
    print(y)
    
    nn = NN(x, y)
    for _ in range(1500):
        nn.feed_forward()
        nn.back_prop()
    
    return nn

if __name__ == '__main__':
    nn = train('../data/diet.txt')
    i = 'egg, cheese'
    words = tokenize(i)
    _, _, word2vec = gen_dict('../data/diet.txt')
    t = [0]*len(word2vec)
    for word in words:
        i = word2vec[word]
        t[i] = 1
    nn.input = t
    nn.feed_forward()
    print(nn.output)
    res = int(round(nn.output[0],1)/.1)
    classification_map = {'up': 0, 'same': 1, 'down': 2}
    translated_map = dict((v,k) for k,v in classification_map.items())
    print(translated_map[res])
