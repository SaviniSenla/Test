import neural
import numpy as np
import pandas as pd
import random


def inOut():
    # prepare inputs and outputs

    # getting inputs from shuffled file
    fload = pd.read_csv('G:\Dialog\Data Set\shuffled.txt')
    dataSet = np.array(fload)

    # getting all inputs
    inputs = np.array(np.delete(dataSet, [4], axis=1), dtype=float)

    # get 100 inputs as a training set
    X = inputs[:100]

    # normalize the input training set
    m = np.amax(X, axis=0)
    X = X / m

    # normalized input set for testing
    testI = inputs[100:] / m

    # one hot encoding for output categories
    df = pd.DataFrame(fload)
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
    outputs = np.array(pd.get_dummies(df['class']).values.tolist(), dtype=int)

    # 100 outputs as training set
    Y = outputs[:100]

    # rest as testing set
    testO = outputs[100:]

    return X, Y, testI, testO


def shuffle():
    # shuffle the data file

    with open('G:\Dialog\Data Set\BezdekIris.data.txt', 'r') as source:
        data = [(random.random(), line) for line in source]
    data.sort()
    with open('G:\Dialog\Data Set\shuffled.txt', 'w') as target:
        for _, line in data:
            target.write(line)
    return None


def softMax(x):
    # calculate soft max values

    result = np.zeros_like(x)
    M, N = x.shape
    S = np.sum(np.exp(x), axis=1)
    for n in range(N):
        result[:, n] = np.exp(x[:, n]) / S
    return result


def minimizeW(iterations, x, y, alpha, inputVal):
    # return outputs for input data set

    NN = neural.NeuralNetwork()
    for i in range(iterations):
        dw1, dw2 = NN.costFunctionPrime(x, y)
        NN.w1 = NN.w1 - alpha * dw1
        NN.w2 = NN.w2 - alpha * dw2
        # print(NN.costFunction(X, y))
    Y = NN.forward(inputVal)
    return Y


def predictionNames(x, y, testInputs):
    # predicts the name of flower for a given input set

    predictedValues = minimizeW(100000, x, y, 0.249, testInputs)
    softValues = softMax(predictedValues)
    maxIndex = np.array(np.argmax(softValues, axis=1), dtype=str)
    maxIndex[maxIndex == "0"] = "Iris-setosa"
    maxIndex[maxIndex == "1"] = "Iris-versicolor"
    maxIndex[maxIndex == "2"] = "Iris-virginica"
    return maxIndex


def prediction(x, y, testO, testI):
    # calculate the hit ratio

    predictedValues = minimizeW(100000, x, y, 0.249, testI)
    softValues = softMax(predictedValues)
    maxIndex = np.array(np.argmax(softValues, axis=1), dtype=int)
    maxIndexOut = np.array(np.argmax(testO, axis=1), dtype=int)
    hit = 0
    size = len(maxIndex)
    for i in range(0, size):
        if maxIndexOut[i] == maxIndex[i]:
            hit = hit + 1
    return "Hit ratio is : %0.2f" % (float(hit) / size * 100)


shuffle()
X, Y, testIn, testOut = inOut()
print(prediction(X, Y, testOut, testIn))
