import numpy as np


def createTrainData():
    data = np.loadtxt("G:\Dialog\Data Set\BezdekIris.data.txt", delimiter=",", dtype=str)
    inputs = np.array(data[:, :4])
    inputs = inputs.astype(np.float)
    outputs = np.array(data[:, 4])
    return inputs, outputs


def label(label):
    if label == "Iris-setosa":
        return [1, 0, 0]
    elif label == "Iris-versicolor":
        return [0, 1, 0]
    else:
        return [0, 0, 1]

a = np.array([1, 0, 3])
b = np.zeros((3, 4))
b[np.arange(3), a] = 1

x, y = createTrainData()
print(y[0])

inputCategories = np.array(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica' ])
oneHot = np.zeros((3, 3))
oneHot[np.arange(3), inputCategories] = 1

print oneHot