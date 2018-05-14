import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import chap02_01_a_perceptron as perceptron

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header = None)
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

X = df.iloc[0:100, [0,2]].values

'''
plt.scatter(X[:50, 0], X[:50,1], color='red', marker='o', label='setosa')
plt.scatter(X[50:, 0], X[50:, 1], color='blue', marker='x', label='versicolor')
plt.xlabel('sepal length in cm')
plt.ylabel('petal length in cm')
plt.legend(loc='upper left')
plt.show()

'''


model = perceptron.Perceptron(alpha = 0.1, no_epochs = 10)
model.fit(X,y)
plt.plot(range(1, len(model.errors_) + 1), model.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassification')
plt.show()
