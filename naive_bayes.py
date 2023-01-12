from math import exp, sqrt, pi
import scipy.stats as stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('iris_dataset.csv')

# copy the data
df = dataset.copy()

setosa_train_df = df.iloc[0:40, :]
versicolor_train_df = df.iloc[50:90, :]
virginica_train_df = df.iloc[100:140, :]

setosa_test_df = df.iloc[40:50, :]
versicolor_test_df = df.iloc[90:100, :]
virginica_test_df = df.iloc[140:150, :]

n = 40
m = 10


def mean(setosa_train_df, versicolor_train_df, virginica_train_df):
    mean_matrix = [[], [], [], []]
    M = 0
    for z in range(3):
        if z == 0:
            train_df = setosa_train_df
        elif z == 1:
            train_df = versicolor_train_df
        else:
            train_df = virginica_train_df
        x = []
        for j in range(4):
            for i in range(n):
                M += train_df.iat[i, j]
            M = M / n
            x.append(round(M, 4))
            M = 0
        y = [[x[0]], [x[1]], [x[2]], [x[3]]]
        mean_matrix = np.append(mean_matrix, y, axis=1)
    return mean_matrix


def variance(mean_matrix, setosa_train_df, versicolor_train_df, virginica_train_df):
    variance_matrix = [[], [], [], []]
    V = 0
    for z in range(3):
        if z == 0:
            train_df = setosa_train_df
        elif z == 1:
            train_df = versicolor_train_df
        else:
            train_df = virginica_train_df
        x = []
        for j in range(4):
            for i in range(n):
                V += pow((train_df.iat[i, j] - mean_matrix[j][z]), 2)
            V = V / (n - 1)
            x.append(round(V, 4))
            V = 0
        y = [[x[0]], [x[1]], [x[2]], [x[3]]]
        variance_matrix = np.append(variance_matrix, y, axis=1)
    return variance_matrix


def normal_distribution(mean_matrix, variance_matrix, x):
    probabilities = {}
    for i in range(3):
        probabilities[i] = 40 / 120
        for j in range(4):
            exponent = exp(-(((x[j] - mean_matrix[j][i]) ** 2) / (2 * variance_matrix[j][i])))
            probabilities[i] *= (1 / (sqrt(2 * pi * variance_matrix[j][i]))) * exponent
    return probabilities


def test(mean_matrix, variance_matrix, setosa_test_df, versicolor_test_df, virginica_test_df):
    t = 0
    for z in range(3):
        if z == 0:
            test_df = setosa_test_df
        elif z == 1:
            test_df = versicolor_test_df
        else:
            test_df = virginica_test_df
        for i in range(m):
            f = normal_distribution(mean_matrix, variance_matrix, test_df.iloc[i, :])
            best_label, best_prob = None, -1
            for tagg, probability in f.items():
                if best_label is None or probability > best_prob:
                    best_prob = probability
                    best_label = tagg
            print(test_df.values[i], best_label)
            if best_label == z:
                t = t + 1
    print(t)
    accuracy = (t / 30) * 100
    print("accuracy is:", accuracy)
    return 0


def plot(mean_matrix, variance_matrix):
    figure, axis = plt.subplots(2, 2)
    for j in range(3):
        sigma = sqrt(variance_matrix[0][j])
        x = np.linspace(mean_matrix[0][j] - 10 * sigma, mean_matrix[0][j] + 10 * sigma, 1000)
        axis[0, 0].plot(x, stats.norm.pdf(x, mean_matrix[0][j], sigma))
        axis[0,0].set_title(dataset.columns.values[0])

        sigma = sqrt(variance_matrix[1][j])
        x = np.linspace(mean_matrix[1][j] - 10 * sigma, mean_matrix[1][j] + 10 * sigma, 1000)
        axis[0, 1].plot(x, stats.norm.pdf(x, mean_matrix[1][j], sigma))
        axis[0,1].set_title(dataset.columns.values[1])

        sigma = sqrt(variance_matrix[2][j])
        x = np.linspace(mean_matrix[2][j] - 10 * sigma, mean_matrix[2][j] + 10 * sigma, 1000)
        axis[1,0].plot(x, stats.norm.pdf(x, mean_matrix[2][j], sigma))
        axis[1,0].set_title(dataset.columns.values[2])

        sigma = sqrt(variance_matrix[3][j])
        x = np.linspace(mean_matrix[3][j] - 10 * sigma, mean_matrix[3][j] + 10 * sigma, 1000)
        axis[1,1].plot(x, stats.norm.pdf(x, mean_matrix[3][j], sigma))
        axis[1,1].set_title(dataset.columns.values[3])
    plt.show()


mean_matrix = mean(setosa_train_df, versicolor_train_df, virginica_train_df)
print("mean_matrix:\n", mean_matrix)
variance_matrix = variance(mean_matrix, setosa_train_df, versicolor_train_df, virginica_train_df)
print("variance_matrix:\n", variance_matrix)
print("test:")
test(mean_matrix, variance_matrix, setosa_test_df, versicolor_test_df, virginica_test_df)
plot(mean_matrix, variance_matrix)
