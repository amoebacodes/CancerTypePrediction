import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# import the data
def import_data(dir_list):
    first_file = True
    for i, dir in enumerate(dir_list):
        for filename in os.listdir(dir):
            if first_file:
                data_np = np.genfromtxt("{0}/{1}".format(dir, filename), delimiter="\t")[:,1].T
                data_np = np.append(data_np, 0)

                data_pd = pd.read_csv("{0}/{1}".format(dir, filename), header=None, delimiter="\t")
                gene_list = data_pd[0].tolist()
                first_file = False
            else:
                temp = np.genfromtxt("{0}/{1}".format(dir, filename), delimiter="\t")[:,1].T
                temp = np.append(temp, i)
                data_np = np.vstack((data_np, temp))
        
    return data_np, gene_list

# mean center the data and divide by the variance per column
def standardize_data(data):
    # z = (x - u) / s
    means = np.mean(data, axis = 1)
    std = np.std(data, axis = 1)
    X = np.zeros(data.shape)
    for i, mean in enumerate(means):
        for j in range(len(data)):
            X[j][i] = (data[j,i] - mean)/std[i]
    return X, means, std

# fit X with standardization
def standardize_fit(data, means, std):
    X = np.zeros(data.shape)
    for i, mean in enumerate(means):
        for j in range(len(data)):
            X[j][i] = (data[j,i] - mean)/std[i]
    return X

# split the data into a train and test set
def split_data(data, percent_train=0.8):
    shuffled_data = np.random.rand(data.shape[0])
    split_mask = shuffled_data < np.percentile(shuffled_data, percent_train*100)

    X_train, y_train = data[split_mask][:,:-1], data[split_mask][:,-1]
    X_test, y_test =  data[~split_mask][:,:-1], data[~split_mask][:,-1]

    return X_train, y_train, X_test, y_test

# sigmoid function
def sigmoid(val):
    return 1/(1+np.exp(-val))

# binary cross entropy loss calculation
def loss(y, y_hat):
    return -np.mean(y*np.log(y_hat)) - (1-y)*np.log(1-y_hat)

# train the model
def train(X, y, batch_size, iterations, learning_rate):
    y = y.reshape(X.shape[0],1)
    # n = X.shape[0]
    w, b = np.zeros((X.shape[1],1)), 0

    loss_list = []

    # mini batch gradient descent
    for _ in range(iterations):
        for j in range(math.floor((X.shape[0]-1)/batch_size)):
            X_batch = X[j*batch_size:min(j*batch_size+batch_size, X.shape[0])]
            y_batch = y[j*batch_size:min(j*batch_size+batch_size, y.shape[0])]

            y_hat = sigmoid(np.dot(X_batch, w) + b)

            grad_w = (1/X.shape[0])*np.dot(X_batch.T, (y_hat - y_batch))
            grad_b = (1/X.shape[0])*np.sum((y_hat - y_batch))

            w, b = w - learning_rate * grad_w, b - learning_rate * grad_b

        loss_list = np.append(loss_list, loss(y, sigmoid(np.dot(X, w) + b)))

    return w, b, loss_list

# predict on unseen data with trained parameters, outputs 0 or 1
def predict(X, w, b):
    predict_proba = sigmoid(np.dot(X, w) + b).reshape(X.shape[0],)
    y_hat = np.asarray([1 if pred >= 0.5 else 0 for pred in predict_proba])
    return y_hat

# predict on unseen data with trained parameters, outputs probability of class 1
def predict_proba(X, w, b):
    return sigmoid(np.dot(X, w) + b).reshape(X.shape[0],)

# calculate accuracy of the model
def accuracy(y, y_hat):
    return np.sum(y == y_hat)/len(y)

# calculate f1 score
def f1_score(y, y_hat):
    # tp/(tp + .5 (fp + fn))
    true_positive, false_positive, false_negative = 0, 0, 0
    for act, pred in zip(y, y_hat):
        if act == 1 and pred == 1:
            true_positive += 1
        elif act == 0 and pred == 1:
            false_positive += 1
        elif act == 1 and pred == 0:
            false_negative += 1

  
    return true_positive/(true_positive + .5 * (false_positive + false_negative))

# calculate precision
def precision(y, y_hat):
    true_positive, false_positive, false_negative = 0, 0, 0
    for act, pred in zip(y, y_hat):
        if act == 1 and pred == 1:
            true_positive += 1
        elif act == 0 and pred == 1:
            false_positive += 1
        elif act == 1 and pred == 0:
            false_negative += 1

    return true_positive/(true_positive + false_positive)

# calculate recall
def recall(y, y_hat):
    true_positive, false_positive, false_negative = 0, 0, 0
    for act, pred in zip(y, y_hat):
        if act == 1 and pred == 1:
            true_positive += 1
        elif act == 0 and pred == 1:
            false_positive += 1
        elif act == 1 and pred == 0:
            false_negative += 1

    return true_positive/(true_positive + false_negative)

# import data
adrenal_dir = './adrenal_gland'
kidney_dir = './kidney'
dir_list = [adrenal_dir, kidney_dir]
data, gene_list = import_data(dir_list)

# train the model
X_train, y_train, X_test, y_test = split_data(data)
X_train, mu, sigma = standardize_data(X_train)
X_test = standardize_fit(X_test, mu, sigma)
w, b, l = train(X_train, y_train, batch_size=32, iterations=100, learning_rate=0.01)

# fit the model
y_hat = predict(X_test, w, b)

# print results
print("Accuracy:", accuracy(y_test, y_hat))
print("F1:", f1_score(y_test, y_hat))
print("Precision:", precision(y_test, y_hat))
print("Recall:", recall(y_test, y_hat))
