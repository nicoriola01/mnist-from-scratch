import pandas as pd
import numpy as np

#csv loading
training_set = pd.read_csv('mnist_train.csv') # put the absolute path of mnist_train.csv
test_set = pd.read_csv('mnist_test.csv') # put the absolute path of mnist_test.csv

#transforming from pandas to numpy arrays
training_set = np.array(training_set)
test_set = np.array(test_set)

#shuffling to improve generality
np.random.shuffle(training_set)
np.random.shuffle(test_set)

#saving size of training and test sets
m_train, n_train = training_set.shape
m_test, n_test = test_set.shape

#transposing for easier manipulation
train = training_set.T
test = test_set.T

#separating labels from data
labels_train = train[0]
data_train = train[1:]

labels_test = test[0]
data_test = test[1:]

#normalizig values for better performance
data_test = data_test / 255
data_train = data_train / 255

print("Data loaded successfully from .csv files")
