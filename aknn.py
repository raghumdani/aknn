"""
This code implements KNN algorithm for predicting 
Heart Disease using K Nearest Neighbors algorithm

@authors:
raghumdani (13IT231)
praveen_kotre (13IT229)
thirthraj (13IT---)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors

# Keep this link relative
# To know more about dataset 
# visit http://archive.ics.uci.edu/ml/datasets/Heart+Disease

DWDM_DATASET = "processed.cleveland.data"

# Number of class present in data set 
# in this case, it is 5 [0, 4]

NO_CLASSES = 5

# How many tuples do I have to consider in training set ?
TRAINING_TUPLES = 200  # I have set this 2/3 of training set size

Xtrain = np.array([]) # Create an empty numpy array
Ytrain = np.array([]) # Result of each sample
Xtest = np.array([]) # '' testing value
Ytest = np.array([]) # '' testing target

tp = 0.0 # True positive 
tn = 0.0 # True negative
fp = 0.0 # False positive
fn = 0.0 # False negative

sz = 0 # Total size of training set

with open(DWDM_DATASET, "r") as data:
	for line in data:
		now = line.split(',')
		np.append(X, np.array(line[: -1])) # learn more about slice on SO
		np.append(Y, line[-1 :]) 

aknn = neighbors.KNeighborsClassifier(5, weights='uniform')

aknn.fit(X, Y)

