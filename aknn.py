"""
This code implements KNN algorithm for predicting 
Heart Disease using K Nearest Neighbors algorithm

@authors:
raghumdani (13IT231)
praveen_kotre (13IT229)
thirthraj (13IT---)
"""

import numpy as np
from sklearn import neighbors

# Keep this link relative
# To know more about dataset 
# visit http://archive.ics.uci.edu/ml/datasets/Heart+Disease

DWDM_DATASET = "processed.cleveland.data"

# Number of class present in data set 
# in this case, it is 5 [0, 4]

NO_CLASSES = 5

Xnow = [] # training dataset
Ynow = []

# How many tuples do I have to consider in training set ?
TRAINING_TUPLES = 200  # I have set this 2/3 of training set size

testTuple = [] # '' testing value

tp = 0.0 # True positive 
tn = 0.0 # True negative
fp = 0.0 # False positive
fn = 0.0 # False negative

sz = 0 # Total size of training set

with open(DWDM_DATASET, "r") as data:
  for line in data:
    sz += 1
    now = line.split(',')
    cur = []
    cnt = 0
    try :
      for i in now:
        cnt += 1
        if (cnt == 13) :
          continue
        if i == '?':
          cur.append(0)
        else:
          cur.append(float(i))
    except ValueError, e:
      print "error", e, "on line" , sz 
    
    print "Processing sample " + str(sz) + " = ", cur
    if sz < TRAINING_TUPLES:
      Xnow.append(cur[: -1]) # learn more about slice on SO
      Ynow.append(cur[-1 : ][0])
    else:
		  testTuple.append(cur)

aknn = neighbors.KNeighborsClassifier(2, weights='distance')

Xtrain = np.array(Xnow) # Create an empty numpy array
Ytrain = np.array(Ynow) # Result of each sample

print Xtrain, Ytrain
aknn.fit(Xtrain, Ytrain)

for tup in testTuple:
  cur = []
  for i in range(12) :
    cur.append(tup[i])
  
  y = aknn.predict(np.array(cur))
  real = tup[-1 : ][0] # take the last value
  if (y <= 1) :
    if real <= 1:
      tn += 1
    else :
      fn += 1
  else:
    if real > 1:
      tp += 1
    else :
      tn += 1


# now calculate precision and recall and analyse the output

precision = tp / (tp + fp)
recall = tp / (tp + fn)
accuracy = (tp + tn) / (tp + tn + fp + fn)

print "Precision = %f and Recall = %f and Accuracy = %f" % (precision, recall, accuracy)
