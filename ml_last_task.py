# -*- coding: utf-8 -*-
"""ML last Task

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19Pi8sx_PIts1064Ad3f5-0aj3a42T5Py
"""

pip install -U scikit-learn

pip install -U matplotlib

import numpy as np
import matplotlib.pyplot as plt
import math
import operator 


#This function used to get distance between ponts
def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


#This function return kNN 
def getKNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

#open files
Train1 = open("/positive.review", "r")
Train2 = open("/negative.review", "r")
Test = open("/unlabeled.review", "r")

#Read the files and split them by lines 
Train_pos = Train1.read().splitlines()
Train_neg = Train2.read().splitlines()
Test_reviews = Test.read().splitlines()

#Close th file pointers
Train1.close()
Train2.close()
Test.close()

#Combine the training sets to deal with one training set (easeir)
train_set= [(x, 1) for x in Train_pos] + [(x, -1) for x in Train_neg]
print("The total number of training data is: ", len(train_set))

#extract the features from the training data 
features_dic = {}
id_ = 0
for review, label in train_set: 
  features = review.strip().split()[:-1]
  for item in features:
    feat, val = item.strip().split(":")
    if feat not in features_dic:
      features_dic[feat]= id_
      id_+=1

print("The total number of features is: ", len(features_dic))

#Represent the reviews using the obtaine features
x = np.zeros((len(train_set), len(features_dic)+1))
y = [y for (x,y) in train_set]



#We need to fill the matrix X properly now using feature ids
for i, (review, label) in enumerate(train_set):
  features = review.strip().split()[:-1]
  for item in features:
    feat, val = item.strip().split(":")
    x[i][features_dic[feat]] = val
  x[i][len(x[i]) - 1] = y[i]

#We need to get test set ready
test_matrix = np.zeros((len(Test_reviews), len(features_dic)))

for i, review in enumerate(Test_reviews): 
  features_test = review.strip().split()[:-1]
  for item in features_test:
    feat, val = item.strip().split(":")
    if feat not in features_dic:
      continue    
    test_matrix[i][features_dic[feat]] = val

#z = np.zeros((len(train_set), len(features_dic)))
plt.plot(x[1500])

import time

#Set timer to count time of proseces 
tic = time.clock()

#test one vector on data set which is X matrex
neighbors = getKNeighbors(x[1000:1100], test_matrix[0], 1)
print(neighbors)
toc = time.clock()
time.strftime('%H:%M:%S', time.gmtime(toc - tic))