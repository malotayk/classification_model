import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors
from sklearn import preprocessing
from sklearn import datasets
from sklearn import linear_model
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

#open files
Train1 = open("/positive.review", "r")
Train2 = open("/negative.review", "r")
Test = open("/unlabeled.review", "r")

#Read the files and split them by lines
Train_pos = Train1.read().splitlines()
Train_neg = Train2.read().splitlines()
Test_reviews = Test.read().splitlines()

#Close the file pointers
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
x = np.zeros((len(train_set), len(features_dic)))
y = [y for (x,y) in train_set]

#We need to fill the matrix X properly now using feature ids
for i, (review, label) in enumerate(train_set):
  features = review.strip().split()[:-1]
  for item in features:
    feat, val = item.strip().split(":")
    x[i][features_dic[feat]] = val

avg=id_/195887
print("The average: ", avg)

X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)

print(neigh.predict([[1.1]]))

print(neigh.predict_proba([[0.9]]))
'''
X = x.reshape(1, -1)
y = y
clf = neighbors.KNeighborsClassifier()
clf.fit(x, y)
print(clf)
y_expect = y
y_pred = clf.predict(x)
print(metrics.classification_report(y_expect, y_pred))
'''
embeddings_dict = {}
with open("/glove.6B.50d.txt", "r", encoding="utf8") as f:
  for line in f:
    values = line.split()
    word = values[0]
    vector = np.asarray(values[1:], "float32")
    embeddings_dict[word] = vector

X, y = load_iris(return_X_y = True)
LRG = linear_model.LogisticRegression(
   random_state = 0, solver = 'liblinear', multi_class= 'auto')
LRG.fit(X, y)
print(LRG.score(X, y))