#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import svm

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100] 


#########################################################
### your code goes here ###
### we handle the import statement and SVC creation for you here
from sklearn.svm import SVC
clf = SVC(C=10000.0,kernel="rbf")


#### now your job is to fit the classifier
#### using the training features/labels, and to
#### make a set of predictions on the test data

clf.fit(features_train, labels_train)

pred = clf.predict(features_test)

#print(clf.predict([[-0.8, -1]]))

#### store your predictions in a list named pred

print("10th: %r, 26th: %r, 50th: %r" % (pred[10], pred[26], pred[50]))

# There are over 1700 test events, how many are predicted to be in the "Chris" (1) class?
print("No. of predicted to be in the 'Chris'(1): %r" % sum(pred))


from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)

print(acc)

def submitAccuracy():
    return acc
#########################################################


