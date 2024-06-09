import numpy as np
import math 
from math import sqrt

def euclidian_distance(point1, point2):
    distance  = 0.0 
    #all points between the two given data points
    x1 = float(point1[0])
    y1 = float(point1[1])
    z1 = float(point1[2])
    x2 = float(point2[0])
    y2 = float(point2[1])
    z2 = float(point2[2])
    distance = math.sqrt(((x2-x1)**2) + ((y2-y1)**2) + ((z2-z1)**2))
    return distance

#Classification of Error Rate
def classification_error(bool, k):
    #if True, then process set for Training set
    if bool == True:
        curr_set = getTrainSet()
        incorrect = 0
        total = len(curr_set)
        for i in curr_set:
            result = predict_Class(curr_set, i, k, bool)
            if result == False:
                #if prediction and input's label not equals, then increment incorrect
                incorrect += 1
        error_rate = get_ratio(incorrect, total)
        error_rate = round(float(error_rate), 4)
        return error_rate
    else:
        #If not, then do Validation Set
        curr_set2 = getValidSet()
        incorrect2 = 0
        total = len (curr_set2)
        for j in curr_set2:
            result2 = predict_Class(curr_set2, j, k, bool)
            if result2 == False:
                incorrect2 += 1
            error_rate2 = get_ratio(incorrect2, total)
            error_rate2 = round(float(error_rate2),4)
        return error_rate2

def predict_Class(set, input, k, bool):
    if bool == False:
        set = getTrainSet()
    neighs = knn(set, input, k, bool)
    Classes = []
    for i in neighs:
        Classes.append(i[-1])
    #Find the counts for most appearances
    prediction = max(Classes, key= Classes.count)
    #print("Input: {}, Prediction: {}, Classes: {}, Neighbors: {}".format(input, prediction, Classes, neighs))
    if int(prediction) != int(input[-1]):
        #If prediction doesn't match input's label, then get incorrect state
        #print("Input: {}, Prediction: {}, Classes: {}, Neighbors: {}".format(input, prediction, Classes, neighs))
        return False
    return True

def knn(set, input, k, bool):
    distances = list()
    data = []
    for i in set:
        dist = euclidian_distance(input, i)
        distances.append(dist)
        data.append(i)
    distances = np.array(distances)
    data = np.array(data)
    # arrage indexes based on ascending
    index_dist = distances.argsort()
    # re-arrange distances based on index_dist
    data = data[index_dist]
    # get amount of neighbors based on k value
    if bool == True:
        # Training set, including itself
        allNeighbors = data[:k]
        return allNeighbors
    else:
        # Validation set, excluding itself
        allNeighbors2 = data[1:k+1]
        return allNeighbors2

# get Ratios between the incorrect counts against whole set
def ratio(a, b):
    a = float(a)
    b = float(b)
    if b == 0:
        return a
    return ratio(b, a % b)
def get_ratio(a, b):
    r = ratio(a, b)
    return "%s" % float((a/r) / (b/r))

# get Train & Valid sets and labels
def getTrainSet():
    lines = tuple(open('training_set_v2', 'r'))
    labels = getTrainLabels()
    trainset = []
    count = 0
    for line in lines:
        temp = line.strip().split(",")
        add = addLabel(temp, labels[count])
        trainset.append(add)
        count += 1
    return trainset

def getTrainLabels():
    lines = tuple(open('training_labels_v2','r'))
    trainlabels = []
    for line in lines:
        temp = int(line.strip())
        trainlabels.append(temp)
    return trainlabels

def getValidSet():
    lines = tuple(open('validation_set_v2','r'))
    labels = getValidLabels()
    validset = []
    count = 0
    for line in lines:
        temp = line.strip().split(",")
        add = addLabel(temp, labels[count])
        validset.append(add)
        count += 1
    return validset

def getValidLabels():
    lines = tuple(open('validation_labels_v2','r'))
    validlabels = []
    for line in lines:
        temp = int(line.strip())
        validlabels.append(temp)
    return validlabels

def addLabel(data, label):
    arr = [data[0], data[1], data[2], label]
    return arr

# Main functions call all relevant methods
if __name__ == '__main__':
    k7_train = classification_error(True, 7)
    k7_valid = classification_error(False, 7)
    k19_train = classification_error(True, 19)
    k19_valid = classification_error(False, 19)
    k31_train = classification_error(True, 31)
    k31_valid = classification_error(False, 31)
    print("k=7\nClassification error on training set  : {}\nClassification error on validation set: {}".format(k7_train, k7_valid))
    print("\nk=19\nClassification error on training set  : {}\nClassification error on validation set: {}".format(k19_train, k19_valid))
    print("\nk=31\nClassification error on training set  : {}\nClassification error on validation set: {}".format(k31_train, k31_valid))