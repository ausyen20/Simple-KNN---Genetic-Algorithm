#classification error for trainset and validation set
import numpy as np

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

def addLabel(data, label):
    arr = [float(data[0]), float(data[1]), float(data[2]), label]
    return arr

def getValidLabels():
    lines = tuple(open('validation_labels_v2','r'))
    validlabels = []
    for line in lines:
        temp = int(line.strip())
        validlabels.append(temp)
    return validlabels
#-------------------------------------------------------

def poly_function(x, y, a0, a1, a2):
    return a0 * (np.cbrt(x-5)) + a1 * (np.cbrt(y - 5)) + a2

def classify_trainset(coefficients):
    trainset = getTrainSet()
    
    incorrect = 0
    correct = 0
    total = len(trainset)
    for i in trainset:
        func = poly_function(i[0],i[1], coefficients[0], coefficients[1],coefficients[2])
        z = i[2]
        label = i[3]
        #print("x: {}, Y: {}, Z: {}, a0: {}, a1: {} , a2: {}, Label: {}".format(i[0],i[1],i[2],coefficients[0], coefficients[1],coefficients[2],  i[3])) 
        #print(func)
        
        if label == -1 and z < func:
            correct += 1
        elif label == 1 and z >= func:
            correct += 1
        else:
            incorrect += 1
    
    #print("Incorrect: {}, Correct: {}".format(incorrect, correct))
    return incorrect/total


def classify_validset(coefficients):
    validset = getValidSet()
    incorrect = 0
    correct = 0
    total = len(validset)
    a0 = coefficients[0]
    a1 = coefficients[1]
    a2 = coefficients[2]
    for j in validset:
        x = j[0]
        y = j[1]
        z = j[2]
        label = j[3]
        func = poly_function(x,y,a0,a1,a2)
        #print("x: {}, Y: {}, Z: {}, a0: {}, a1: {} , a2: {}, Label: {}".format(x,y,z,a0,a1,a2,label))
        #print(func)
        if label == -1 and z < func:
            correct += 1
        elif label == 1 and z >= func:
            correct += 1
        else:
            incorrect += 1
    
    #print("Incorrect: {}, Correct: {}".format(incorrect, correct))
    return incorrect/total

def find_average(arr):
    allXs = []
    allYs = []
    allZs = []
 
    for i in arr:
        x = i[0]
        y = i[1]
        z = i[2]

        allXs.append(x)
        allYs.append(y)
        allZs.append(z)
    
    average = [Average(allXs), Average(allYs), Average(allZs)]

    return average

def Average(lst): 
    return sum(lst) / len(lst) 

if __name__ == '__main__':

    
    """
    all_coefficients = [[1.3007071018218994, -1.5851657390594482, 0.7101187705993652], [1.073455810546875, -1.9424703121185303, -0.07545778155326843],
                        [0.7466449737548828, -1.775388479232788, -0.06446552276611328], [1.5055959224700928, -1.716930866241455, 0.9833319187164307],
                        [1.051788568496704, -1.9779648780822754, 0.9779269695281982], [1.896963357925415, -1.7994964122772217, 0.8520278930664062],
                        [1.1479322910308838, -1.990422010421753, 0.053217023611068726], [1.073430061340332, -1.6307706832885742, 0.770355224609375],
                        [0.5780307054519653, -0.9109597206115723, 0.784489631652832],  [0.21398061513900757, -0.002752542495727539, 0.23851972818374634] ]

    allAverage = find_average(all_coefficients)
    #print(allAverage)
    """
    # You can edit you coefficient here, to test the classification error
    #Modified the variable coefficients to test the classification errors
    coefficients = [0,0,0] #This one here!
    print(coefficients)
    error_in_Trainset = classify_trainset(coefficients)
    error_in_ValidSet = classify_validset(coefficients)

    print(f"Error in Trainset : {error_in_Trainset:.0%} in 360 data points")
    print(f"Error in Validset : {error_in_ValidSet:.0%} in 540 data points")