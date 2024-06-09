Simple KNN & Genetic Algorithm

K-Nearest Neighbor (KNN):
The classifier is designed and implemented which it performs like a K-Nearest Neighbor Algorithm (KNN). 
The KNN essentially uses the current data point or vector as points, and allocates the nearest neighbors (it determines the distance between the two data points) to classify itself belonging to a 
certain class or label. The amount of neighbors to find is determined by the “k-value”. In this case, we have two separate datasets are the Train set and Validation set along with their corresponding 
labels. The implementation needs to satisfy two different set, where the training set uses its current vector along with k-value amount of neighbor for classification of label. However, in validation 
set, which it uses each data as vector and used to allocate its neighbors within the training set. Therefore, the implementation are designed to satisfied by both usages while considering the k-value.

Distance Metric:
For this KNN, the euclidean distance measure, which takes two data points as parameters. For each point, it uses x-axis, y-axis and z-axis such that these individual axises represent the location 
of a point in terms of 3-dimensional space. Then, it uses the two sets of axises applying the Pythagorean theorem to compute the distance.

Instruction for KNN:
The KNN is prompted using the 'classifer.py', where the program compute the classification error percentage at each k-value (using either Valid or Training sets).

-----------------------------------------------------------------------------------------------------------
Genetic Algorithm:

The Project attempt to create genetic algorithm avoid using Sratch or any external libraries. The Genetic Algorithm aim to solve the problem:
The problem aims to find the coefficients of a0, a1 and a2 from the function f(x,y). In order to find suitable coefficients which can later be plugged against the training set. 
The initial population would need to be set within appropriate range, where the a0 need be some randomized decimal point between 0 and 2, a1 need to be randomized between -2 and 0 
and finally a2 need to be randomized between -1 and 1.
The program uses each set of points as chromosomes converting into 32 bits in IEEE 754 format. the chromosome would need to be applied with the objective function(or fitness function) to be used to determine the compatibility. 
For each chromosome in the population, the objective function includes plugging each data point in the training set with corresponding chromosomes (i.e. each gene represent a0, a1 and a2 in this order) using the (z - f(x,y))**2. 
The x-axis, y-axis, and z-axis are from the data point in the dataset, and applied all the coefficients into the function to yield the result. The function aims to minimize the distance between the data point and surface. 
The lower the value is, it represents the closer the result is closer to the surface. Each individual result will then be summed up and returned. The result will be later appended into the an array for computing 
the cumulative probabilities.
Using the cumulative probabilities, appropriate parents will be selected and therefore to create offsprings performing crossover and mutation depending on the rates.

Instructions:
The Genetic Algorithm is prompted using 'regression.py', where the program compute the genetic algorithm with corresponding parameters at the end of the program.
Such that generations, population_size, crossover & mutation.

-----------------------------------------------------------------------------------------------------------
More Information can be found in Explanation PDF: https://docs.google.com/document/d/1Qc8QFxidqc4tLv-tBBAcrdetQJEGFPdkSL7GAWKljW8/edit?usp=sharing
