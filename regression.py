import numpy as np
import random

# f(x,y) = a0*(x-5)^1/3 + a1*(y+5)^1/3 + a2   a0=[0,2] a1=[-2,0] a2=[-1,1] 
# vector : [1.2, -3.2, -4.1, 1] 1=above surface & -1=below surface
# Some helper methods
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

def addLabel(data, label):
    arr = [float(data[0]), float(data[1]), float(data[2]), label]
    return arr

#help method to support integer converts to binary within a decimal
def int_to_binary(int_number, available_places):
    bits = ''
    while int_number > 0 and len(bits) < available_places:
        quotient = int_number / 2
        has_decimal = quotient % 1 != 0
        if has_decimal:
            bits = '1' + bits
        else:
            bits = '0' + bits
        int_number = int(quotient)
    if not bits:
        bits = '0'
    return bits

#Given the floating number into binary representation
def decimal_to_binary(floatnum):
    #Set up and breaking the inputs with signals, integers, fraction part 
    floatstr = str(floatnum).replace('-', '')
    decimal_part = floatstr .split('.')[0]
    fraction_part = floatstr .split('.')[1]
    signal_bit = (1 if floatnum < 0 else 0)
    decimal_bits = ''
    fraction_bits = ''
    mantissa_available_bits = 23

    if decimal_part:
        decimal_part = int(decimal_part)
        decimal_bits = int_to_binary(decimal_part, mantissa_available_bits)

    if fraction_part:
        fraction_part = float('0.' + fraction_part)
        extra_bits = 0
        count_zeros_to_the_left = True
        while len(fraction_bits) < (mantissa_available_bits + extra_bits) :
            product = fraction_part * 2
            bit = '1' if product >= 1 else '0'
            fraction_bits += bit
            fraction_part = str(product).split('.')[1]
            if fraction_part == '0':
                fraction_part = 0
                break

            fraction_part = float('0.' + fraction_part)
            if count_zeros_to_the_left:
                extra_bits = 0
                for fraction_bit in fraction_bits:
                    if fraction_bit == '0':
                        extra_bits += 1
                    else:
                        count_zeros_to_the_left = False
                        break
        if not fraction_bits:
            fraction_bits = '0'
        # round binary
        if fraction_part != 0:
            fraction_bits += '1'

    mantissa = decimal_bits + fraction_bits
    exponent = 0
    zeros_shifted = mantissa.find('1')
    exponent = len(decimal_bits) - 1

    if decimal_bits == '0':
        exponent -= 1

    if zeros_shifted > 0:
        exponent -= zeros_shifted - 1

    mantissa = mantissa[zeros_shifted + 1:mantissa_available_bits].ljust(mantissa_available_bits, '0')
    exponent += 127
    exponent_bits = int_to_binary(exponent, mantissa_available_bits).rjust(8, '0')
    return f'{signal_bit}{exponent_bits}{mantissa}'

# get binary for integers from inputed binarys
def binary_to_int(binary):
    output = 0
    for idx, bit in enumerate(reversed(binary)):
        if bit == '1':
            output += 2 ** idx

    return output
# binary to floating point number
def binary_to_decimal(bin):
    signal_bit = bin[0]
    exponent_binary = bin[1:9]
    mantissa = bin[9:]
    decimal_bits = '0'
    fraction_bits = '0'

    exponent = binary_to_int(exponent_binary) - 127
    if exponent >= 0:
        decimal_bits = '1' + mantissa[: exponent]
        fraction_bits = mantissa[exponent:]
    else:
        fraction_bits = '0' * (abs(exponent) - 1) + '1' + mantissa
    decimal_number = binary_to_int(decimal_bits)
    fraction_number = 0
    for idx, char in enumerate(fraction_bits):
        if char == '1':
            fraction_number += 2 ** -(idx + 1)
    return (-1)**int(signal_bit) * float(decimal_number + fraction_number)

def check_child(child):
    a2 = binary_to_decimal(child[2])
    if a2 > 1 or a2 <-1:
        return True
    return False

#------------------------------------------------------------------------------------------------------
#Below is genetic algorithm
def initialize_population(population_size):
    pop = np.random.rand(population_size, 3) * np.array([2, 2, 2]) - np.array([0, 2, 1])
    population = []
    print(pop)
    for i in pop:
        add_pop = []
        for j in i:
            temp = decimal_to_binary(j)
            add_pop.append(temp)
        population.append(add_pop)
    population = np.array(population)
    print("Population : {}".format(population))     
    return population

def poly_function(x, y, a0, a1, a2):
    return a0 * (np.cbrt(x-5)) + a1 * (np.cbrt(y - 5)) + a2

def fitness(chromo, data):
    fitnesses = []
    a0, a1, a2 = binary_to_decimal(chromo[0]),binary_to_decimal(chromo[1]), binary_to_decimal(chromo[2])
    for i in data:
        min = i[2] - poly_function(i[0], i[1], a0, a1, a2)
        fitnesses.append(min)
    
    sum_errors = np.sum(np.array(fitnesses))
    return sum_errors**2


def select_Chromosomes(population, data):
    fitness_values = []
    for chromosome in population:
        fit = fitness(chromosome, data)
        fitness_values.append(fit)

    fitness_Sum = sum(fitness_values)
    probabilities = [float(i)/fitness_Sum for i in fitness_values]
    cumulative_probability = [i for i in probabilities]
    for i in range(1,len(population)):
        cumulative_probability[i] = cumulative_probability[i-1] + probabilities[i]
    # Use roulette spins to pick random chromosomes
    """
    print("Cumulative probability:")
    for i in range(len(population)):
        print(f'{population[i]} -- {cumulative_probability[i]}')
    """
    roulette_spins = [(random.uniform(0,1),random.uniform(0,1)) for i in range(len(population)//2)]

    #print("Random numbers for selection:")
    """
    for i in range(len(roulette_spins)):
        print(f'({roulette_spins[i][0]} -- {roulette_spins[i][1]})')
    print()
    """
    parents = []

    for i, j in roulette_spins:
        #print("i: {}, j: {}".format(i, j))
        parent1_pos = select_parent(cumulative_probability, i)
        parent2_pos = select_parent(cumulative_probability, j)
        #print("P1: {}, P2: {}".format(parent1_pos, parent2_pos))
        parents += [(population[parent1_pos], population[parent2_pos])]
    #print()
    return parents

#Use to transfrom binaries to float, ordered
def transfrom_binary_float(pop):
    arr = [binary_to_decimal(pop[0]), binary_to_decimal(pop[1]), binary_to_decimal(pop[2])]
    return arr

# select a parent based on the cummulative probabilties
def select_parent(cp, value):    
    for i in range(len(cp)):
        if value <= cp[i]:
            return i

def point_Crossover(parent1, parent2, crossover):
    if random.uniform(0,1) <= crossover:
        #3 different crossover points for 3 different genes in a single chromosome
        crossover_point = random.randint(1, len(parent1[0])-1)
        crossover2 = random.randint(1, len(parent1[1])-1)
        crossover3 = random.randint(1, len(parent1[0])-1)

        child1 = [parent1[0][0:crossover_point] + parent2[0][crossover_point:], parent1[1][0:crossover2] + parent2[1][crossover2:], parent1[2][0:crossover3] + parent2[2][crossover3:]]
        child2 = [parent2[0][0:crossover_point] + parent1[0][crossover_point:], parent2[1][0:crossover2] + parent1[1][crossover2:], parent2[2][0:crossover3] + parent1[2][crossover3:]]
        return child1, child2

    #print(f"No crossover beetween {parent1} and {parent2}\n")
    return parent1, parent2

def mutate(chromosome, mutation_probability):
    for i in chromosome:
        mutated = False
        for j in range(len(i)):
            if random.uniform(0,1) <= mutation_probability:
                #print(f"Performing mutation on chromosome {i} at index {j}")
                mutated = True
    
                if i[j] == 0:
                    #i[j] = '1'
                    new_Mutate = i[:j] + i[j:].replace(i[j], '1', 1)
                    #print(new_Mutate)
                else:
                    new_Mutate = i[:j] + i[j:].replace(i[j], '0', 1)
                    #print(new_Mutate)
    """  
    if mutated:
            #print(f"Resulting chromosome: {chromosome}\n")
            
    else:
            #print(f"No mutation for {chromosome}\n")
            
    """
    return chromosome

def genetic_Algorithm(data, population_Size, generations, crossover, mutation):
    population = initialize_population(population_Size)
    final_pop = []
    for _ in range(generations):
        population = initialize_population(population_Size)
        parents = select_Chromosomes(population, data)
        
        for parent1, parent2 in parents:
            child1, child2 = point_Crossover(parent1, parent2, crossover)
            child1 = mutate(child1, mutation)
            child2 = mutate(child2, mutation)

            ch1 = check_child(child1)
            ch2 = check_child(child2)
            
            if ch1 == True:
                continue
            elif ch2 == True:
                continue
            else:
                final_pop += [child1,child2]
    
    best = get_Best(final_pop, data)
    print("Best: {}".format(best))
    return

def get_Best(pops, data):
    fitness_values = []
    for chromo in pops:
        fit = fitness(chromo, data)
        fitness_values.append(fit)

    min_value = min(fitness_values)
    min_index = fitness_values.index(min_value)
    
    lowest_Chromo = pops[min_index]
    ret_Coefficients = []
    for i in lowest_Chromo:
        ret_Coefficients.append(binary_to_decimal(i))

    return ret_Coefficients

def print_Pop(population):
    count = 0
    for i in population:
        print("Count: {}\n".format(count))
        print("A's: {} , {} , {}".format(binary_to_decimal(i[0])) )
        

train = getTrainSet()
total = len(train)
train = np.array(train)
generations = 10
ppopulation_size = 20
crossover = 0.8
mutation = 0.1

genetic_Algorithm(train, ppopulation_size, generations, crossover, mutation)
