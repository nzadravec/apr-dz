import numpy as np
from random import *
from math import *

class Fitness:

    def __init__(self, f, decode):
        self.f = f
        self.decode = decode

    def fitnessOf(self, individual):
        return self.f(self.decode(individual.genes))
        
def bool2int(bool_):
    int_ = 0
    for b in bool_:
        int_ = (int_ << 1) | b
    return int_
        
class BinaryDecoder:

    def __init__(self, ll, ul, bits, n):
        self.ll = ll
        self.ul = ul
        self.bits = bits
        self.n = n
        self.totalBits = n * bits
        
    def decode(genes):
        x = np.empty(self.n)
        for i in range(0, self.n):
            x[i] = binaryToNum(
                bool2int(genes[i*self.bits:(i+1)*self.bits]), self.ll, self.ul, self.bits)
                
class GrayDecoder:

    def __init__(self, ll, ul, bits, n):
        self.binaryDecoder = BinaryDecoder(ll, ul, bits, n)
        
    def decode(genes):
        genes = grayToBinary(genes)
        return self.binaryDecoder.decode(genes)
   
def passThroughDecoder(genes):
    return genes

class Individual:
    
    def __init__(self, genes):
        self.fitness = None
        self.genes = genes

def numToBinary(x, ll, ul, n):
    return (x - ll)*(pow(2, n) - 1) / (ul - ll)
    
def binaryToNum(b, ll, ul, n):
    return ll + b*(ul - ll) / (pow(2, n) - 1)
    
def precision(p, ll, ul):
    return ceil(log(1 + (ul - ll)*pow(10, p)) / log(2))
    
def binaryToGray(binary):
    gray = np.empty(len(binary))
    gray[0] = binary[0]
    for i in range(1, len(binary)):
        gray[i] = binary[i-1] ^ binary[i]
    
    return gray
    
def grayToBinary(gray):
    binary = np.empty(len(binary))
    binary[0] = value = gray[0]
    for i in range(1, len(binary)):
        if gray[i]:
            value = not value
        binary[i] = value
     
    return binary
        
def rouletteWheelSelection(pop):
    worstFitness = np.min(list(map(lambda x : x.fitness, pop)))
    fitnessSum = sum(map(
        lambda x : (x.fitness - worstFitness), pop))
    p = [(x.fitness - worstFitness) / fitnessSum for x in pop]
    
    return np.random.choice(pop, 1, p)[0]
    
def singlePointCrossover(parent1, parent2):
    childsGenes1 = np.empty(len(parent1.genes))
    childsGenes2 =  np.empty(len(parent1.genes))
    point = randint(0, len(parent1.genes))
    for i in range(point):
        childsGenes1[i] = parent1.genes[i]
        childsGenes2[i] = parent2.genes[i]
    for i in range(point, len(parent1.genes)):
        childsGenes1[i] = parent2.genes[i]
        childsGenes2[i] = parent1.genes[i]
    return Individual(childsGenes1), Individual(childsGenes2)
    
def uniformCrossover(parent1, parent2):
    childsGenes = np.array(len(parent1.genes))
    for i in range(len(parent1.genes)):
        if parent1.genes[i]==parent2.genes[i]:
            childsGenes[i] = parent1.genes[i]
        else:
            f = getrandbits(1)
            childsGenes[i] = f * parent1.genes[i] + (1-f) * parent2.genes[i]
    
    return Individual(childsGenes), None
    
def arithmeticCrossover(parent1, parent2):
    childsGenes = np.empty(len(parent1.genes))
    for i in range(len(parent1.genes)):
        a = random()
        childsGenes[i] = a * parent1.genes[i] + (1-a) * parent2.genes[i]
    
    return Individual(childsGenes), None
    
def pomicnaTockaCrossover(parent1, parent2):
    pass
    
def binaryMutation(ll, ul, pm, orginal):
    for i in range(len(orginal.genes)):
        if random() < pm:
            orginal.genes[i] = 1 - orginal.genes
    
def gaussMutation(ll, ul, sigma, pm, orginal):
    orginal.genes = orginal.genes + np.random.normal(
            0, sigma, len(orginal.genes))
    
    np.maximum(orginal.genes, ll)
    np.minimum(orginal.genes, ul)
    
class FixedCount:
    
    def __init__(self, maxCount):
        self.maxCount = maxCount
        self.count = 0
        
    def isSatisfied(self):
        self.count += 1
        return self.count > self.maxCount
    
def GGA(f, selections, crossovers, mutations, population, stopCondition, elitism):

    popSize = len(population)
    for x in population:
        x.fitness = f(x)
        
    numGens = 0
    bestSoFar = max(population, key=lambda x : x.fitness)
    print(numGens, bestSoFar.fitness)
    
    while not stopCondition():
        nextPopulation = np.empty(popSize, dtype=Individual)
        
        if elitism:
            best = max(population, key=lambda x : x.fitness)
            nextPopulation[0] = best
            i = 1
        else:
            i = 0
          
        while i < popSize:
        
            selection = selections[randint(0, len(selections)-1)]
            crossover = crossovers[randint(0, len(crossovers)-1)]
            mutation = mutations[randint(0, len(mutations)-1)]
        
            parent1 = selection(population)
            parent2 = selection(population)
            child1, child2 = crossover(parent1, parent2)
            mutation(child1)
            child1.fitness = f(child1)
            nextPopulation[i] = child1; i += 1

            if child2 != None and i < popSize:
                mutation(child2)
                child2.fitness = f(child2)
                nextPopulation[i] = child2; i += 1
            
        population = nextPopulation
        numGens += 1
        if bestSoFar.fitness < max(population, key=lambda x : x.fitness).fitness:
            bestSoFar = max(population, key=lambda x : x.fitness)
            print(numGens, bestSoFar.fitness)
        
    return max(population, key=lambda x : x.fitness)
        

def EGA(f, crossovers, mutations, population, stopCondition):
    
    popSize = len(population)
    for x in population:
        x.fitness = f(x)
        
    numIters = 0
    bestSoFar = max(population, key=lambda x : x.fitness)
    print(numIters, bestSoFar.fitness)
    
    while not stopCondition():
    
        crossover = crossovers[randint(0, len(crossovers)-1)]
        mutation = mutations[randint(0, len(mutations)-1)]
    
        candidates = np.random.choice(
                        popSize, 3, replace=False)
                         
        argmin = np.argmin(map(
            lambda x : population[x].fitness, candidates))
            
        np.delete(candidates, argmin)
        
        child1, child2 = crossover(
            population[candidates[0]], population[candidates[1]])
        mutation(child1)
        child1.fitness = f(child1)
        population[argmin] = child1
        numIters += 1
        
        if bestSoFar.fitness < child1.fitness:
            bestSoFar = child1
            print(numIters, bestSoFar.fitness)
        
    return max(population, key=lambda x : x.fitness)
        
