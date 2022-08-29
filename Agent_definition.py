

from random import uniform, seed, randint, choice, gauss, sample, shuffle
import matplotlib.pyplot as plt
import pandas as pd
from time import sleep
import numpy as np
import random
from Problem_definition import loss_function, hyper_parameters

# #### define the agent(particle)

# In[313]:


class Particle:
    def __init__(self, g='None', cost_function=None, gene_min=None, gene_max=None, n_gene=None):
        self.gene_max = gene_max
        self.gene_min = gene_min
        self.n_gene = n_gene
        if g != 'None':
            self.genome = g
        else:
            genome = [uniform(gene_min[i], gene_max[i]) for i in range(n_gene)]
            self.genome = np.array(genome)
        self.fitness = cost_function(self.genome)

    def crossover(self, other):
        # blend crossoever
        # child = y*parrent1 + (1-y)*parrent2
        # y = uniform(-alpha, 1+alpha)

        alpha = 0.5

        y = np.random.uniform(-alpha, 1 + alpha, self.n_gene)

        offspring1 = []
        offspring2 = []
        offspring1 = self.genome * y + (1 - y) * other.genome
        offspring2 = other.genome * y + (1 - y) * self.genome

        child1 = Child(g=offspring1, n_gene=7, cost_function=loss_function)
        child2 = Child(g=offspring2, n_gene=7, cost_function=loss_function)

        return [child1, child2]

    def mutate(self):
        temp_genome = self.genome[:]
        i = random.choice(range(hyper_parameters['n_gene']))
        sigma = (self.gene_max[i] - self.gene_min[i])
        temp_genome[i] = min(max(temp_genome[i] + random.gauss(0, sigma), self.gene_min[i]), self.gene_max[i])
        return Child(g=temp_genome, cost_function=loss_function)


class Child(Particle):
    pass
