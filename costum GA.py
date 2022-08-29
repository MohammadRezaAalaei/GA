
from random import uniform, seed, randint, choice, gauss, sample, shuffle
import matplotlib.pyplot as plt 
import pandas as pd
from time import sleep
import numpy as np
import random


# #### Define the problem 

# In[312]:


def loss_function(x):
    df = pd.DataFrame(x)
    df.to_csv('genomes.csv')
    sleep(0.7)
    df2 = pd.read_csv('loss.csv')
    loss = df2['loss'][0]
    return loss


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

        y = np.random.uniform(-alpha, 1+alpha, self.n_gene)

        offspring1 = []
        offspring2 = []
        offspring1 = self.genome*y + (1-y)*other.genome
        offspring2 = other.genome*y + (1-y)*self.genome

        child1 = Child(g=offspring1, n_gene=7, cost_function=loss_function)
        child2 = Child(g=offspring2, n_gene=7, cost_function=loss_function)

        return [child1 , child2]

    def mutate(self):
        temp_genome = self.genome[:]
        i = random.choice(range(n_gene))
        sigma = (self.gene_max[i] - self.gene_min[i])
        temp_genome[i] = min(max(temp_genome[i] + random.gauss(0, sigma), self.gene_min[i]), self.gene_max[i])
        return Child(g = temp_genome, cost_function=loss_function)


class Child(Particle):
    pass


# #### Define the GA

# In[314]:


class Genetic_algorithm():
    def __init__(self, gene_max=None, gene_min=None, cost_function=None, n_pop=100, n_crossover=70, n_mutation=20, n_generation=100,):
        if gene_max:
            self.n_gene = len(gene_max)
        self.n_mutation = n_mutation
        self.n_generation = n_generation
        self.pop = [Particle(gene_max=gene_max, gene_min=gene_min, cost_function=cost_function, n_gene=self.n_gene) for _ in range(n_pop)]
        self.n_pop = n_pop
        self.best_solution = []
        print('Initialization finished')


    def sort_population(self, x):
        return sorted(x, key = lambda t: t.fitness)



  

    def generate(self):
        """generates n_generation of particles to search for the optimal point"""

        for generation in range(self.n_generation):

            # selection & reproduction (crossover)
            popc = []
            candidates = random.sample(self.pop, n_crossover)
            random.shuffle(candidates)
            for i in range(0 , n_crossover-1 , 2):
                parents1 = candidates[i]
                parents2 = candidates[i+1]
                popc += parents1.crossover(parents2)

            # selection & mutation
            popm = []
            candidates = sample(self.pop, self.n_mutation)
            for i in candidates:
                popm.append(i.mutate())

            # merge & sort & truncate the population
            pop_overall = self.pop + popc + popm
            pop_overall = self.sort_population(pop_overall)

            self.pop = pop_overall[:self.n_pop]

            # save best results & show the information
            self.best_solution.append(self.pop[0])
            print (f"Generation {generation}: Best Solution = {self.pop[0].fitness}")


# #### define hyperparameters

# In[315]:


# problem
n_gene = 12
gene_min = [-5 for _ in range(n_gene)]
gene_max = [5 for _ in range(n_gene)]

gene_door_min = 0
gene_door_max = 1

gene_min.append(gene_door_min)
gene_max.append(gene_door_max)

# genetic
n_pop = 100
n_crossover = 70
n_mutation = 20

# termination condition
n_generation = 100




ga = Genetic_algorithm(
    gene_max = gene_max,
    gene_min = gene_min,
    cost_function = loss_function,
    n_pop = n_pop,
    n_crossover = n_crossover,
    n_mutation = n_mutation,
    n_generation = n_generation
    )


ga.generate()


# In[ ]:





# In[ ]:




