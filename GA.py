# #### Define the GA
from random import sample
import random
from Agent_definition import Particle, Child
from Problem_definition import hyper_parameters


class Genetic_algorithm():
    def __init__(self, gene_max=None, gene_min=None, cost_function=None, n_pop=100, n_crossover=70, n_mutation=20,
                 n_generation=100, ):
        if gene_max:
            self.n_gene = len(gene_max)
        self.n_mutation = n_mutation
        self.n_generation = n_generation
        self.pop = [Particle(gene_max=gene_max, gene_min=gene_min, cost_function=cost_function, n_gene=self.n_gene) for
                    _ in range(n_pop)]
        self.n_pop = n_pop
        self.best_solution = []
        print('Initialization finished')

    def sort_population(self, x):
        return sorted(x, key=lambda t: t.fitness)

    def generate(self):
        """generates n_generation of particles to search for the optimal point"""

        for generation in range(self.n_generation):

            # selection & reproduction (crossover)
            popc = []
            candidates = random.sample(self.pop, hyper_parameters['n_crossover'])
            random.shuffle(candidates)
            for i in range(0, hyper_parameters['n_crossover'] - 1, 2):
                parents1 = candidates[i]
                parents2 = candidates[i + 1]
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
            print(f"Generation {generation}: Best Solution = {self.pop[0].fitness}")


ga = Genetic_algorithm(
    gene_max = hyper_parameters['gene_max'],
    gene_min = hyper_parameters['gene_min'],
    cost_function = hyper_parameters['loss_function'],
    n_pop = hyper_parameters['n_pop'],
    n_crossover = hyper_parameters['n_crossover'],
    n_mutation = hyper_parameters['n_mutation'],
    n_generation = hyper_parameters['n_generation']
    )


ga.generate()