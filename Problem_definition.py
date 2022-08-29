
# Defining the problem
import pandas as pd
from time import sleep

def loss_function(x):
    """loss_function receives particles genomes and returns the its loss value"""
    df = pd.DataFrame(x)
    df.to_csv('genomes.csv')
    sleep(0.7)
    df2 = pd.read_csv('loss.csv')
    loss = df2['loss'][0]
    return loss

# Define hyperparameters
# problem

n_gene = 13
gene_min = [-5 for _ in range(n_gene)]
gene_max = [5 for _ in range(n_gene)]
# genetic
n_pop = 100
n_crossover = 70
n_mutation = 20
# termination condition
n_generation = 100

hyper_parameters = {
    'n_gene':n_gene,
    'gene_min':gene_min,
    'gene_max':gene_max,
    'n_pop': n_pop,
    'n_crossover':n_crossover,
    'n_mutation':n_mutation,
    'n_generation':n_generation
}