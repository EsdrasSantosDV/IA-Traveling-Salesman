import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

# Ler o arquivo Excel
df = pd.read_excel('distancias.xlsx')

# Retirar o nome das cidades do dataframe
cidades = df.columns[1:]

# Convertendo o DataFrame em uma matriz de distâncias
distancias = df.values[:,1:].astype(float)


# Convertendo a lista de listas em uma matriz NumPy para visualização mais fácil
distancias_np = np.array(distancias)

cities = [0, 1, 2, 3, 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]

adjacency_mat = distancias_np
def create_individual(cities):
    return random.sample(cities, len(cities))

def create_population(individuals, cities):
    return [create_individual(cities) for _ in range(individuals)]

def fitness(individual):
    return sum(adjacency_mat[individual[i]][individual[i+1]] for i in range(len(individual) - 1))

def mutate(individual):
    idx1, idx2 = random.sample(range(len(individual)), 2)
    individual[idx1], individual[idx2] = individual[idx2], individual[idx1]

def crossover(parent1, parent2):
    child = parent1[:len(parent1)//2]
    child += [item for item in parent2 if item not in child]
    return child

def genetic_algorithm(cities, pop_size, generations, mutation_rate):
    population = create_population(pop_size, cities)
    population.sort(key=fitness)

    for _ in range(generations):
        new_population = []

        for i in range(0, pop_size, 2):
            parent1, parent2 = population[i], population[i+1]
            child1, child2 = crossover(parent1, parent2), crossover(parent2, parent1)

            if random.random() < mutation_rate:
                mutate(child1)
                mutate(child2)

            new_population += [child1, child2]

        population = sorted(new_population, key=fitness)

    return population[0]

result = genetic_algorithm(cities, 50, 1000, 0.05)
print(f"Optimized route: {result}")
print(f"Optimized distance: {fitness(result)}")