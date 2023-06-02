import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_excel('distancias.xlsx')


cidades = df.columns[1:]


distancias = df.values[:,1:].astype(float)


# Convertendo a lista de listas em uma matriz NumPy para visualização mais fácil
distancias_np = np.array(distancias)

supermercados = [0, 1, 2, 3, 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]

adjacency_mat = distancias_np
def create_individual(supermercados):
    return random.sample(supermercados, len(supermercados))

def create_population(individuals, supermercados):
    return [create_individual(supermercados) for _ in range(individuals)]

def fitness(individual):
    return sum(adjacency_mat[individual[i]][individual[i+1]] for i in range(len(individual) - 1))

def mutate(individual):
    idx1, idx2 = random.sample(range(len(individual)), 2)
    individual[idx1], individual[idx2] = individual[idx2], individual[idx1]

def crossover(parent1, parent2):
    child = parent1[:len(parent1)//2]
    child += [item for item in parent2 if item not in child]
    return child

def genetic_algorithm(supermercados, pop_size, generations, mutation_rate):
    population = create_population(pop_size, supermercados)
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

def print_route_with_distances(route):
    total_distance = 0
    for i in range(len(route) - 1):
        distance = adjacency_mat[route[i]][route[i + 1]]
        total_distance += distance
        print(f"Distancia do supermercado {cidades[route[i]]}  pro supermercado {cidades[route[i+1]]}: {distance}")

    print(f"Total distance: {total_distance}")

result = genetic_algorithm(supermercados, 50, 1000, 0.05)
print(f"Rota Otimizada: {result}")
print(f"Distancia Otimizada: {fitness(result)}")
print_route_with_distances(result)