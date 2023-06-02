import random
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import *
from tkinter.ttk import Combobox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from copy import deepcopy

# Variáveis default
POPULATION_SIZE = 10
MUTATION_RATE = 0.05
CROSSOVER_RATE = 0.85
GENERATION_NUMBER = 100
ELITISM_LEN = 1
IS_TOURNAMENT = True
TOURNAMENT_SIZE = 5
cidades = []
CROSSOVER_TYPE = 'ox'


def create_individual():
    df = pd.read_excel('distancias.xlsx')
    distancias = df.values[:, 1:].astype(float)
    distancias_np = np.array(distancias)
    np.random.shuffle(distancias_np)
    return deepcopy(distancias_np)


# Criar a população
def create_population(population_size):
    return [create_individual() for _ in range(population_size)]


# Função de fitness
def fitness(individual, print_params=False):
    km_percorrido = 0
    for i in range(len(individual) - 1):
        index_of = np.where(individual[i + 1] == 0)[0][0]
        km_percorrido += individual[i][index_of]

    return 4000 - (km_percorrido * 5)


def select_roulette(population, fitnesses, num_parents):
    total_fitness = sum(fitnesses)
    probs = [f / total_fitness for f in fitnesses]

    parents = []

    for _ in range(num_parents):
        r = random.random()  # Gera um número aleatório entre 0 e 1
        for i, individual in enumerate(population):
            r -= probs[i]
            if r <= 0:
                parents.append(deepcopy(individual))
                break

    return parents


def select_tournament(population, fitnesses, num_parents, tournament_size=TOURNAMENT_SIZE):
    parents = []

    for _ in range(num_parents):
        # Seleciona 'tournament_size' indivíduos aleatoriamente
        contenders = random.sample(list(zip(population, fitnesses)), tournament_size)

        # Escolhe o indivíduo com maior fitness
        winner = max(contenders, key=lambda x: x[1])[0]

        parents.append(deepcopy(winner))

    return parents


def pmx_crossover(parent1, parent2, crossover_rate):
    # Escolher um ponto de cruzamento aleatório
    should_cross = random.random()
    if should_cross < crossover_rate:
        child1 = np.concatenate((parent1[0: 9], parent2[9: 16], parent1[16: 22]))
        child2 = np.concatenate((parent2[0: 9], parent1[9: 16], parent2[16: 22]))
        remove_repeated(child1)
        remove_repeated(child2)
        return deepcopy(child1), deepcopy(child2)
    else:
        return parent1, parent2


def cx_crossover(parent1, parent2, crossover_rate):
    should_cross = random.random()
    if should_cross < crossover_rate:
        cycles = [-1] * len(parent1)
        cycle_no = 1
        cyclestart = (i for i, v in enumerate(cycles) if v < 0)

        for pos in cyclestart:

            while cycles[pos] < 0:
                cycles[pos] = cycle_no
                mask = np.all(parent1 == parent2[pos], axis=1)
                pos = np.where(mask)[0][0]

            cycle_no += 1

        child1 = np.array([parent1[i] if n % 2 else parent2[i] for i, n in enumerate(cycles)])
        child2 = np.array([parent2[i] if n % 2 else parent1[i] for i, n in enumerate(cycles)])

        return deepcopy(child1), deepcopy(child2)
    else:
        return parent1, parent2


def ox_crossover(parent1, parent2, crossover_rate):
    should_cross = random.random()
    if should_cross < crossover_rate:
        size = len(parent1)

        start, end = sorted(random.sample(range(size), 2))

        child1, child2 = np.array([[-1.0] * size] * size), np.array([[-1.0] * size] * size)

        child1[start:end] = parent1[start:end]
        child2[start:end] = parent2[start:end]

        for p1, p2 in zip(np.concatenate((parent1[end:], parent1[:end]), axis=0),
                          np.concatenate((parent2[end:], parent2[:end]), axis=0)):
            mask1 = np.all(child1 == p1, axis=1)
            if len(np.where(mask1)[0]) == 0:
                child1[np.where(child1 == -1)[0][0]] = p1

            mask2 = np.all(child2 == p2, axis=1)
            if len(np.where(mask2)[0]) == 0:
                child2[np.where(child2 == -1)[0][0]] = p2

        return child1, child2
    else:
        return parent1, parent2


def remove_repeated(child):
    abc = []
    contador = {i: 0 for i in range(22)}
    for array in child:
        abc.append(np.where(array == 0)[0][0])
        contador[np.where(array == 0)[0][0]] += 1
    for i in abc:
        if contador[i] > 1:
            indice_escolhido = zero_position(contador)
            contador[i] -= 1
            child[i] = distancias[indice_escolhido]
            contador[indice_escolhido] += 1

    return child
    print('a')


def zero_position(contador):
    for k, v in contador.items():
        if v == 0:
            return k


def mutate(individual, mutation_rate):
    individual_copy = deepcopy(individual)
    for _ in range(22):
        should_mutate = random.random()
        if should_mutate < mutation_rate:
            j_chosen1 = random.randint(0, 21)
            j_chosen2 = random.randint(0, 21)
            individual_copy[[j_chosen1, j_chosen2]] = individual_copy[[j_chosen2, j_chosen1]]
    return deepcopy(individual_copy)


def replace_population(population, new_individuals):
    # Substitui os indivíduos menos aptos pelos novos indivíduos
    population.sort(key=fitness)
    population[:len(new_individuals)] = deepcopy(new_individuals)

    return deepcopy(population)


def calculate_distance(fitness):
    return (-1 * fitness + 4000) / 5


def genetic_algorithm():
    # Cria a população inicial

    print()
    print()
    print()
    print("==================================================")

    fig1, ax1 = plt.subplots(figsize=(5, 4))
    canvas1 = FigureCanvasTkAgg(fig1, master=input_frame)
    canvas1.get_tk_widget().grid(row=0, column=3, rowspan=10, pady=5, sticky='E')
    canvas1.get_tk_widget().delete('all')

    population_size = int(population_size_entry.get())
    num_generations = int(generation_amount_entry.get())

    tournament = bool(selection_method_entry.get()) != True
    tournament_size = None
    if (tournament):
        tournament_size = int(tournament_size_entry.get())

    crossover_value = float(crossover_entry.get())
    elitism = int(elitism_size_entry.get())
    mutation = float(mutation_probability_entry.get())
    crossover_type = CROSSOVER_TYPE

    population = create_population(population_size)
    df = pd.read_excel('distancias.xlsx')
    global distancias
    distancias = df.values[:, 1:].astype(float)
    distancias = np.array(distancias)
    best_gen = 0
    best_fitness = 0

    x = []
    best_array = []
    avg_array = []
    worst_array = []

    for gen in range(num_generations):
        # Calcula o fitness de cada indivíduo na população
        fitnesses = [fitness(individual) for individual in population]

        # Seleciona os pais
        if tournament:
            parents = select_tournament(population, fitnesses, population_size // 2, tournament_size)
        else:
            parents = select_roulette(population, fitnesses, population_size // 2)

        # Gera os filhos por cruzamento
        children = []
        for i in range(0, len(parents) - 1, 2):
            if crossover_type == 'pmx':
                child1, child2 = pmx_crossover(parents[i], parents[i + 1], crossover_value)
            elif crossover_type == 'ox':
                child1, child2 = ox_crossover(parents[i], parents[i + 1], crossover_value)
            elif crossover_type == 'cx':
                child1, child2 = cx_crossover(parents[i], parents[i + 1], crossover_value)

            children.append(child1)
            children.append(child2)

        # Implementação do elitismo
        if (elitism > 0):
            population.sort(key=fitness, reverse=True)  # Ordena a população pelo fitness (de maior para menor)
            elites = deepcopy(population[:elitism])

        # Aplica a mutação nos filhos
        mutated_children = [mutate(child, mutation) for child in children]

        # Substitui a população
        population = replace_population(population, mutated_children)

        # Insere os melhores indivíduos na população
        if (elitism > 0):
            population.sort(key=fitness)
            population[:elitism] = deepcopy(elites)

        statistic_fitness = [fitness(individual) for individual in population]

        if (elitism > 0):
            fitness(elites[0], True)

        if max(statistic_fitness) > best_fitness:
            best_gen = gen
            var_best_individual_generation.set(f'{best_gen + 1}º')
            best_fitness = max(statistic_fitness)

        best_array.append(max(statistic_fitness))
        avg_array.append(sum(statistic_fitness) / len(statistic_fitness))
        worst_array.append(min(statistic_fitness))
        x.append(gen)

        print('Geração: ', gen + 1, 'Fitness: ', max(statistic_fitness), 'Melhor Geração: ', best_gen + 1,
              'Distância total: ', calculate_distance(max(statistic_fitness)), 'km')





    best_individual = max(population, key=fitness)
    return best_individual


def generate_schedule():
    schedule = genetic_algorithm()


root = Tk()
root.title('Organizador de horários')
root.resizable(False, False)

input_frame = Frame(root)  # Cria um Frame para agrupar os widgets
input_frame.pack(padx=10, pady=10)

###### POPULATION SIZE ###################################################################################

population_size_label = Label(
    input_frame, text="Tamanho da População: ", font=("Helvetica", 13))
population_size_label.grid(
    row=0, column=0, padx=5, pady=5, sticky="w")

population_size_entry = Entry(
    input_frame, font=("Helvetica", 13))
population_size_entry.grid(
    row=0, column=1, padx=5, pady=5, sticky="w")

population_size_entry.insert(0, POPULATION_SIZE)

###### MUTATION PROBABILITY ##############################################################################

mutation_probability_label = Label(
    input_frame, text="Probabilidade de Mutação: ", font=("Helvetica", 13))
mutation_probability_label.grid(
    row=1, column=0, padx=5, pady=5, sticky="w")

mutation_probability_entry = Entry(
    input_frame, font=("Helvetica", 13))
mutation_probability_entry.grid(
    row=1, column=1, padx=5, pady=5, sticky="w")

mutation_probability_entry.insert(0, MUTATION_RATE)

###### CROSSOVER ########################################################################################

crossover_label = Label(
    input_frame, text="Taxa de Cruzamento: ", font=("Helvetica", 13))
crossover_label.grid(
    row=2, column=0, padx=5, pady=5, sticky="w")

crossover_entry = Entry(
    input_frame, font=("Helvetica", 13))
crossover_entry.grid(
    row=2, column=1, padx=5, pady=5, sticky="w")

crossover_entry.insert(0, CROSSOVER_RATE)

###### GENERATION AMOUNT #################################################################################

generation_amount_label = Label(
    input_frame, text="Quantidade de gerações: ", font=("Helvetica", 13))
generation_amount_label.grid(
    row=3, column=0, padx=5, pady=5, sticky="w")

generation_amount_entry = Entry(
    input_frame, font=("Helvetica", 13))
generation_amount_entry.grid(
    row=3, column=1, padx=5, pady=5, sticky="w")

generation_amount_entry.insert(0, GENERATION_NUMBER)

###### ELITISM ###########################################################################################

elitism_size_label = Label(
    input_frame, text="Tamanho do Elitismo: ", font=("Helvetica", 13))
elitism_size_label.grid(
    row=4, column=0, padx=5, pady=5, sticky="w")

elitism_size_entry = Entry(
    input_frame, font=("Helvetica", 13))
elitism_size_entry.grid(
    row=4, column=1, padx=5, pady=5, sticky="w")

elitism_size_entry.insert(0, ELITISM_LEN)

###### SELECTION METHOD ##################################################################################

selection_method_label = Label(
    input_frame, text="Método de Seleção: ", font=("Helvetica", 13))
selection_method_label.grid(
    row=6, column=0, padx=5, pady=5, sticky="w")

selection_method_entry = Combobox(
    input_frame, values=["Roleta", "Torneio"], font=("Helvetica", 13))
selection_method_entry.grid(
    row=6, column=1, padx=5, pady=5, sticky="w")

tournament_size_entry = None  # Declarando a variável global inicialmente
tournament_size_label = None  # Declarando a variável global inicialmente


def on_selection_change(event):
    global tournament_size_entry  # Declarando a variável como global para acessá-la fora da função
    global tournament_size_label  # Declarando a variável como global para acessá-la fora da função

    if selection_method_entry.get() == "Torneio":
        tournament_size_label = Label(
            input_frame, text="Tamanho do torneio: ", font=("Helvetica", 13))
        tournament_size_label.grid(
            row=7, column=0, padx=5, pady=5, sticky="w")
        tournament_size_entry = Entry(
            input_frame, font=("Helvetica", 13))
        tournament_size_entry.grid(
            row=7, column=1, padx=5, pady=5, sticky="w")

        tournament_size_entry.insert(0, TOURNAMENT_SIZE)

    if selection_method_entry.get() == "Roleta":
        if tournament_size_label:
            tournament_size_label.grid_forget()
            tournament_size_label = None
        if tournament_size_entry:
            tournament_size_entry.grid_forget()
            tournament_size_entry = None


selection_method_entry.bind("<<ComboboxSelected>>", on_selection_change)

selection_method_entry.current(1 if IS_TOURNAMENT else 0)
on_selection_change(None)

###### GENERATIONS #######################################################################################

generations_label = Label(
    input_frame, text="Geração Atual: ", font=("Helvetica", 13))
generations_label.grid(row=8, column=0, padx=5, pady=5, sticky="w")

var_generations = StringVar()
var_generations.set(0)

generations = Label(
    input_frame, textvariable=var_generations, font=("Helvetica", 13))
generations.grid(row=8, column=1, padx=5, pady=5, sticky="w")

best_individual_generation_label = Label(
    input_frame, text="Geração do Melhor Indivíduo: ", font=("Helvetica", 13))
best_individual_generation_label.grid(row=9, column=0, padx=5, pady=5, sticky="w")

var_best_individual_generation = StringVar()
var_best_individual_generation.set(0)

best_individual_generation_entry = Label(
    input_frame, textvariable=var_best_individual_generation, font=("Helvetica", 13))
best_individual_generation_entry.grid(row=9, column=1, padx=5, pady=5, sticky="w")

# canGenerate = generations.get()
# population_size_entry
# mutation_probability_entry
# crossover_entry
# generation_amount_entry
# elitism_size_entry
# selection_method_entry
# tournament_size_entry


generate_button = Button(input_frame, text="Aproximar", font=(
    "Helvetica", 12), command=generate_schedule)
generate_button.grid(row=12, column=0, columnspan=2, pady=5)

root.mainloop()
# data frame da lib pandas pelo tkinter visual