import tkinter
from tkinter import *
from tkinter import ttk
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random as rd
from scipy.optimize import dual_annealing
from matplotlib.figure import Figure
import math
import random
import pandas as pd
from copy import deepcopy


df = pd.read_excel('distancias.xlsx')


cidades = df.columns[1:]


distancias = df.values[:,1:].astype(float)


supermercados = [0, 1, 2, 3, 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]


def create_individual():
    df = pd.read_excel('distancias.xlsx')
    distancias = df.values[:, 1:].astype(float)
    distancias_np = np.array(distancias)
    np.random.shuffle(distancias_np)
    return deepcopy(distancias_np)


def create_population(population_size):
    return [create_individual() for _ in range(population_size)]


def fitness(individual):
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


def select_tournament(population, fitnesses, num_parents, tournament_size):
    parents = []

    for _ in range(num_parents):
        contenders = random.sample(list(zip(population, fitnesses)), tournament_size)

        winner = max(contenders, key=lambda x: x[1])[0]

        parents.append(deepcopy(winner))

    return parents



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
    population.sort(key=fitness)
    population[:len(new_individuals)] = deepcopy(new_individuals)

    return deepcopy(population)

def calculate_distance(fitness):
    return (-1 * fitness + 4000) / 5


def genetic_algorithm(population_size,num_generations,mutacao_probabilidade,tamanho_elitismo,probabilidade_de_cruzamento,selection_method,tamanho_torneio=None,):
    population = create_population(population_size)
    df = pd.read_excel('distancias.xlsx')
    global distancias
    distancias = df.values[:, 1:].astype(float)
    distancias = np.array(distancias)
    geracaoencontrada = 0
    best_fitness = 0
    for gen in range(num_generations):
        fitnesses = [fitness(individual) for individual in population]

        if selection_method == "tournament":
            parents = select_tournament(population, fitnesses, population_size // 2, tamanho_torneio)
        else:
            parents = select_roulette(population, fitnesses, population_size // 2)

        children = []
        for i in range(0, len(parents) - 1, 2):
            child1, child2 = ox_crossover(parents[i], parents[i + 1], probabilidade_de_cruzamento)
            children.append(child1)
            children.append(child2)
        if (tamanho_elitismo > 0):
            population.sort(key=fitness, reverse=True)
            elites = deepcopy(population[:tamanho_elitismo])
        mutated_children = [mutate(child, mutacao_probabilidade) for child in children]
        population = replace_population(population, mutated_children)

        if (tamanho_elitismo > 0):
            population.sort(key=fitness)
            population[:tamanho_elitismo] = deepcopy(elites)
        statistic_fitness = [fitness(individual) for individual in population]

        if (tamanho_elitismo > 0):
            fitness(elites[0])

        if max(statistic_fitness) > best_fitness:
            geracaoencontrada = gen
            best_fitness = max(statistic_fitness)
        # print('Geração: ', gen + 1, 'Fitness: ', max(statistic_fitness), 'Melhor Geração: ', geracaoencontrada + 1,
        #       'Distância total: ', calculate_distance(max(statistic_fitness)))

    melhor_geracao.config(text=f"Geração em que foi encontrado a melhor geração: {geracaoencontrada+1}")
    melhor_distancia.config(text=f"Melhor Distancia Encontrada: { calculate_distance(max(statistic_fitness))}")
    best_individual = max(population, key=fitness)


    km_percorrido = 0
    grande_array=np.full((21, 3), '', dtype=object)


    for i in range(len(best_individual) - 1):
        index_of = np.where(best_individual[i + 1] == 0)[0][0]
        grande_array[i][0]=cidades[np.where(best_individual[i] == 0)[0][0]]
        grande_array[i][1] = cidades[index_of]
        grande_array[i][2]=best_individual[i][index_of]
        print(f"Distancia do supermercado {cidades[np.where(best_individual[i] == 0)[0][0]]}  pro supermercado {cidades[index_of]}: distancia {best_individual[i][index_of]}")
        km_percorrido += best_individual[i][index_of]

    print(grande_array)

    root = tkinter.Tk()
    root.title('TABELA DE DESTINOS E AS SUAS DISTANCIAS')

    # Crie os cabeçalhos da tabela
    headers = ['Origem', 'Destino', 'Distancia']
    for i in range(3):
        tkinter.Label(root, text=headers[i], borderwidth=1, relief='solid').grid(row=0, column=i)



    for i in range(21):
        for j in range(3):
            tkinter.Label(root, text=grande_array[i][j], borderwidth=1, relief='solid').grid(row=i + 1, column=j)

    root.mainloop()

    print(f'Melhor Distancia {km_percorrido}')
    return best_individual




def submit_button_event():
    population_size = int(form_tamanho_da_populacao.get())
    probabilidade_de_cruzamento = float(form_probabilidade_de_cruzamento.get())
    mutacao_probabilidade = float(form_mutacao_probabilidade.get())
    num_generations = int(form_quantidade_geracoes.get())
    if check_var.get():
        tamanho_torneio = int(form_tamanho_torneio.get())
    else:
        tamanho_torneio = None
    tamanho_elitismo = int(form_tamanho_elitismo.get())
    tamanho_elitismo = int(form_tamanho_elitismo.get())
    selection_method = "tournament" if check_var.get() else "roulette"
    print("Tamanho da população:", population_size)
    print("Probabilidade de cruzamento:", probabilidade_de_cruzamento)
    print("Probabilidade de mutação:", mutacao_probabilidade)
    print("Quantidade de gerações:", num_generations)
    print("Tamanho do torneio:", tamanho_torneio)
    print("Tamanho do elitismo:", tamanho_elitismo)
    print("Selection method:", selection_method)
    schedule = genetic_algorithm(population_size,num_generations,mutacao_probabilidade,tamanho_elitismo,probabilidade_de_cruzamento,selection_method,tamanho_torneio)


def fill_form():
    form_tamanho_da_populacao.delete(0, tkinter.END)
    form_tamanho_da_populacao.insert(0, str(100))
    form_probabilidade_de_cruzamento.delete(0, tkinter.END)
    form_probabilidade_de_cruzamento.insert(0, str(0.85))
    form_mutacao_probabilidade.delete(0, tkinter.END)
    form_mutacao_probabilidade.insert(0, str(0.5))
    form_quantidade_geracoes.delete(0, tkinter.END)
    form_quantidade_geracoes.insert(0, str(100))
    form_tamanho_torneio.delete(0, tkinter.END)
    form_tamanho_torneio.insert(0, str(15))
    form_tamanho_elitismo.delete(0, tkinter.END)
    form_tamanho_elitismo.insert(0, str(3))

def toggle_torneio_visibility():
    if check_var.get():
        label_tamanho_torneio.place(x=100, y=200)
        form_tamanho_torneio.place(x=325, y=200)
    else:
        label_tamanho_torneio.place_forget()
        form_tamanho_torneio.place_forget()


window = Tk()
window.title("IA-TRABALHO-GRUPO-ESDRAS-JOAO-OTAVIO-FELIPE MENDES")
window.geometry('600x600')
window.configure(background="gray")

label_form=tkinter.Label(window,text="Dados das Entradas",background="gray")
label_tamanho_da_populacao = tkinter.Label(window, text="Tamanho da População:",background="gray")
label_probabilidade_de_cruzamento = tkinter.Label(window, text="Probabilidade de Cruzamento:",background="gray")
label_mutacao_probabilidade = tkinter.Label(window, text="Probabilidade de Mutação:",background="gray")
label_quantidade_geracoes = tkinter.Label(window, text="Quantidade de Geracoes:",background="gray")
label_tamanho_torneio =  tkinter.Label(window, text="Tamanho do Torneio:",background="gray")
label_tamanho_elitismo = tkinter.Label(window, text="Tamanho do Elitismo:", background="gray")
result_best_individual = tkinter.Label(window, text="Melhor Individuo Encontrado:")
result_function_value = tkinter.Label(window, text="Aptidão:")
melhor_geracao=tkinter.Label(window,text="Geração em que foi encontrado a melhor geração:")
melhor_distancia=tkinter.Label(window,text="Melhor Distancia Encontrada:")
resultadoreal=tkinter.Label(window,text="Valor Maximo da Função:")
porcentagem_de_erro=tkinter.Label(window,text="Porcentagem de erro entre o Máximo encontrado e o Maximo Real:")


form_tamanho_elitismo = tkinter.Entry()
form_tamanho_da_populacao=tkinter.Entry()
form_probabilidade_de_cruzamento=tkinter.Entry()
form_mutacao_probabilidade=tkinter.Entry()
form_quantidade_geracoes=tkinter.Entry()
form_tamanho_torneio=tkinter.Entry()


label_form.place(x=200,y=10)
label_tamanho_da_populacao.place(x=100,y=50)
form_tamanho_da_populacao.place(x=325,y=50)
label_probabilidade_de_cruzamento.place(x=100,y=80)
form_probabilidade_de_cruzamento.place(x=325,y=80)
label_quantidade_geracoes.place(x=100,y=110)
form_quantidade_geracoes.place(x=325,y=110)
label_mutacao_probabilidade.place(x=100,y=140)
form_mutacao_probabilidade.place(x=325,y=140)
label_tamanho_elitismo.place(x=100, y=170)
form_tamanho_elitismo.place(x=325, y=170)
result_best_individual.place(x=100, y=700)
result_function_value.place(x=100, y=750)
melhor_geracao.place(x=100,y=400)
melhor_distancia.place(x=100,y=430)
resultadoreal.place(x=100,y=850)
porcentagem_de_erro.place(x=100,y=900)
#CHECKBOX
check_var = tkinter.BooleanVar()
check_torneio = tkinter.Checkbutton(window, text="Torneio", variable=check_var, command=toggle_torneio_visibility, background="gray")
check_torneio.place(x=100, y=230)
label_tamanho_torneio.place_forget()
form_tamanho_torneio.place_forget()


#SUBMIT
submit_button = tkinter.Button(window, text="Submit", command=submit_button_event)
submit_button.place(x=350, y=300)

#BOTAO PREENCHER
preencher_button=tkinter.Button(window,text="To Fill",command=fill_form)
preencher_button.place(x=350,y=350)

window.mainloop()