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
    form_tamanho_elitismo.insert(0, str(10))
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