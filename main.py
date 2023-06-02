import pandas as pd
import numpy as np

# Ler o arquivo Excel
df = pd.read_excel('distancias.xlsx')

# Retirar o nome das supermercados_excel do dataframe
supermercados_excel = df.columns[1:]

# Convertendo o DataFrame em uma matriz de distâncias
distancias = df.values[:,1:].astype(float)


# Convertendo a lista de listas em uma matriz NumPy para visualização mais fácil
distancias_np = np.array(distancias)

print(distancias_np)













