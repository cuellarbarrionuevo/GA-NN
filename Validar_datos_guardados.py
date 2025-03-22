import matplotlib.pyplot as plt
import os
import pickle
from datos import *
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import ast
import pandas as pd
import math


# Ejemplo de diccionario con datos

fichero_backup = f"{train_dir.replace('/', '-')[-5:]}_{test_dir.replace('/', '-')[-5:]}.pkl"
if os.path.exists(fichero_backup):
    # Si el archivo existe, lo cargamos
    with open(fichero_backup, 'rb') as file:
        poblacion = pickle.load(file)
        evaluaciones_cache = pickle.load(file)
    print("Datos recuperados:")
#ast.literal_eval(evaluaciones_cache[mejor_clave][1])
datos = evaluaciones_cache

# Extraer claves y valores del diccionario
claves = list(datos.keys())
valores = list(datos.values())

# Procesar los datos
# Procesar los datos
x_data, y_data, z_data = [], [], []
for metric, config in valores:
    config_dict = ast.literal_eval(config)  # Convertir el string a diccionario
    num_filtros = len(config_dict['filtros'])  # Número de capas de filtros
    sum_filtros = sum(config_dict['filtros'])  # Suma de los filtros
    sum_kernel = sum([k[0] * k[1] for k in config_dict['kernel_size']])  # Suma de tamaños de kernel
    sum_max_pooling = sum([mp[0] * mp[1] for mp in config_dict['max_pooling']])  # Suma de tamaños de max pooling
    num_densa = len(config_dict['densa'])  # Número de capas densas
    sum_densa = sum(config_dict['densa'])  # Suma de neuronas en capas densas
    mean_drop = sum(config_dict['dropout_rate'])/len(config_dict['dropout_rate'])
    x_data.append(f"NF:{str(num_filtros).zfill(1)} | ΣFilt:{str(sum_filtros).zfill(3)} | ΣKernel:{str(sum_kernel).zfill(2)} | ΣMaxPool:{str(sum_max_pooling).zfill(2)}")
    y_data.append(f"Dens:{str(num_densa).zfill(1)} | ΣD:{str(sum_densa).zfill(3)}| x̅_Drop:{str(round(mean_drop,1))}")
    z_data.append(metric)

# Crear un DataFrame para el heatmap
df = pd.DataFrame({'X': x_data, 'Y': y_data, 'Métrica': z_data})
pivot_table = df.pivot_table(index='Y', columns='X', values='Métrica', aggfunc=np.mean)
pivot_table = pivot_table.sort_index(axis=0)  # Ordena las filas (índices)
pivot_table = pivot_table.sort_index(axis=1)  # Ordena las columnas

# Crear heatmap
plt.figure(figsize=(20, 14))  # Ampliar el tamaño de la figura
plt.subplots_adjust(left=0.2, bottom=0.4, right=0.95, top=0.95)
sns.heatmap(pivot_table, annot=True, cmap='coolwarm', fmt='.3f', linewidths=0.5)
plt.xlabel('Parámetros de Filtros y Kernel', fontsize=10, labelpad=20)
plt.ylabel('Parámetros de Capas Densas', fontsize=10, labelpad=20)
plt.title('Heatmap de Configuraciones vs. Métrica', fontsize=10)
plt.xticks(rotation=45, ha='right', fontsize=7)
plt.yticks(rotation=0, fontsize=7)
plt.show()