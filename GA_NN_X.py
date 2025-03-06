import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import random


# Definir función para crear un modelo CNN

def crear_modelo(filtros, kernel_size, dropout_rate, dense_neuronas, activacion):
    model = Sequential([
        Conv2D(filtros, kernel_size=kernel_size, activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(pool_size=(2,2)),
        Flatten(),
        Dense(dense_neuronas, activation=activacion),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Generar población inicial

def generar_poblacion(tamano):
    poblacion = []
    for _ in range(tamano):
        individuo = {
            "filtros": random.choice([16, 32, 64]),
            "kernel_size": random.choice([(3,3), (5,5)]),
            "dropout_rate": random.uniform(0.2, 0.5),
            "dense_neuronas": random.choice([64, 128, 256]),
            "activacion": random.choice(["relu", "tanh"])
        }
        poblacion.append(individuo)
    return poblacion

# Evaluar fitness de cada modelo

def evaluar_fitness(individuo, train_gen, val_gen):
    model = crear_modelo(
        individuo["filtros"], individuo["kernel_size"], individuo["dropout_rate"], 
        individuo["dense_neuronas"], individuo["activacion"]
    )
    model.fit(train_gen, epochs=10, validation_data=val_gen, verbose=0)
    _, accuracy = model.evaluate(val_gen, verbose=0)
    print (individuo["filtros"], individuo["kernel_size"], individuo["dropout_rate"], 
        individuo["dense_neuronas"], individuo["activacion"], "Accuracy: ", accuracy)
    return accuracy

# Selección por torneo

def seleccion(poblacion, train_gen, val_gen):
    puntuaciones = [(ind, evaluar_fitness(ind, train_gen, val_gen)) for ind in poblacion]
    puntuaciones.sort(key=lambda x: x[1], reverse=True)
    #print(puntuaciones)
    return [ind[0] for ind in puntuaciones[:len(poblacion)//2]]

# Cruce y mutación

def cruzar(padre, madre):
    hijo = {
        "filtros": random.choice([padre["filtros"], madre["filtros"]]),
        "kernel_size": random.choice([padre["kernel_size"], madre["kernel_size"]]),
        "dropout_rate": random.choice([padre["dropout_rate"], madre["dropout_rate"]]),
        "dense_neuronas": random.choice([padre["dense_neuronas"], madre["dense_neuronas"]]),
        "activacion": random.choice([padre["activacion"], madre["activacion"]])
    }
    return hijo

def mutar(individuo):
    if random.random() < 0.2:
        individuo["filtros"] = random.choice([16, 32, 64])
    if random.random() < 0.2:
        individuo["kernel_size"] = random.choice([(3,3), (5,5)])
    return individuo

# Evolución

def evolucionar(poblacion, generaciones, train_gen, val_gen):
    for i in range(generaciones):
        print('Evolucion + ',i)
        seleccionados = seleccion(poblacion, train_gen, val_gen)
        nueva_poblacion = seleccionados[:]
        while len(nueva_poblacion) < len(poblacion):
            padre, madre = random.sample(seleccionados, 2)
            hijo = cruzar(padre, madre)
            nueva_poblacion.append(mutar(hijo))
        poblacion = nueva_poblacion
        print(seleccionados[0])
    return seleccionados[0]

# Cargar datos
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory('C:/Users/W10/chest_xray/train1', target_size=(150, 150), batch_size=32, class_mode='binary')
val_gen = val_datagen.flow_from_directory('C:/Users/W10/chest_xray/val1', target_size=(150, 150), batch_size=32, class_mode='binary')

# Ejecutar optimización
tamano_poblacion = 10
generaciones = 5
poblacion = generar_poblacion(tamano_poblacion)
mejor_modelo = evolucionar(poblacion, generaciones, train_gen, val_gen)
print("Mejor configuración encontrada:", mejor_modelo)
