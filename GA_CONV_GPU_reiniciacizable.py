import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import random
import os
from datos import *
import time
import pickle

# Configurar la memoria de la GPU
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU configurada correctamente")
    except RuntimeError as e:
        print(e)

#No sacar mensajes
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

#Evitar run out of memory en CMD ..................................................................
# set TF_GPU_ALLOCATOR=cuda_malloc_async
#..................................................................................................

# Parametros
'''
epochs_v=5
tamano_poblacion_v = 5
generaciones_v = 5
'''

# Definir función para crear un modelo CNN con un número variable de capas
def crear_modelo(individuo):
    #########################################
    #print(individuo)
    #########################################
    model = Sequential()
    model.add(Conv2D(individuo["filtros"][0], kernel_size=individuo["kernel_size"][0], activation='relu', input_shape=(150, 150, 3),padding='same'))
    model.add(MaxPooling2D(pool_size=individuo["max_pooling"][0]))
    
    for i in range(1, len(individuo["filtros"])):
        model.add(Conv2D(individuo["filtros"][i], kernel_size=individuo["kernel_size"][i], activation='relu',padding='same'))
        model.add(MaxPooling2D(pool_size=individuo["max_pooling"][i]))
    
    model.add(Flatten())
    
    for i in range(len(individuo["densa"])):
        model.add(Dense(individuo["densa"][i], activation=individuo["activacion"]))
        model.add(Dropout(individuo["dropout_rate"][i]))
    
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Generar población inicial con un número variable de capas convolucionales y densas
def generar_poblacion(tamano):
    poblacion = []
    for _ in range(tamano):
        num_capas_conv = random.randint(1, 4)
        num_capas_densa = random.randint(1, 3)
        individuo = {
            "filtros": [random.choice([16, 32, 64]) for _ in range(num_capas_conv)],
            "kernel_size": [random.choice([(3,3), (5,5)]) for _ in range(num_capas_conv)],
            "max_pooling": [random.choice([(2,2), (3,3),(4,4)]) for _ in range(num_capas_conv)],
            #"dropout_rate": [random.uniform(0.2, 0.5) for _ in range(num_capas_densa)],
            "dropout_rate": [random.choice([0.1, 0.2, 0.3,0.4,0.5]) for _ in range(num_capas_densa)],
            "densa": [random.choice([64, 128, 256]) for _ in range(num_capas_densa)],
            "activacion": random.choice(["relu", "tanh"])
        }
        #Validamos se cumplan las condciones de factibilidad del individuo, sino buscamos otro
        while not(validar_poblacion(individuo)):
            individuo = {
            "filtros": [random.choice([16, 32, 64]) for _ in range(num_capas_conv)],
            "kernel_size": [random.choice([(3,3), (5,5)]) for _ in range(num_capas_conv)],
            "max_pooling": [random.choice([(2,2), (3,3),(4,4)]) for _ in range(num_capas_conv)],
            #"dropout_rate": [random.uniform(0.2, 0.5) for _ in range(num_capas_densa)],
            "dropout_rate": [random.choice([0.1, 0.2, 0.3,0.4,0.5]) for _ in range(num_capas_densa)],
            "densa": [random.choice([64, 128, 256]) for _ in range(num_capas_densa)],
            "activacion": random.choice(["relu", "tanh"])
                        }
        poblacion.append(individuo)
    return poblacion

# Evaluar fitness de cada modelo
def evaluar_fitness(individuo, train_gen, val_gen):
    clave = str(individuo)
    if clave in evaluaciones_cache:
        return evaluaciones_cache[clave][0]
    model = crear_modelo(individuo)
    #print(individuo)
    model.fit(train_gen, epochs=epochs_v, validation_data=val_gen, verbose=1)
    _, accuracy = model.evaluate(val_gen, verbose=1)
    #print(accuracy)
    evaluaciones_cache[clave] = (accuracy, clave)
    return accuracy

# Selección por torneo
def seleccion(poblacion, train_gen, val_gen):
    puntuaciones = [(ind, evaluar_fitness(ind, train_gen, val_gen)) for ind in poblacion]
    puntuaciones.sort(key=lambda x: x[1], reverse=True)
    #print(puntuaciones[0])
    return [ind[0] for ind in puntuaciones[:len(poblacion)//2]]


# Cruce y mutación
def cruzar(padre, madre):
    num_capas_conv = random.randint(1, max(len(padre["filtros"]), len(madre["filtros"])))
    num_capas_densa = random.randint(1, max(len(padre["densa"]), len(madre["densa"])))
    hijo = {
        "filtros": random.sample(padre["filtros"] + madre["filtros"], num_capas_conv),
        "kernel_size": random.sample(padre["kernel_size"] + madre["kernel_size"], num_capas_conv),
        "max_pooling": random.sample(padre["max_pooling"] + madre["max_pooling"], num_capas_conv),
        "dropout_rate": random.sample(padre["dropout_rate"] + madre["dropout_rate"], num_capas_densa),
        "densa": random.sample(padre["densa"] + madre["densa"], num_capas_densa),
        "activacion": random.choice([padre["activacion"], madre["activacion"]])
    }
    return hijo

def mutar(individuo):
    if random.random() < 0.5:
        individuo["filtros"].append(random.choice([16, 32, 64]))
        individuo["kernel_size"].append(random.choice([(3,3), (5,5)]))
        individuo["max_pooling"].append(random.choice([(2,2), (3,3), (4,4)]))
        #print("incrementar longitud",len(individuo["max_pooling"]))
        if not(validar_poblacion(individuo)): #Si no es valido anulamos la mutacion
                    #print("anulamos una mutacion que crece antes de reducir ",(individuo["max_pooling"]))
                    individuo["filtros"].pop()
                    individuo["kernel_size"].pop()
                    individuo["max_pooling"].pop()

    if random.random() < 0.2 and len(individuo["filtros"]) > 1:
        individuo["filtros"].pop()
        individuo["kernel_size"].pop()
        individuo["max_pooling"].pop()
    if random.random() < 0.2:
        individuo["densa"].append(random.choice([64, 128, 256]))
        individuo["dropout_rate"].append(random.choice([0.1, 0.2, 0.3,0.4,0.5]))
    if random.random() < 0.2 and len(individuo["densa"]) > 1:
        individuo["densa"].pop()
        individuo["dropout_rate"].pop()
    return individuo

# Evolución
def evolucionar(poblacion, generaciones, train_gen, val_gen):
    end_time = time.time()
    for _ in range(generaciones):
        start_time = end_time
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Generacion",str(_), f"Tiempo transcurrido: {elapsed_time:.2f} segundos", "Numero de poblacion explorado: ",len(evaluaciones_cache))
        seleccionados = seleccion(poblacion, train_gen, val_gen)
        nueva_poblacion = seleccionados[:]
        while len(nueva_poblacion) < len(poblacion):
            padre, madre = random.sample(seleccionados, 2)
            hijo = cruzar(padre, madre)
            if validar_poblacion(hijo):
                nueva_poblacion.append(mutar(hijo))
        poblacion = nueva_poblacion
        # Guardaos la cofiguracio hasta el 
        with open(fichero_backup, 'wb') as file:
            pickle.dump(poblacion[:len(poblacion)], file) 
            #pickle.dump(poblacion[:len(poblacion)], file) #Guardamos los mejores de la ultima evolucion
            pickle.dump(evaluaciones_cache, file)
            #pickle.dump(mejor_modelo, file)
            #return seleccionados[0]

def validar_poblacion(hijo):
    #valida que al hacer el max_pooling la imagen tiene por lo menos un tamaño de 4
    dimension = 150
    for parametro in hijo["max_pooling"]:
        dimension = dimension/parametro[0]
    if dimension >= 4:
        return True
    else:
        return False

# Cargar datos
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=32, class_mode='binary')
#train_gen = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=32, class_mode='binary')
val_gen = val_datagen.flow_from_directory(test_dir, target_size=(150, 150), batch_size=32, class_mode='binary')

# Ejecutar optimización
#tamano_poblacion = 10
tamano_poblacion = tamano_poblacion_v
generaciones = generaciones_v

# si existe un ejecucion previa continuamos
fichero_backup = f"{train_dir.replace('/', '-')[-5:]}_{test_dir.replace('/', '-')[-5:]}.pkl"
if os.path.exists(fichero_backup):
    # Si el archivo existe, lo cargamos
    with open(fichero_backup, 'rb') as file:
        poblacion = pickle.load(file)
        evaluaciones_cache = pickle.load(file)
    print("Datos recuperados:")
else:
    poblacion = generar_poblacion(tamano_poblacion)
    # Diccionario para almacenar evaluaciones previas
    evaluaciones_cache = {}

mejor_modelo = evolucionar(poblacion, generaciones, train_gen, val_gen)
print("Mejor configuración encontrada:", mejor_modelo)

#print(evaluaciones_cache)
# Imprimir solo los valores de accuracy
for valor in evaluaciones_cache.values():
    print(valor)
#Guardamos los datos claves poblaciones y diccionario
print(fichero_backup)
with open(fichero_backup, 'wb') as file:
    pickle.dump(poblacion[:len(poblacion)], file) 
    #pickle.dump(poblacion[:len(poblacion)], file) #Guardamos los mejores de la ultima evolucion
    pickle.dump(evaluaciones_cache, file)
    #pickle.dump(mejor_modelo, file)