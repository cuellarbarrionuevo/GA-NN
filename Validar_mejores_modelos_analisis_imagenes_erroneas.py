import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import ast
from tensorflow import keras
from tensorflow.keras import layers
from skimage.io import imread
from skimage.transform import resize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import heapq

# Directorios de imágenes (modifica con la ruta real)
BASE_DIR = "C:/Users/W10/chest_xray"
TRAIN_DIR = (BASE_DIR + "/train1")
TEST_DIR = (BASE_DIR + "/test1")
CATEGORIES = ["NORMAL", "PNEUMONIA"]  # Carpetas de imágenes
IMG_SIZE = (150, 150, 1)  # Tamaño uniforme para las imágenes

def load_data(directory):
    data, labels, file_paths = [], [], []
    for label, category in enumerate(CATEGORIES):
        folder = (directory + "/"+category)
        i=0
        for filename in os.listdir(folder):
            i+=1
            img_path = (folder+"/"+ filename)
            img = imread(img_path, as_gray=True)  # Cargar en escala de grises
            img_resized = resize(img, IMG_SIZE[:2], anti_aliasing=True)
            data.append(img_resized.reshape(IMG_SIZE))  # Asegurar forma adecuada
            labels.append(label)
            file_paths.append(img_path)
            print(i)
    return np.array(data, dtype=np.float32), np.array(labels), file_paths

# Cargar datos
X_train, y_train, paths_train = load_data(TRAIN_DIR)
X_test, y_test, paths_test = load_data(TEST_DIR)

# Normalizar imágenes
X_train /= 255.0
X_test /= 255.0

############################################
def pobla_hist():
# si existe un ejecucion previa continuamos
    #fichero_backup = f"{TRAIN_DIR.replace('/', '-')[-5:]}_{TEST_DIR.replace('/', '-')[-5:]}.pkl"
    fichero_backup = "train_-test.pkl"
    if os.path.exists(fichero_backup):
        # Si el archivo existe, lo cargamos
        with open(fichero_backup, 'rb') as file:
            poblacion = pickle.load(file)
            evaluaciones_cache = pickle.load(file)
            #mejor_clave = max(evaluaciones_cache, key=lambda k: evaluaciones_cache[k][0])
            # Obtener las dos mejores claves directamente
            #mejores = heapq.nlargest(2, evaluaciones_cache.keys(), key=lambda k: evaluaciones_cache[k][0])
            #mejor_clave = mejores[1] if len(mejores) > 1 else None
            #return ast.literal_eval(evaluaciones_cache[mejor_clave][1]) #convertimos a diccionario el codigo de la red,q ue viene en texto
            evaluaciones_cache = heapq.nlargest(2, evaluaciones_cache.keys(), key=lambda k: evaluaciones_cache[k][0])
            return evaluaciones_cache
    else:
            return({'filtros': [32, 64, 32], 'kernel_size': [(5, 5), (3, 3), (5, 5)], 'max_pooling': [(3, 3), (2, 2), (2, 2)], 'dropout_rate': [0.5, 0.5, 0.2], 'densa': [256, 128, 128], 'activacion': 'relu'})


# Función para construir modelos CNN con diferentes configuraciones
def build_model(individuo):
    #########################################
    #print(individuo)
    #########################################
    model = Sequential()
    model.add(Conv2D(individuo["filtros"][0], kernel_size=individuo["kernel_size"][0], activation='relu', input_shape=(150, 150, 1),padding='same'))
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
'''
def build_model(conv_layers=2, dense_units=128, dropout_rate=0.5):
    model = keras.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=IMG_SIZE))
    model.add(layers.MaxPooling2D((2, 2)))
    
    for _ in range(conv_layers - 1):
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(dense_units, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(1, activation='sigmoid'))  # Salida binaria
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
'''
# Definir diferentes configuraciones de modelo
model_configs=pobla_hist()
'''
model_configs = [
    {"conv_layers": 2, "dense_units": 128, "dropout_rate": 0.5},
    {"conv_layers": 3, "dense_units": 256, "dropout_rate": 0.3},
    {"conv_layers": 4, "dense_units": 512, "dropout_rate": 0.2}
]
'''
results = {}
misclassified_counts = {path: 0 for path in paths_test}

for i, config in enumerate(model_configs):
    print(f"Entrenando modelo {i+1} con config: {config}")
    model = build_model(ast.literal_eval(config))
    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32, verbose=1)
    print(f"Modelo {i+1} entrenado")
    # Evaluar modelo
    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    test_acc = np.mean(y_pred == y_test)
    
    results[f"Modelo {i+1}"] = {
        "config": config,
        "test_accuracy": test_acc,
        "history": history.history
    }
    
    # Contar errores por imagen
    for true_label, pred_label, path in zip(y_test, y_pred, paths_test):
        if true_label != pred_label:
            misclassified_counts[path] += 1

# Mostrar ranking de imágenes mal clasificadas
sorted_misclassified = sorted(misclassified_counts.items(), key=lambda x: x[1], reverse=True)
print("\nImágenes más mal clasificadas:")
for path, count in sorted_misclassified[:10]:
    print(f"{path} - Errores: {count}")
    img = imread(path, as_gray=True)
    plt.imshow(img, cmap='gray')
    plt.title(f"Errores: {count}")
    plt.axis('off')
    plt.show()

# Graficar precisión de cada modelo
for model_name, result in results.items():
    plt.plot(result["history"]["accuracy"], label=f'{model_name} Train')
    plt.plot(result["history"]["val_accuracy"], label=f'{model_name} Test')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

