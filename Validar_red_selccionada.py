import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import cv2
import os
import pickle
import ast
import heapq


labels = ['PNEUMONIA', 'NORMAL']
img_size = 150
def get_training_data(data_dir):
    data = [] 
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return (data)            
    #return np.array(data)

# Definir función para crear un modelo CNN con un número variable de capas
def crear_modelo(individuo):
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

def selec_ind_pobla_hist():
# si existe un ejecucion previa continuamos
    fichero_backup = f"{train_dir.replace('/', '-')[-5:]}_{test_dir.replace('/', '-')[-5:]}.pkl"
    if os.path.exists(fichero_backup):
        # Si el archivo existe, lo cargamos
        with open(fichero_backup, 'rb') as file:
            poblacion = pickle.load(file)
            evaluaciones_cache = pickle.load(file)
            #mejor_clave = max(evaluaciones_cache, key=lambda k: evaluaciones_cache[k][0])
            # Obtener las dos mejores claves directamente
            mejores = heapq.nlargest(2, evaluaciones_cache.keys(), key=lambda k: evaluaciones_cache[k][0])
            mejor_clave = mejores[1] if len(mejores) > 1 else None
            return ast.literal_eval(evaluaciones_cache[mejor_clave][1]) #convertimos a diccionario el codigo de la red,q ue viene en texto
    else:
            return({'filtros': [32, 64, 32], 'kernel_size': [(5, 5), (3, 3), (5, 5)], 'max_pooling': [(3, 3), (2, 2), (2, 2)], 'dropout_rate': [0.5, 0.5, 0.2], 'densa': [256, 128, 128], 'activacion': 'relu'})
#Loading dataset

from datos import *
train = get_training_data(train_dir)
test = get_training_data(test_dir)
val = get_training_data(val_dir)

# Visualizacion
df = pd.DataFrame(train, columns=['Image', 'Label'])

df['Class'] = df['Label'].map({0: 'Pneumonia', 1: 'Normal'})

sns.set_style('darkgrid')
sns.countplot(data=df, x='Class')
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()  

# Visualizacion
df = pd.DataFrame(test, columns=['Image', 'Label'])

df['Class'] = df['Label'].map({0: 'Pneumonia', 1: 'Normal'})

sns.set_style('darkgrid')
sns.countplot(data=df, x='Class')
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show() 


x_train = []
y_train = []

x_val = []
y_val = []

x_test = []
y_test = []

for feature, label in train:
    x_train.append(feature)
    y_train.append(label)

for feature, label in test:
    x_test.append(feature)
    y_test.append(label)
    
for feature, label in val:
    x_val.append(feature)
    y_val.append(label)

 # Normalize the data
x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255
x_test = np.array(x_test) / 255  

# resize data for deep learning 
x_train = x_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)

x_val = x_val.reshape(-1, img_size, img_size, 1)
y_val = np.array(y_val)

x_test = x_test.reshape(-1, img_size, img_size, 1)
y_test = np.array(y_test)


# With data augmentation to prevent overfitting and handling the imbalance in dataset

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = True,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(x_train)

#Training model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential()
'''
array([[383,   7],
       [ 67, 167]]

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
'''

'''
[[381   9]
 [ 31 203]]
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 1), padding= 'same'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu', padding= 'same'))
model.add(MaxPooling2D((4, 4)))
model.add(Conv2D(64, (5, 5), activation='relu', padding= 'same'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu', padding= 'same'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='tanh'))
model.add(Dropout(0.4))
model.add(Dense(128, activation='tanh'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
'''
'''
[[379  11]
 [ 46 188]]

model.add(Conv2D(16, (5, 5), activation='relu', input_shape=(150, 150, 1), padding= 'same'))
model.add(MaxPooling2D((3, 3)))
model.add(Conv2D(16, (5, 5), activation='relu', padding= 'same'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(16, (5, 5), activation='relu', padding= 'same'))
model.add(MaxPooling2D((3, 3)))
model.add(Conv2D(32, (3, 3), activation='relu', padding= 'same'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='tanh'))
model.add(Dropout(0.4))
model.add(Dense(64, activation='tanh'))
model.add(Dropout(0.4))
model.add(Dense(128, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
'''
model = crear_modelo(selec_ind_pobla_hist())

# Model summary
model.summary()

#history = model.fit(datagen.flow(x_train,y_train, batch_size = 32) ,epochs =50 , validation_data = datagen.flow(x_val, y_val) )
history = model.fit(datagen.flow(x_train,y_train, batch_size = 32) ,epochs =30 , validation_data = datagen.flow(x_test, y_test) )

#Analisys after modeling
epochs = list(range(1, len(history.history['accuracy']) + 1))

train_acc = history.history['accuracy']
train_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']

# Create a figure
plt.figure(figsize=(20, 10))

# Plot training and validation accuracy
plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
plt.plot(epochs, train_acc, 'go-', label='Training Accuracy')
plt.plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
plt.title('Training & Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# Plot training and validation loss
plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
plt.plot(epochs, train_loss, 'g-o', label='Training Loss')
plt.plot(epochs, val_loss, 'r-o', label='Validation Loss')
plt.title('Training & Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# Display the plots
plt.tight_layout()
plt.show()

## Predict : 
predictions = model.predict(x_test)
predictions = np.where(predictions>0.5,1,0)
#print(predictions,y_test)

print(classification_report(y_test, predictions, target_names = ['Pneumonia (Class 0)','Normal (Class 1)']))

cm = confusion_matrix(y_test,predictions)
print(cm)

cm = pd.DataFrame(cm , index = ['0','1'] , columns = ['0','1'])

plt.figure(figsize = (10,10))
sns.heatmap(cm,cmap= "Blues", linecolor = 'black' , linewidth = 1 , annot = True, fmt='',xticklabels = labels,yticklabels = labels)
#.............................................................................
# Imagenes mal clasificadas
#.............................................................................

# Obtener predicciones (probabilidades entre 0 y 1)
y_pred_probs = model.predict(x_test)

# Convertir probabilidades a etiquetas (0 o 1)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()  # Asegurar formato adecuado
y_true = y_test.flatten()  # Si `y_test` está en formato 1D

# Encontrar índices de imágenes mal clasificadas
misclassified_indices = np.where(y_pred != y_true)[0]

# Mostrar cuántas imágenes fueron mal clasificadas
print(f"Total de imágenes mal clasificadas: {len(misclassified_indices)}")

# Número de imágenes a visualizar
num_images = min(80, len(misclassified_indices))

plt.figure(figsize=(12, 6))
for i, index in enumerate(misclassified_indices[:num_images]):
    plt.subplot(16, 5, i + 1)
    plt.imshow(x_test[index], cmap='gray')  # Asegurar escala de grises
    plt.title(f"Real: {y_true[index]} / Pred: {y_pred[index]} / IMG: {index}")
    plt.axis("off")

plt.tight_layout()
plt.show()