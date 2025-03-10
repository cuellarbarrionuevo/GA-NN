import tensorflow as tf
from tensorflow import keras

tf.__version__

#from keras.preprocessing.image import ImageDataGenerator

Train= tf.keras.utils.image_dataset_from_directory(    
    directory = 'C:/Users/W10/chest_xray/train1',
    labels = 'inferred',
    label_mode = 'categorical',
    batch_size=32,
    image_size = (256,256)
)
Test=tf.keras.utils.image_dataset_from_directory(
    directory = 'C:/Users/W10/chest_xray/train1',
    labels = 'inferred',
    label_mode = 'categorical',
    batch_size=32,
    image_size = (256,256)
)
Validation=tf.keras.utils.image_dataset_from_directory(
    directory = 'C:/Users/W10/chest_xray/train1',
    labels = 'inferred',
    label_mode = 'categorical',
    batch_size=32,
    image_size = (256,256)
)

# Create Convouloution Nural Network
CNN= tf.keras.Sequential()
# Step 1: Create Convouloution layer
CNN.add(tf.keras.layers.Conv2D(filters=32 , kernel_size= 3 ,activation= 'relu', input_shape= [256,256,3]))
# Step 2: Create Pooling layer
CNN.add(tf.keras.layers.MaxPool2D(pool_size=2,strides= 2))
'''
CNN.add(tf.keras.layers.Conv2D(filters=32 , kernel_size= 3 ,activation= 'relu', input_shape= [256,256,3]))¶
CNN.add(tf.keras.layers.MaxPool2D(pool_size=2,strides= 2))
'''
#Step 3: Flattening
CNN.add(tf.keras.layers.Flatten())

#Step 4: Full Connection
CNN.add(tf.keras.layers.Dense(units= 512,activation='relu'))
#step 5: Output layer
CNN.add(tf.keras.layers.Dense(units=2 , activation = 'sigmoid'))
CNN.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#print(Test.shape)
print(Train)
print(Validation)

CNN.fit(x=Train, validation_data=Validation , epochs= 4  )

import numpy as np
test_image=tf.keras.utils.load_img('C:/Users/W10/chest_xray/train1/NORMAL/IM-0115-0001.jpeg' ,target_size= (256,256))

test_image = tf.keras.utils.img_to_array(test_image)
test_image= np.expand_dims(test_image, axis=0)
Result= CNN.predict(test_image)
print(Result)

test_image=tf.keras.utils.load_img('C:/Users/W10/chest_xray/train1/PNEUMONIA/person1_bacteria_1' ,target_size= (256,256))

test_image = tf.keras.utils.img_to_array(test_image)
test_image= np.expand_dims(test_image, axis=0)
Result= CNN.predict(test_image)
print(Result)