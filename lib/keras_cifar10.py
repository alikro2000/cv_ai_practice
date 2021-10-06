import tensorflow as tf
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from pathlib import Path

#Load data set
# https://www.tensorflow.org/api_docs/python/tf/keras/datasets/cifar10/load_data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#Normalize data to range 0 to 1.
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

#Break the labels into Categories
# https://www.tensorflow.org/api_docs/python/tf/keras/utils/to_categorical
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

#Decalre & train the model
#A little bit of explanation: https://www.pyimagesearch.com/2018/12/31/keras-conv2d-and-convolutional-layers/
#On Conv2D: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D?hl=vi
#ON Activations: https://www.tensorflow.org/api_docs/python/tf/keras/activations?hl=vi
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3), padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

print('Model configured, Buidling...')

#Build the model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print('Model built. Training the model...')

model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=30,
    validation_data=(x_test, y_test),
    shuffle=True
)

print('Model training complete. Saving the output...')

#Save the model as JSON
model_structure = model.to_json()
f = Path("./../models/model_structure.json")
f.write_text(model_structure)

print('Saving weights...')

#Save the Neural Network's trained weights
model.save_weights('./../models/model_weights.h5')

print('DONE.')