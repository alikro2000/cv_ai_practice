import tensorflow as tf
import cv2 as cv
from keras.models import model_from_json
from keras.datasets import cifar10

# load json and create model
json_file = open('./../models/model_structure.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("./../models/model_weights.h5")
print("Loaded model from disk")
 
# Load evaluation data from a dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_test = x_test.astype('float32')
x_test /= 255
y_test = tf.keras.utils.to_categorical(y_test, 10)
print("Dataset loaded.")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(x_test, y_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

#Evaluate one image
# img = cv.imread('f22.jpeg')
# category = loaded_model.predict_on_batch(img)
# cv.imshow('Category: ', img)

cv.waitKey(0)