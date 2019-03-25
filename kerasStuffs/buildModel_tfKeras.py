# from __future__ import print_function
'''
import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
'''
import tensorflow as tf
from tensorflow import keras
from scipy import ndimage
from sklearn.utils import shuffle
import math
import cv2
import numpy as np
import os

batch_size = 32
num_classes = 2
epochs = 30
# input image dimensions
img_x, img_y = 128,128

# List of folders where the images of the digits are located
input_folder_list = ["0", "1"]

# Init empty arrays to store image, language used and digit shown in the image
input_data = []
labels = []

# Traverse through all files in the language folder and read data
for folder in input_folder_list:
    images = os.listdir("drive/Colab/" + folder)
    # lots of preprocessing were required to achieve good accuracy(86%)
    for image in images:
        gray = cv2.imread("drive/Colab/" + folder + "/" + image, cv2.IMREAD_GRAYSCALE)
        # resize and invert colors(i.e convert to image with black background and white number)
        gray = cv2.resize(gray, (img_x, img_y))
        cv2.normalize(gray, gray, 0, 255, cv2.NORM_MINMAX)
        (thresh, gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY| cv2.THRESH_OTSU)

        # append image and correct value
        input_data.append(np.asarray(gray))
        labels.append(folder)

input_data = np.asarray(input_data)
n_sample = len(input_data)
# data = input_data.reshape((n_sample, -1))
x_train = input_data
y_train = labels

# shuffle the training data
x_train, y_train = shuffle(x_train, y_train, random_state=2)

x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 1)

# convert the data to the right type
x_train = x_train.astype('float32')
x_train /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')

# convert class vectors to binary class matrices - this is for use in the
# categorical_crossentropy loss below
y_train = keras.utils.to_categorical(y_train, num_classes)

# save numpy arrays
np.save('drive/Colab/xtrain.npy',x_train)
np.save('drive/Colab/ytrain.npy',y_train)

#load numpy arrays
x_train = np.load('drive/Colab/xtrain.npy')
y_train = np.load('drive/Colab/ytrain.npy')

input_shape = (img_x, img_y, 1)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=input_shape))
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='sigmoid'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy',
             optimizer=tf.train.AdamOptimizer(),
             metrics=['accuracy'])

model.fit(x_train,
         y_train,
         batch_size=batch_size,
         epochs=epochs,
         shuffle=True,verbose=1,validation_split=0.1)


model.save('drive/Colab/my_model.h5')
print("Saved model to disk")


model = tf.keras.models.load_model("drive/Colab/my_model.h5")
print("Loaded model from disk")

model.compile(loss='categorical_crossentropy',
             optimizer=tf.train.AdamOptimizer(),
             metrics=['accuracy'])


predicted = []
images_for_prediction = []
images = os.listdir("drive/Colab/Predict")
for image in images:
    gray = cv2.imread("drive/Colab/Predict/" + image, cv2.IMREAD_GRAYSCALE)
    # same preprocessing as above
    gray = cv2.resize(gray, (img_x, img_y))  # Resize all images to a fixed resolution
    cv2.normalize(gray, gray, 0, 255, cv2.NORM_MINMAX)
	(thresh, gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY| cv2.THRESH_OTSU)
    images_for_prediction.append(np.asarray(gray))

    pr = model.predict_classes(gray.reshape((1, img_x, img_y, 1)), verbose=0)
    predicted.append(pr[0])

images_for_prediction = np.asarray(images_for_prediction)
predicted = np.asarray(predicted)


n = len(predicted)
for i in range(0, n, 1):
    lan_index = predicted[i]
    cv2.imwrite("drive//Colab//Predicted//" + str(lan_index) + " " + str(i) + ".jpg", images_for_prediction[i])

print("done with predictions")
print("check the folder named as Predicted")
print("Thank You")
